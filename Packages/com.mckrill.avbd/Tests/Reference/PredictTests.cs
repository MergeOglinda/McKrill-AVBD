using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Stage P. The CPU oracle is checked against the reference solver (which, with zero iterations, runs exactly
    /// the predict phase), and the GPU kernel is checked against the oracle.
    /// </summary>
    public class PredictTests
    {
        const int Capacity = 128;

        // ---- settings ---------------------------------------------------------------------------------

        [Test]
        public void Settings_DefaultsAreValid_AndBadValuesAreRejected()
        {
            Assert.DoesNotThrow(() => AvbdSolverSettings.Default.Validate());

            AvbdSolverSettings s = AvbdSolverSettings.Default;
            s.dt = 0f;
            Assert.Throws<ArgumentException>(() => s.Validate());

            s = AvbdSolverSettings.Default;
            s.gravity = new float3(float.NaN, 0f, 0f);
            Assert.Throws<ArgumentException>(() => s.Validate());

            s = AvbdSolverSettings.Default;
            s.iterations = -1;
            Assert.Throws<ArgumentException>(() => s.Validate());

            s = AvbdSolverSettings.Default;
            s.alpha = 1.5f;
            Assert.Throws<ArgumentException>(() => s.Validate());

            s = AvbdSolverSettings.Default;
            s.penaltyMax = 0.5f;
            Assert.Throws<ArgumentException>(() => s.Validate());
        }

        // ---- the oracle against the reference solver -------------------------------------------------------

        [Test]
        public void Oracle_MatchesTheReferenceSolverPredictPhase()
        {
            var rng = new System.Random(5);
            const int count = 24;

            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            settings.gravity = new float3(0f, 0f, -10f);

            var solver = new ReferenceSolver();
            solver.Params.iterations = 0; // body init only: the predict phase, then velocities
            solver.Params.dt = settings.dt;
            solver.Params.gravity = settings.gravity;

            var queue = new BodyEditQueue(Capacity);
            var oracle = new CpuBodyArrays(Capacity);
            var spawned = new List<EditCommand>();
            var bodies = new List<AvbdBody>();

            for (int i = 0; i < count; i++)
            {
                float3 size = new float3(F(rng, 0.3f, 2f), F(rng, 0.3f, 2f), F(rng, 0.3f, 2f));
                float density = i % 5 == 0 ? 0f : F(rng, 0.2f, 2f);
                float3 position = new float3(i * 100f, F(rng, -5f, 5f), F(rng, -5f, 5f)); // far apart: no contacts
                quaternion rotation = quaternion.EulerXYZ(F(rng, 0f, 6f), F(rng, 0f, 6f), F(rng, 0f, 6f));
                float3 linear = V(rng, 5f);
                float3 angular = V(rng, 5f);
                float3 previous = V(rng, 5f);

                AvbdBody body = solver.AddBody(size, density, 0.5f, position, linear);
                body.PositionAng = rotation;
                body.VelocityAng = angular;
                body.PrevVelocityLin = previous;
                bodies.Add(body);

                BodyDescriptor d = BodyDescriptor.Box(size, density, position);
                d.rotation = rotation;
                d.linearVelocity = linear;
                d.angularVelocity = angular;
                BodyHandle h = queue.Spawn(d);
                Assert.AreEqual(i, h.Slot);
                spawned.Clear();
                queue.Drain(spawned);
                oracle.Apply(spawned);
                oracle.Motion[i].prevLinearVel = new float4(previous, 0f);
            }

            solver.Step();
            CpuPredict.Run(oracle, settings, count);

            for (int i = 0; i < count; i++)
            {
                AvbdBody b = bodies[i];
                string what = "body " + i + (b.IsDynamic ? "" : " (static)");

                AssertNear(b.PositionLin, oracle.State[i].position.xyz, 1e-5f, what + " position");
                AssertNear(b.PositionAng.value, oracle.State[i].rotation, 1e-6f, what + " rotation");

                if (b.IsDynamic)
                {
                    AssertNear(b.InitialLin, oracle.State[i].startPosition.xyz, 1e-5f, what + " start position");
                    AssertNear(b.InertialLin, oracle.Inertial[i].position.xyz, 1e-5f, what + " inertial position");
                    AssertNear(b.InertialAng.value, oracle.Inertial[i].rotation, 1e-6f, what + " inertial rotation");
                }
            }
        }

        // ---- readable GPU cases --------------------------------------------------------------------------------

        [Test]
        public void Gpu_AtRestWithNoAcceleration_KeepsThePoseAndFallsOnlyInTheTarget()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            {
                AvbdSolverSettings settings = AvbdSolverSettings.Default;
                settings.dt = 0.1f;
                settings.gravity = new float3(0f, -10f, 0f);

                BodyHandle h = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 10f, 0f)));
                buffers.ApplyEdits();

                new AvbdSolver(buffers, settings).DispatchPredict();

                var state = new BodyStateRecord[16];
                var inertial = new BodyInertialRecord[16];
                buffers.State.GetData(state);
                buffers.Inertial.GetData(inertial);

                int s = h.Slot;
                Assert.AreEqual(10f, state[s].position.y, 1e-5f, "previous velocity equals velocity, so the warm start adds no gravity");
                Assert.AreEqual(10f, state[s].startPosition.y, 1e-5f);
                Assert.AreEqual(9.9f, inertial[s].position.y, 1e-5f, "inertial target = x + g dt^2");
            }
        }

        [Test]
        public void Gpu_FullAdaptiveWeight_AppliesFullGravityToThePose()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            {
                AvbdSolverSettings settings = AvbdSolverSettings.Default;
                settings.dt = 0.1f;
                settings.gravity = new float3(0f, -10f, 0f);

                BodyDescriptor d = BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 10f, 0f));
                d.linearVelocity = new float3(0f, -1f, 0f);
                BodyHandle h = buffers.Spawn(d);
                buffers.ApplyEdits();

                // Previous velocity 0, velocity (0, -1, 0): acceleration exactly g, so the weight is 1.
                var motion = new BodyMotionRecord[16];
                buffers.Motion.GetData(motion);
                motion[h.Slot].prevLinearVel = float4.zero;
                buffers.Motion.SetData(motion);

                new AvbdSolver(buffers, settings).DispatchPredict();

                var state = new BodyStateRecord[16];
                buffers.State.GetData(state);
                // y + v dt + g dt^2 = 10 - 0.1 - 0.1
                Assert.AreEqual(9.8f, state[h.Slot].position.y, 1e-5f);
            }
        }

        [Test]
        public void Gpu_StaticAndKinematicBodies_AreNotPredicted()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            {
                BodyDescriptor stat = BodyDescriptor.Box(new float3(1f), 0f, new float3(1f, 2f, 3f));
                stat.linearVelocity = new float3(5f, 5f, 5f);
                BodyDescriptor kin = BodyDescriptor.Box(new float3(1f), 1f, new float3(4f, 5f, 6f));
                kin.kinematic = true;
                kin.linearVelocity = new float3(5f, 5f, 5f);
                BodyHandle hs = buffers.Spawn(stat);
                BodyHandle hk = buffers.Spawn(kin);
                buffers.ApplyEdits();

                var before = new BodyStateRecord[16];
                buffers.State.GetData(before);

                new AvbdSolver(buffers, AvbdSolverSettings.Default).DispatchPredict();

                var after = new BodyStateRecord[16];
                var inertial = new BodyInertialRecord[16];
                buffers.State.GetData(after);
                buffers.Inertial.GetData(inertial);

                foreach (BodyHandle h in new[] { hs, hk })
                {
                    Assert.IsTrue(math.all(before[h.Slot].position == after[h.Slot].position), "pose of slot " + h.Slot);
                    Assert.IsTrue(math.all(before[h.Slot].startPosition == after[h.Slot].startPosition), "start pose of slot " + h.Slot);
                    Assert.IsTrue(math.all(inertial[h.Slot].position == float4.zero), "inertial target of slot " + h.Slot + " must stay untouched");
                }
            }
        }

        // ---- GPU against the oracle ----------------------------------------------------------------------------

        [Test]
        public void Gpu_MatchesTheOracle_OnRandomBodies()
        {
            RequireCompute();
            var rng = new System.Random(77);

            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            settings.gravity = new float3(1f, -9f, 2f); // not axis aligned, to exercise the dot product

            var oracle = new CpuBodyArrays(Capacity);
            var applied = new List<EditCommand>();
            var handles = new List<BodyHandle>();

            using (var buffers = new AvbdBodyBuffers(Capacity))
            {
                for (int i = 0; i < 100; i++)
                {
                    BodyDescriptor d = BodyDescriptor.Box(new float3(F(rng, 0.3f, 2f), F(rng, 0.3f, 2f), F(rng, 0.3f, 2f)),
                        i % 7 == 0 ? 0f : F(rng, 0.2f, 2f), V(rng, 30f));
                    d.rotation = quaternion.EulerXYZ(F(rng, 0f, 6f), F(rng, 0f, 6f), F(rng, 0f, 6f));
                    d.linearVelocity = V(rng, 5f);
                    d.angularVelocity = V(rng, 5f);
                    d.kinematic = i % 11 == 0;
                    handles.Add(buffers.Spawn(d));
                }
                for (int i = 3; i < handles.Count; i += 9)
                    buffers.Remove(handles[i]); // leave free slots inside the used range

                buffers.ApplyEdits(applied);
                oracle.Apply(applied);

                // Distinct previous velocities, so the adaptive weight varies between 0 and 1.
                for (int slot = 0; slot < buffers.HighWaterMark; slot++)
                {
                    float3 v = oracle.Motion[slot].linearVel.xyz;
                    float3 prev = rng.Next(3) == 0 ? v : v - settings.gravity * settings.dt * F(rng, -0.5f, 1.5f);
                    oracle.Motion[slot].prevLinearVel = new float4(prev, 0f);
                }
                buffers.Motion.SetData(oracle.Motion);

                CpuPredict.Run(oracle, settings, buffers.HighWaterMark);
                new AvbdSolver(buffers, settings).DispatchPredict();

                var state = new BodyStateRecord[Capacity];
                var inertial = new BodyInertialRecord[Capacity];
                buffers.State.GetData(state);
                buffers.Inertial.GetData(inertial);

                for (int slot = 0; slot < Capacity; slot++)
                {
                    string what = "slot " + slot;
                    AssertNear(oracle.State[slot].position, state[slot].position, 1e-5f, what + " position");
                    AssertNear(oracle.State[slot].rotation, state[slot].rotation, 1e-6f, what + " rotation");
                    AssertNear(oracle.State[slot].startPosition, state[slot].startPosition, 1e-5f, what + " start position");
                    AssertNear(oracle.State[slot].startRotation, state[slot].startRotation, 1e-6f, what + " start rotation");
                    AssertNear(oracle.Inertial[slot].position, inertial[slot].position, 1e-5f, what + " inertial position");
                    AssertNear(oracle.Inertial[slot].rotation, inertial[slot].rotation, 1e-6f, what + " inertial rotation");
                }
            }
        }

        // ---- helpers --------------------------------------------------------------------------------------------

        static void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }

        static float F(System.Random rng, float min, float max) => min + (float)rng.NextDouble() * (max - min);

        static float3 V(System.Random rng, float range) => new float3(F(rng, -range, range), F(rng, -range, range), F(rng, -range, range));

        // Relative tolerance: floating-point results on the GPU may differ in the last bits (fused multiply-add, rsqrt).
        static void AssertNear(float3 expected, float3 actual, float tol, string what)
        {
            for (int k = 0; k < 3; k++)
                Assert.AreEqual(expected[k], actual[k], tol * math.max(1f, math.abs(expected[k])), what + "[" + k + "]");
        }

        static void AssertNear(float4 expected, float4 actual, float tol, string what)
        {
            for (int k = 0; k < 4; k++)
                Assert.AreEqual(expected[k], actual[k], tol * math.max(1f, math.abs(expected[k])), what + "[" + k + "]");
        }
    }
}
