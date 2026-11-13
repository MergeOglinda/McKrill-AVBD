using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Stage U (velocity update) and the whole step. The CPU oracles are checked against the reference solver over many steps, and
    /// the GPU solver is checked against the reference solver on the rope and the spring scene.
    ///
    /// Tolerances come from how sensitive the trajectories are to rounding noise: upstream's solver (C++, run on both scenes twice, once with
    /// every initial position nudged by 1e-6) drifts by at most 3e-5 in position, 6e-6 in rotation and 1.4e-4 in velocity after 60 steps on
    /// the rope, and the spring scene does not amplify at all. The bounds below are 30 to 70 times that.
    /// </summary>
    public class VelocityTests
    {
        const int Capacity = 64;
        const float PositionTolerance = 2e-3f;
        const float RotationTolerance = 2e-4f;
        const float VelocityTolerance = 1e-2f; // relative to max(1, |v|)

        static readonly int[] Checkpoints = { 1, 2, 10, 30, 60 };

        // ---- the velocity oracle -------------------------------------------------------------------------------

        [Test]
        public void Oracle_DerivesTheVelocityFromTheDisplacement()
        {
            CpuBodyArrays bodies = Spawn(out BodyHandle h, BodyDescriptor.Box(new float3(1f), 1f, float3.zero));
            AvbdSolverSettings s = AvbdSolverSettings.Default;

            bodies.Motion[h.Slot].linearVel = new float4(7f, 8f, 9f, 0f);
            bodies.State[h.Slot].startPosition = new float4(1f, 2f, 3f, 0f);
            bodies.State[h.Slot].position = new float4(1f + 3f * s.dt, 2f - 6f * s.dt, 3f, 0f);

            CpuVelocity.Run(bodies, s, 1);

            AssertNear(new float3(3f, -6f, 0f), bodies.Motion[h.Slot].linearVel.xyz, 1e-4f, "velocity");
            AssertNear(new float3(7f, 8f, 9f), bodies.Motion[h.Slot].prevLinearVel.xyz, 0f, "previous velocity is the old velocity");
            Assert.AreEqual(0f, bodies.Motion[h.Slot].linearVel.w);
        }

        [Test]
        public void Oracle_DerivesTheAngularVelocityFromTheRotation()
        {
            CpuBodyArrays bodies = Spawn(out BodyHandle h, BodyDescriptor.Box(new float3(1f), 1f, float3.zero));
            AvbdSolverSettings s = AvbdSolverSettings.Default;

            // A rotation of 0.1 rad about z. The difference is 2 sin(angle / 2) along the axis, as upstream computes it.
            bodies.State[h.Slot].startRotation = new float4(0f, 0f, 0f, 1f);
            bodies.State[h.Slot].rotation = new float4(0f, 0f, math.sin(0.05f), math.cos(0.05f));

            CpuVelocity.Run(bodies, s, 1);

            float expected = 2f * math.sin(0.05f) / s.dt;
            AssertNear(new float3(0f, 0f, expected), bodies.Motion[h.Slot].angularVel.xyz, 1e-4f, "angular velocity");
        }

        [Test]
        public void Oracle_LeavesStaticAndKinematicBodiesAlone_ButSnapshotsTheirVelocity()
        {
            var bodies = new CpuBodyArrays(8);
            var queue = new BodyEditQueue(8);
            var commands = new List<EditCommand>();

            BodyDescriptor stat = BodyDescriptor.Box(new float3(1f), 0f, new float3(1f, 2f, 3f));
            stat.linearVelocity = new float3(1f, 0f, 0f);
            BodyDescriptor kin = BodyDescriptor.Box(new float3(1f), 1f, new float3(4f, 5f, 6f));
            kin.kinematic = true;
            kin.linearVelocity = new float3(0f, 2f, 0f);
            BodyHandle hs = queue.Spawn(stat);
            BodyHandle hk = queue.Spawn(kin);
            queue.Drain(commands);
            bodies.Apply(commands);

            // Pretend something moved them: the oracle must not turn that into a velocity.
            bodies.State[hs.Slot].position = new float4(9f, 9f, 9f, 0f);
            bodies.State[hk.Slot].position = new float4(9f, 9f, 9f, 0f);

            CpuVelocity.Run(bodies, AvbdSolverSettings.Default, 2);

            AssertNear(new float3(1f, 0f, 0f), bodies.Motion[hs.Slot].linearVel.xyz, 0f, "static velocity");
            AssertNear(new float3(1f, 0f, 0f), bodies.Motion[hs.Slot].prevLinearVel.xyz, 0f, "static previous velocity");
            AssertNear(new float3(0f, 2f, 0f), bodies.Motion[hk.Slot].linearVel.xyz, 0f, "kinematic velocity");
            AssertNear(new float3(0f, 2f, 0f), bodies.Motion[hk.Slot].prevLinearVel.xyz, 0f, "kinematic previous velocity");
        }

        [Test]
        public void Oracle_DoesNotTouchFreeSlots()
        {
            var bodies = new CpuBodyArrays(4);
            bodies.Motion[2].linearVel = new float4(5f, 5f, 5f, 0f);
            BodyMotionRecord before = bodies.Motion[2];

            CpuVelocity.Run(bodies, AvbdSolverSettings.Default, 4);

            Assert.AreEqual(before, bodies.Motion[2], "a free slot keeps its record");
        }

        // ---- the whole step: oracle against the reference solver ----------------------------------------------------

        [TestCase("rope", 60)]
        [TestCase("spring", 60)]
        public void Oracle_Step_MatchesTheReferenceSolver(string scene, int steps)
        {
            ReferenceSolver solver = Colored(scene, out AvbdSolverSettings settings);
            SceneArrays arrays = ConstraintTestScenes.FromReference(solver, Capacity);
            ConstraintTopology topology = arrays.BuildTopology();

            for (int step = 1; step <= steps; step++)
            {
                solver.Step();
                int failures = CpuStep.Run(arrays.bodies, arrays.constraints, arrays.constraintCount, topology, arrays.bodyCount, settings);
                Assert.AreEqual(0, failures, scene + " step " + step + ": the oracle must not fail a solve");

                if (System.Array.IndexOf(Checkpoints, step) < 0)
                    continue;
                for (int s = 0; s < arrays.bodyCount; s++)
                    AssertBodyNear(solver.Bodies[s], arrays.bodies, s, scene + " step " + step + " slot " + s);
            }
        }

        [Test]
        public void Oracle_Step_FollowsTheClosedFormOfAFreeFall()
        {
            // No constraints: the primal update puts the body on its inertial target, so velocity grows by g dt every step and the
            // position follows y_n = y_0 + g dt^2 n (n + 1) / 2.
            AvbdSolverSettings s = AvbdSolverSettings.Default;
            CpuBodyArrays bodies = Spawn(out BodyHandle h, BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 10f, 0f)));
            var topology = new ConstraintTopology();
            topology.Build(new ConstraintRecord[0], 0, new[] { true }, 1);
            System.Array.Copy(topology.Ranges, bodies.Range, 1);

            const int steps = 30;
            for (int n = 1; n <= steps; n++)
                CpuStep.Run(bodies, new ConstraintRecord[0], 0, topology, 1, s);

            float g = s.gravity.y;
            Assert.AreEqual(steps * g * s.dt, bodies.Motion[h.Slot].linearVel.y, 5e-3f, "velocity");
            Assert.AreEqual(10f + g * s.dt * s.dt * steps * (steps + 1) / 2f, bodies.State[h.Slot].position.y, 2e-3f, "position");
        }

        // ---- GPU -----------------------------------------------------------------------------------------------

        [Test]
        public void Gpu_Velocity_MatchesTheOracle_OnRandomBodies()
        {
            RequireCompute();
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            var rng = new System.Random(77);
            var handles = new List<BodyHandle>();
            var applied = new List<EditCommand>();

            using (var buffers = new AvbdBodyBuffers(Capacity))
            {
                var oracle = new CpuBodyArrays(Capacity);
                for (int i = 0; i < 40; i++)
                {
                    BodyDescriptor d = BodyDescriptor.Box(new float3(F(rng, 0.3f, 2f)), i % 6 == 0 ? 0f : F(rng, 0.2f, 2f), V(rng, 20f));
                    d.linearVelocity = V(rng, 5f);
                    d.angularVelocity = V(rng, 5f);
                    d.kinematic = i % 9 == 0;
                    handles.Add(buffers.Spawn(d));
                }
                for (int i = 2; i < handles.Count; i += 7)
                    buffers.Remove(handles[i]); // free slots inside the used range
                buffers.ApplyEdits(applied);
                oracle.Apply(applied);

                // Poses a step has moved: the start pose and the current pose differ, and both are unit quaternions.
                for (int slot = 0; slot < buffers.HighWaterMark; slot++)
                {
                    oracle.State[slot].startPosition = new float4(V(rng, 20f), 0f);
                    oracle.State[slot].position = oracle.State[slot].startPosition + new float4(V(rng, 0.2f), 0f);
                    oracle.State[slot].startRotation = quaternion.EulerXYZ(V(rng, 3f)).value;
                    oracle.State[slot].rotation = math.mul(quaternion.EulerXYZ(V(rng, 0.1f)), new quaternion(oracle.State[slot].startRotation)).value;
                }
                buffers.State.SetData(oracle.State);
                buffers.Motion.SetData(oracle.Motion);

                CpuVelocity.Run(oracle, settings, buffers.HighWaterMark);
                new AvbdSolver(buffers, settings).DispatchVelocity();

                var motion = new BodyMotionRecord[Capacity];
                buffers.Motion.GetData(motion);
                for (int slot = 0; slot < Capacity; slot++)
                {
                    string what = "slot " + slot;
                    AssertNear(oracle.Motion[slot].linearVel, motion[slot].linearVel, 1e-3f, what + " linear velocity");
                    AssertNear(oracle.Motion[slot].angularVel, motion[slot].angularVel, 1e-3f, what + " angular velocity");
                    AssertNear(oracle.Motion[slot].prevLinearVel, motion[slot].prevLinearVel, 1e-6f, what + " previous velocity");
                }
            }
        }

        [Test]
        public void Gpu_Step_FollowsTheClosedFormOfAFreeFall_WithoutConstraintBuffers()
        {
            RequireCompute();
            AvbdSolverSettings s = AvbdSolverSettings.Default;
            using (var buffers = new AvbdBodyBuffers(8))
            using (var solver = new AvbdSolver(buffers, s))
            {
                BodyHandle h = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 10f, 0f)));

                const int steps = 30;
                for (int n = 0; n < steps; n++)
                    solver.Step();

                AssertFall(buffers, h, 10f, steps, s);
            }
        }

        [Test]
        public void Gpu_Step_SolvesABodySpawnedAfterTheFirstStep()
        {
            RequireCompute();
            AvbdSolverSettings s = AvbdSolverSettings.Default;
            using (var buffers = new AvbdBodyBuffers(8))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4))
            {
                BodyHandle first = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 10f, 0f)));
                var solver = new AvbdSolver(buffers, s, null, constraints);
                for (int n = 0; n < 3; n++)
                    solver.Step();

                BodyHandle second = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(5f, 10f, 0f)));
                const int steps = 10;
                for (int n = 0; n < steps; n++)
                    solver.Step();

                AssertFall(buffers, first, 10f, 3 + steps, s);
                AssertFall(buffers, second, 10f, steps, s);
            }
        }

        [Test]
        public void Gpu_Step_KeepsTheVelocityOfStaticAndKinematicBodies()
        {
            RequireCompute();
            AvbdSolverSettings s = AvbdSolverSettings.Default;
            using (var buffers = new AvbdBodyBuffers(8))
            using (var solver = new AvbdSolver(buffers, s))
            {
                BodyDescriptor stat = BodyDescriptor.Box(new float3(1f), 0f, new float3(1f, 2f, 3f));
                stat.linearVelocity = new float3(1f, 0f, 0f);
                BodyDescriptor kin = BodyDescriptor.Box(new float3(1f), 1f, new float3(4f, 5f, 6f));
                kin.kinematic = true;
                kin.linearVelocity = new float3(0f, 2f, 0f);
                BodyHandle hs = buffers.Spawn(stat);
                BodyHandle hk = buffers.Spawn(kin);

                for (int n = 0; n < 5; n++)
                    solver.Step();

                var motion = new BodyMotionRecord[8];
                var state = new BodyStateRecord[8];
                buffers.Motion.GetData(motion);
                buffers.State.GetData(state);
                AssertNear(new float3(1f, 0f, 0f), motion[hs.Slot].linearVel.xyz, 1e-6f, "static velocity");
                AssertNear(new float3(0f, 2f, 0f), motion[hk.Slot].linearVel.xyz, 1e-6f, "kinematic velocity");
                AssertNear(new float3(1f, 2f, 3f), state[hs.Slot].position.xyz, 1e-6f, "static position");
                AssertNear(new float3(4f, 5f, 6f), state[hk.Slot].position.xyz, 1e-6f, "kinematic position");
            }
        }

        [TestCase("rope", 60)]
        [TestCase("spring", 60)]
        public void Gpu_Step_MatchesTheReferenceSolver(string scene, int steps)
        {
            RequireCompute();
            ReferenceSolver reference = Colored(scene, out AvbdSolverSettings settings);
            SceneArrays arrays = ConstraintTestScenes.FromReference(reference, Capacity);

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 128))
            {
                for (int i = 0; i < arrays.descriptors.Count; i++)
                    Assert.AreEqual(i, buffers.Spawn(arrays.descriptors[i]).Slot);
                for (int i = 0; i < arrays.constraintCount; i++)
                    Assert.AreEqual(i, constraints.AddRecord(arrays.constraints[i]));

                var solver = new AvbdSolver(buffers, settings, null, constraints);
                var state = new BodyStateRecord[Capacity];
                var motion = new BodyMotionRecord[Capacity];

                for (int step = 1; step <= steps; step++)
                {
                    reference.Step();
                    solver.Step();

                    if (System.Array.IndexOf(Checkpoints, step) < 0)
                        continue;

                    buffers.State.GetData(state);
                    buffers.Motion.GetData(motion);
                    for (int s = 0; s < arrays.bodyCount; s++)
                    {
                        string what = scene + " step " + step + " slot " + s;
                        AvbdBody b = reference.Bodies[s];
                        AssertNear(b.PositionLin, state[s].position.xyz, PositionTolerance, what + " position");
                        AssertNear(b.PositionAng.value, state[s].rotation, RotationTolerance, what + " rotation");
                        AssertNear(b.VelocityLin, motion[s].linearVel.xyz, VelocityTolerance, what + " linear velocity");
                        AssertNear(b.VelocityAng, motion[s].angularVel.xyz, VelocityTolerance, what + " angular velocity");
                        AssertNear(b.PrevVelocityLin, motion[s].prevLinearVel.xyz, VelocityTolerance, what + " previous velocity");
                    }
                    Assert.AreEqual(0, constraints.ReadSolveFailures(), scene + " step " + step + ": GPU solve failures");
                }
            }
        }

        // ---- helpers -------------------------------------------------------------------------------------------

        static ReferenceSolver Colored(string scene, out AvbdSolverSettings settings)
        {
            var solver = new ReferenceSolver();
            switch (scene)
            {
                case "rope": AvbdScenes.Rope(solver); break;
                case "spring": AvbdScenes.Spring(solver); break;
                default: throw new System.ArgumentException("Unknown scene " + scene);
            }

            settings = AvbdSolverSettings.Default;
            settings.gravity = new float3(0f, 0f, -10f); // the upstream scenes are z-up
            solver.Params.dt = settings.dt;
            solver.Params.gravity = settings.gravity;
            solver.Params.iterations = settings.iterations;
            solver.Params.mode = GaussSeidelMode.Colored;
            return solver;
        }

        static CpuBodyArrays Spawn(out BodyHandle handle, BodyDescriptor d)
        {
            var bodies = new CpuBodyArrays(8);
            var queue = new BodyEditQueue(8);
            var commands = new List<EditCommand>();
            handle = queue.Spawn(d);
            queue.Drain(commands);
            bodies.Apply(commands);
            return bodies;
        }

        // The closed form of a free fall that started at rest at height y0 and has run n steps (see Oracle_Step_FollowsTheClosedFormOfAFreeFall).
        static void AssertFall(AvbdBodyBuffers buffers, BodyHandle h, float y0, int n, in AvbdSolverSettings s)
        {
            var state = new BodyStateRecord[buffers.Capacity];
            var motion = new BodyMotionRecord[buffers.Capacity];
            buffers.State.GetData(state);
            buffers.Motion.GetData(motion);

            float g = s.gravity.y;
            Assert.AreEqual(n * g * s.dt, motion[h.Slot].linearVel.y, 5e-3f, "velocity of slot " + h.Slot);
            Assert.AreEqual(y0 + g * s.dt * s.dt * n * (n + 1) / 2f, state[h.Slot].position.y, 2e-3f, "height of slot " + h.Slot);
        }

        static void AssertBodyNear(AvbdBody expected, CpuBodyArrays actual, int slot, string what)
        {
            AssertNear(expected.PositionLin, actual.State[slot].position.xyz, PositionTolerance, what + " position");
            AssertNear(expected.PositionAng.value, actual.State[slot].rotation, RotationTolerance, what + " rotation");
            AssertNear(expected.VelocityLin, actual.Motion[slot].linearVel.xyz, VelocityTolerance, what + " linear velocity");
            AssertNear(expected.VelocityAng, actual.Motion[slot].angularVel.xyz, VelocityTolerance, what + " angular velocity");
            AssertNear(expected.PrevVelocityLin, actual.Motion[slot].prevLinearVel.xyz, VelocityTolerance, what + " previous velocity");
        }

        static void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }

        static float F(System.Random rng, float min, float max) => min + (float)rng.NextDouble() * (max - min);

        static float3 V(System.Random rng, float range) => new float3(F(rng, -range, range), F(rng, -range, range), F(rng, -range, range));

        // Tolerance relative to max(1, |expected|), so that large values are not held to a tiny absolute error.
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
