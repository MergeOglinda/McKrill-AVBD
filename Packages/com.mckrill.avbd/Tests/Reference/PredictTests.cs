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
                PoseMirror.Assert(buffers, "after Predict");
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
                PoseMirror.Assert(buffers, "after Predict");
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
                PoseMirror.Assert(buffers, "after Predict");

                new AvbdSolver(buffers, AvbdSolverSettings.Default).DispatchPredict();

                var after = new BodyStateRecord[16];
                var inertial = new BodyInertialRecord[16];
                buffers.State.GetData(after);
                PoseMirror.Assert(buffers, "after Predict");
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
                PoseMirror.Assert(buffers, "after Predict");
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

        // ---- explosions ------------------------------------------------------------------------------------------

        static CpuBodyArrays OneBody(float3 size, float density, float3 position, quaternion rotation, float3 linear, float3 angular)
        {
            var queue = new BodyEditQueue(8);
            var oracle = new CpuBodyArrays(8);
            BodyDescriptor d = BodyDescriptor.Box(size, density, position);
            d.rotation = rotation;
            d.linearVelocity = linear;
            d.angularVelocity = angular;
            queue.Spawn(d);
            var commands = new List<EditCommand>();
            queue.Drain(commands);
            oracle.Apply(commands);
            return oracle;
        }

        static CpuExplosion Blast(float3 center, float radius, float impulse, uint seed = 0u) =>
            new CpuExplosion { center = center, radius = radius, impulse = impulse, seed = seed };

        [Test]
        public void Explosion_Hash_IsDeterministic_AndTheRandomNumbersStayInRange()
        {
            Assert.AreEqual(CpuExplosionMath.Hash(12345u), CpuExplosionMath.Hash(12345u));
            Assert.AreNotEqual(CpuExplosionMath.Hash(1u), CpuExplosionMath.Hash(2u));

            float3 min = new float3(float.MaxValue), max = new float3(float.MinValue), sum = float3.zero;
            const int n = 4000;
            for (uint slot = 0; slot < n; slot++)
            {
                float3 r = CpuExplosionMath.Random(slot, 3u);
                min = math.min(min, r);
                max = math.max(max, r);
                sum += r;
            }
            Assert.GreaterOrEqual(min.x, -1f);
            Assert.Less(max.x, 1f);
            Assert.IsTrue(math.all(min < -0.95f) && math.all(max > 0.95f), "the numbers cover the range");
            Assert.IsTrue(math.all(math.abs(sum / n) < 0.05f), "and have no bias");
            Assert.IsFalse(math.all(CpuExplosionMath.Random(5u, 1u) == CpuExplosionMath.Random(5u, 2u)), "another explosion, another kick");
        }

        [Test]
        public void Explosion_ChangesTheVelocityByTheImpulseOverTheMass_AlongTheLineFromTheCenter_AndFallsOffToZero()
        {
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            // A 2 kg box (density 2, 1 m side) 3 m to the +x side of the center, with a radius of 6 m: falloff (1 - 3/6)^2 = 0.25.
            CpuBodyArrays bodies = OneBody(new float3(1f), 2f, new float3(3f, 0f, 0f), quaternion.identity, new float3(1f, 2f, 3f), float3.zero);
            bodies.Motion[0].prevLinearVel = new float4(0.5f, 0.5f, 0.5f, 0f);

            CpuPredict.Run(bodies, settings, 1, new[] { Blast(float3.zero, 6f, 80f) }, 1);

            float expected = 80f * 0.25f / 2f; // 10 m/s
            AssertNear(new float3(1f + expected, 2f, 3f), bodies.Motion[0].linearVel.xyz, 1e-5f, "velocity");
            AssertNear(new float3(0.5f + expected, 0.5f, 0.5f), bodies.Motion[0].prevLinearVel.xyz, 1e-5f, "previous velocity moves with it, so the warm start sees no blast");

            // At the edge (and outside) nothing happens.
            bodies = OneBody(new float3(1f), 2f, new float3(6f, 0f, 0f), quaternion.identity, new float3(1f, 2f, 3f), float3.zero);
            CpuPredict.Run(bodies, settings, 1, new[] { Blast(float3.zero, 6f, 80f) }, 1);
            AssertNear(new float3(1f, 2f, 3f), bodies.Motion[0].linearVel.xyz, 0f, "at the radius");
            AssertNear(float3.zero, bodies.Motion[0].angularVel.xyz, 0f, "no spin either");
        }

        [Test]
        public void Explosion_HasNoUpwardBias_AndLighterBodiesFlyFaster()
        {
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            // Below the center: pushed down. Beside it: pushed sideways, not up.
            CpuBodyArrays below = OneBody(new float3(1f), 1f, new float3(0f, -2f, 0f), quaternion.identity, float3.zero, float3.zero);
            CpuPredict.Run(below, settings, 1, new[] { Blast(float3.zero, 10f, 100f) }, 1);
            Assert.Less(below.Motion[0].linearVel.y, 0f);
            Assert.AreEqual(0f, below.Motion[0].linearVel.x, 1e-6f);

            CpuBodyArrays beside = OneBody(new float3(1f), 1f, new float3(2f, 0f, 0f), quaternion.identity, float3.zero, float3.zero);
            CpuPredict.Run(beside, settings, 1, new[] { Blast(float3.zero, 10f, 100f) }, 1);
            Assert.AreEqual(0f, beside.Motion[0].linearVel.y, 1e-6f);

            CpuBodyArrays heavy = OneBody(new float3(1f), 4f, new float3(2f, 0f, 0f), quaternion.identity, float3.zero, float3.zero);
            CpuPredict.Run(heavy, settings, 1, new[] { Blast(float3.zero, 10f, 100f) }, 1);
            Assert.AreEqual(beside.Motion[0].linearVel.x / 4f, heavy.Motion[0].linearVel.x, 1e-4f, "four times the mass, a quarter of the speed");
        }

        [Test]
        public void Explosion_GivesASmallSpin_ThatTheSlotAndTheSeedDetermine()
        {
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            float3 size = new float3(1f, 0.5f, 0.5f);
            CpuBodyArrays a = OneBody(size, 1f, new float3(2f, 0f, 0f), quaternion.identity, float3.zero, float3.zero);
            CpuPredict.Run(a, settings, 1, new[] { Blast(float3.zero, 10f, 100f, 4u) }, 1);
            float3 spin = a.Motion[0].angularVel.xyz;
            Assert.Greater(math.length(spin), 0f);

            // The same body and explosion again: the same spin.
            CpuBodyArrays again = OneBody(size, 1f, new float3(2f, 0f, 0f), quaternion.identity, float3.zero, float3.zero);
            CpuPredict.Run(again, settings, 1, new[] { Blast(float3.zero, 10f, 100f, 4u) }, 1);
            AssertNear(spin, again.Motion[0].angularVel.xyz, 0f, "repeatable");

            // Another explosion: another spin.
            CpuBodyArrays other = OneBody(size, 1f, new float3(2f, 0f, 0f), quaternion.identity, float3.zero, float3.zero);
            CpuPredict.Run(other, settings, 1, new[] { Blast(float3.zero, 10f, 100f, 5u) }, 1);
            Assert.Greater(math.length(spin - other.Motion[0].angularVel.xyz), 0f);

            // The angular impulse is bounded by the offset: |offset| <= sqrt(3) * kick * radius of the body, |impulse| = 100 * falloff.
            float falloff = 0.8f * 0.8f;
            float bound = math.sqrt(3f) * CpuExplosionMath.Kick * 0.5f * math.length(size) * 100f * falloff;
            float3 moments = a.Mass[0].momentMass.xyz;
            Assert.LessOrEqual(math.length(spin * moments), bound * 1.001f, "the spin is a small kick, not a blow");
        }

        [Test]
        public void Explosion_AtTheCenter_PicksADirection_AndStaysFinite()
        {
            CpuBodyArrays bodies = OneBody(new float3(1f), 1f, float3.zero, quaternion.identity, float3.zero, float3.zero);
            CpuPredict.Run(bodies, AvbdSolverSettings.Default, 1, new[] { Blast(float3.zero, 5f, 10f) }, 1);

            float3 v = bodies.Motion[0].linearVel.xyz;
            Assert.IsTrue(math.all(math.isfinite(v)));
            Assert.AreEqual(10f, math.length(v), 1e-4f, "the full impulse, along a hashed direction");
        }

        [Test]
        public void Explosion_DoesNotTouchStaticOrKinematicBodies()
        {
            var queue = new BodyEditQueue(8);
            var oracle = new CpuBodyArrays(8);
            BodyDescriptor stat = BodyDescriptor.Box(new float3(1f), 0f, new float3(1f, 0f, 0f));
            BodyDescriptor kin = BodyDescriptor.Box(new float3(1f), 1f, new float3(2f, 0f, 0f));
            kin.kinematic = true;
            queue.Spawn(stat);
            queue.Spawn(kin);
            var commands = new List<EditCommand>();
            queue.Drain(commands);
            oracle.Apply(commands);

            CpuPredict.Run(oracle, AvbdSolverSettings.Default, 2, new[] { Blast(float3.zero, 10f, 100f) }, 1);

            Assert.IsTrue(math.all(oracle.Motion[0].linearVel == float4.zero));
            Assert.IsTrue(math.all(oracle.Motion[1].linearVel == float4.zero));
            Assert.IsTrue(math.all(oracle.Motion[1].angularVel == float4.zero));
        }

        [Test]
        public void Explosion_NoExplosions_LeavesThePredictPhaseUnchanged()
        {
            var rng = new System.Random(3);
            CpuBodyArrays plain = OneBody(new float3(1f), 1f, new float3(1f, 2f, 3f), quaternion.EulerXYZ(0.3f, 0.2f, 0.1f), V(rng, 3f), V(rng, 3f));
            CpuBodyArrays none = OneBody(new float3(1f), 1f, new float3(1f, 2f, 3f), quaternion.EulerXYZ(0.3f, 0.2f, 0.1f), plain.Motion[0].linearVel.xyz, plain.Motion[0].angularVel.xyz);

            CpuPredict.Run(plain, AvbdSolverSettings.Default, 1);
            CpuPredict.Run(none, AvbdSolverSettings.Default, 1, new CpuExplosion[4], 0);

            Assert.IsTrue(math.all(plain.State[0].position == none.State[0].position));
            Assert.IsTrue(math.all(plain.Inertial[0].position == none.Inertial[0].position));
        }

        [Test]
        public void Gpu_Explosions_MatchTheOracle_OnRandomBodies()
        {
            RequireCompute();
            var rng = new System.Random(91);

            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            var oracle = new CpuBodyArrays(Capacity);
            var applied = new List<EditCommand>();
            var handles = new List<BodyHandle>();

            using (var buffers = new AvbdBodyBuffers(Capacity))
            {
                for (int i = 0; i < 100; i++)
                {
                    BodyDescriptor d = BodyDescriptor.Box(new float3(F(rng, 0.3f, 2f), F(rng, 0.3f, 2f), F(rng, 0.3f, 2f)),
                        i % 7 == 0 ? 0f : F(rng, 0.2f, 2f), V(rng, 12f));
                    d.rotation = quaternion.EulerXYZ(F(rng, 0f, 6f), F(rng, 0f, 6f), F(rng, 0f, 6f));
                    d.linearVelocity = V(rng, 5f);
                    d.angularVelocity = V(rng, 5f);
                    d.kinematic = i % 11 == 0;
                    handles.Add(buffers.Spawn(d));
                }
                for (int i = 3; i < handles.Count; i += 9)
                    buffers.Remove(handles[i]);

                buffers.ApplyEdits(applied);
                oracle.Apply(applied);
                for (int slot = 0; slot < buffers.HighWaterMark; slot++)
                    oracle.Motion[slot].prevLinearVel = new float4(oracle.Motion[slot].linearVel.xyz - V(rng, 0.2f), 0f);
                buffers.Motion.SetData(oracle.Motion);

                using (var solver = new AvbdSolver(buffers, settings))
                {
                    // Overlapping explosions: a body in the reach of several is hit by each in turn, with the seeds 0, 1 and 2 that the solver hands out.
                    var blasts = new[]
                    {
                        Blast(new float3(0f, 0f, 0f), 9f, 40f, 0u),
                        Blast(new float3(4f, -2f, 3f), 7f, -25f, 1u),
                        Blast(new float3(-3f, 3f, 1f), 12f, 60f, 2u),
                    };
                    foreach (CpuExplosion e in blasts)
                        Assert.IsTrue(solver.Explode(e.center, e.radius, e.impulse));
                    Assert.AreEqual(3, solver.PendingExplosionCount);

                    CpuPredict.Run(oracle, settings, buffers.HighWaterMark, blasts, blasts.Length);
                    solver.DispatchPredict();
                    Assert.AreEqual(0, solver.PendingExplosionCount, "the queue is emptied by the step that applies it");
                }

                var state = new BodyStateRecord[Capacity];
                var motion = new BodyMotionRecord[Capacity];
                var inertial = new BodyInertialRecord[Capacity];
                buffers.State.GetData(state);
                PoseMirror.Assert(buffers, "after Predict");
                buffers.Motion.GetData(motion);
                buffers.Inertial.GetData(inertial);

                int moved = 0;
                for (int slot = 0; slot < Capacity; slot++)
                {
                    string what = "slot " + slot;
                    AssertNear(oracle.Motion[slot].linearVel, motion[slot].linearVel, 1e-4f, what + " velocity");
                    AssertNear(oracle.Motion[slot].prevLinearVel, motion[slot].prevLinearVel, 1e-4f, what + " previous velocity");
                    AssertNear(oracle.Motion[slot].angularVel, motion[slot].angularVel, 1e-3f, what + " angular velocity");
                    AssertNear(oracle.State[slot].position, state[slot].position, 1e-4f, what + " position");
                    AssertNear(oracle.State[slot].rotation, state[slot].rotation, 1e-5f, what + " rotation");
                    AssertNear(oracle.Inertial[slot].position, inertial[slot].position, 1e-4f, what + " inertial position");
                    if (math.any(oracle.Motion[slot].linearVel != float4.zero))
                        moved++;
                }
                Assert.Greater(moved, 20, "the scene is meant to have bodies in reach");
            }
        }

        [Test]
        public void Gpu_ExplosionRelativeToABody_GoesOffAtThatPointOfTheBody_AsAWorldExplosionThere()
        {
            RequireCompute();
            var rng = new System.Random(17);
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            var oracle = new CpuBodyArrays(Capacity);
            var applied = new List<EditCommand>();

            using (var buffers = new AvbdBodyBuffers(Capacity))
            {
                BodyHandle anchor = BodyHandle.Invalid;
                for (int i = 0; i < 40; i++)
                {
                    BodyDescriptor d = BodyDescriptor.Box(new float3(F(rng, 0.5f, 1.5f), F(rng, 0.5f, 1.5f), F(rng, 0.5f, 1.5f)), F(rng, 0.5f, 2f), V(rng, 6f));
                    d.rotation = quaternion.EulerXYZ(F(rng, 0f, 6f), F(rng, 0f, 6f), F(rng, 0f, 6f));
                    BodyHandle h = buffers.Spawn(d);
                    if (i == 5)
                        anchor = h;
                }
                buffers.ApplyEdits(applied);
                oracle.Apply(applied);

                // The center is a point of a turned body that is not at its own center of mass: the world position follows from its pose.
                float3 local = new float3(0.4f, -0.3f, 0.2f);
                BodyStateRecord a = oracle.State[anchor.Slot];
                float3 world = a.position.xyz + math.rotate(new quaternion(a.rotation), local);

                using (var solver = new AvbdSolver(buffers, settings))
                {
                    Assert.IsTrue(solver.Explode(anchor, local, 8f, 30f));
                    Assert.AreEqual(1, solver.PendingExplosionCount);
                    var blasts = new[] { Blast(world, 8f, 30f, 0u) };

                    CpuPredict.Run(oracle, settings, buffers.HighWaterMark, blasts, blasts.Length);
                    solver.DispatchPredict();
                    Assert.AreEqual(0, solver.PendingExplosionCount);
                }

                var motion = new BodyMotionRecord[Capacity];
                buffers.Motion.GetData(motion);
                int moved = 0;
                for (int slot = 0; slot < buffers.HighWaterMark; slot++)
                {
                    AssertNear(oracle.Motion[slot].linearVel, motion[slot].linearVel, 1e-3f, "slot " + slot + " velocity");
                    AssertNear(oracle.Motion[slot].angularVel, motion[slot].angularVel, 1e-2f, "slot " + slot + " angular velocity");
                    if (math.any(oracle.Motion[slot].linearVel != float4.zero))
                        moved++;
                }
                Assert.Greater(moved, 3, "the scene is meant to have bodies in reach");
            }
        }

        [Test]
        public void Gpu_ExplosionRelativeToABody_IsDropped_ForAStaleHandle_AndForABodyRemovedBeforeTheStep()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var solver = new AvbdSolver(buffers, AvbdSolverSettings.Default))
            {
                BodyHandle gone = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 0f, 0f)));
                BodyHandle other = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(1f, 0f, 0f)));
                BodyHandle removedLater = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(2f, 0f, 0f)));
                buffers.ApplyEdits();
                Assert.IsTrue(buffers.Remove(gone));

                Assert.IsFalse(solver.Explode(gone, float3.zero, 5f, 10f), "a removed body");
                Assert.IsFalse(solver.Explode(BodyHandle.Invalid, float3.zero, 5f, 10f), "no body");
                Assert.AreEqual(0, solver.PendingExplosionCount);
                Assert.Throws<ArgumentException>(() => solver.Explode(other, new float3(float.NaN, 0f, 0f), 5f, 10f));
                Assert.Throws<ArgumentException>(() => solver.Explode(other, float3.zero, 0f, 10f));

                // Queued while it was alive, removed before the step applies it: nothing is hit (not even the bodies next to it).
                Assert.IsTrue(solver.Explode(removedLater, float3.zero, 5f, 10f));
                Assert.IsTrue(buffers.Remove(removedLater));
                buffers.ApplyEdits();
                solver.DispatchPredict();

                var motion = new BodyMotionRecord[Capacity];
                buffers.Motion.GetData(motion);
                for (int slot = 0; slot < buffers.HighWaterMark; slot++)
                    Assert.IsTrue(math.all(motion[slot].linearVel == float4.zero), "slot " + slot + " must not have been hit");
            }
        }

        [Test]
        public void Gpu_Explosion_IsOneShot_AndTheNextPredictLeavesTheVelocityAlone()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var solver = new AvbdSolver(buffers, AvbdSolverSettings.Default))
            {
                BodyHandle h = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(2f, 0f, 0f)));
                buffers.ApplyEdits();

                Assert.IsTrue(solver.Explode(float3.zero, 4f, 8f));
                solver.DispatchPredict();

                var motion = new BodyMotionRecord[16];
                buffers.Motion.GetData(motion);
                float falloff = 0.5f * 0.5f;
                Assert.AreEqual(8f * falloff / 1f, motion[h.Slot].linearVel.x, 1e-4f, "impulse * falloff / mass, away from the center");
                Assert.AreEqual(0f, motion[h.Slot].linearVel.y, 1e-4f, "no upward bias");
                float3 first = motion[h.Slot].linearVel.xyz;

                solver.DispatchPredict();
                buffers.Motion.GetData(motion);
                AssertNear(first, motion[h.Slot].linearVel.xyz, 0f, "no explosion queued: the second predict does not touch the velocity");
            }
        }

        [Test]
        public void Explode_RejectsBadValues_AndDropsWhatDoesNotFit()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var solver = new AvbdSolver(buffers, AvbdSolverSettings.Default))
            {
                Assert.Throws<ArgumentException>(() => solver.Explode(new float3(float.NaN, 0f, 0f), 1f, 1f));
                Assert.Throws<ArgumentException>(() => solver.Explode(float3.zero, 0f, 1f));
                Assert.Throws<ArgumentException>(() => solver.Explode(float3.zero, -1f, 1f));
                Assert.Throws<ArgumentException>(() => solver.Explode(float3.zero, 1f, float.PositiveInfinity));
                Assert.AreEqual(0, solver.PendingExplosionCount);

                for (int i = 0; i < AvbdSolver.MaxExplosionsPerStep; i++)
                    Assert.IsTrue(solver.Explode(float3.zero, 1f, 1f));
                Assert.IsFalse(solver.Explode(float3.zero, 1f, 1f), "the queue is full");
                Assert.AreEqual(AvbdSolver.MaxExplosionsPerStep, solver.PendingExplosionCount);
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
