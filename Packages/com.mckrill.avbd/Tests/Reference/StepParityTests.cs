using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Stage U (velocity update) and the complete step. The scenes are the upstream rope and spring scenes, which the reference solver runs
    /// without any contact (nothing reaches the ground). The reference solver runs in <see cref="GaussSeidelMode.Colored"/>, the order the GPU
    /// uses. Without collisions the GPU step equals the reference step except for floating-point rounding, so the checks use tolerances
    /// that are far below any logic error (a wrong sign or order gives errors of order 1 within a step).
    /// </summary>
    public class StepParityTests
    {
        const int Capacity = 64;

        // Position and rotation are compared relative to max(1, |value|). Velocities are displacements divided by dt = 1/60, so they carry
        // 60 times the rounding error of the positions and get their own, larger tolerance.
        const float OraclePositionTolerance = 5e-3f;
        const float GpuPositionTolerance = 1e-2f;
        const float OracleVelocityTolerance = 1e-1f;
        const float GpuVelocityTolerance = 2e-1f;

        // ---- stage U on its own --------------------------------------------------------------------------------

        [Test]
        public void Oracle_Velocity_IsTheDisplacementOverDt_AndKeepsThePreviousVelocity()
        {
            var bodies = new CpuBodyArrays(8);
            var queue = new BodyEditQueue(8);
            var commands = new List<EditCommand>();

            BodyDescriptor moving = BodyDescriptor.Box(new float3(1f), 1f, new float3(1f, 2f, 3f));
            moving.linearVelocity = new float3(4f, 5f, 6f);
            BodyHandle h = queue.Spawn(moving);
            BodyHandle fixedBody = queue.Spawn(BodyDescriptor.Box(new float3(1f), 0f, float3.zero));
            BodyDescriptor kin = BodyDescriptor.Box(new float3(1f), 1f, float3.zero);
            kin.kinematic = true;
            kin.linearVelocity = new float3(7f, 0f, 0f);
            BodyHandle k = queue.Spawn(kin);
            queue.Drain(commands);
            bodies.Apply(commands);

            AvbdSolverSettings settings = AvbdSolverSettings.Default;

            // The solver moved the body by (0.1, -0.2, 0.3) and turned it 0.06 rad about z.
            bodies.State[h.Slot].position += new float4(0.1f, -0.2f, 0.3f, 0f);
            float3 axis = new float3(0f, 0f, 1f);
            bodies.State[h.Slot].rotation = math.mul(quaternion.AxisAngle(axis, 0.06f), new quaternion(bodies.State[h.Slot].startRotation)).value;
            // Moving the static and the kinematic body must change nothing either.
            bodies.State[fixedBody.Slot].position += new float4(1f, 0f, 0f, 0f);
            bodies.State[k.Slot].position += new float4(1f, 0f, 0f, 0f);

            CpuVelocity.Run(bodies, settings, 3);

            BodyMotionRecord m = bodies.Motion[h.Slot];
            Assert.AreEqual(6f, m.linearVel.x, 1e-3f);
            Assert.AreEqual(-12f, m.linearVel.y, 1e-3f);
            Assert.AreEqual(18f, m.linearVel.z, 1e-3f);
            Assert.AreEqual(2f * math.sin(0.03f) * 60f, m.angularVel.z, 1e-3f);
            Assert.AreEqual(0f, m.angularVel.x, 1e-4f);
            Assert.AreEqual(4f, m.prevLinearVel.x, 1e-6f, "the previous velocity is the velocity before the update");
            Assert.AreEqual(6f, m.prevLinearVel.z, 1e-6f);

            Assert.AreEqual(0f, bodies.Motion[fixedBody.Slot].linearVel.x, "static bodies keep their (zero) velocity");
            Assert.AreEqual(7f, bodies.Motion[k.Slot].linearVel.x, "kinematic bodies keep their velocity");
        }

        [Test]
        public void Gpu_Velocity_MatchesTheOracle_OnRandomBodies()
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

                // Pretend the solver has moved every body: new positions and rotations, and different previous velocities.
                for (int slot = 0; slot < buffers.HighWaterMark; slot++)
                {
                    oracle.State[slot].position += new float4(V(rng, 0.5f), 0f);
                    oracle.State[slot].rotation = quaternion.EulerXYZ(F(rng, 0f, 6f), F(rng, 0f, 6f), F(rng, 0f, 6f)).value;
                    oracle.Motion[slot].prevLinearVel = new float4(V(rng, 5f), 0f);
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
                    AssertNear(oracle.Motion[slot].linearVel, motion[slot].linearVel, 1e-4f, what + " linear velocity");
                    AssertNear(oracle.Motion[slot].angularVel, motion[slot].angularVel, 1e-4f, what + " angular velocity");
                    AssertNear(oracle.Motion[slot].prevLinearVel, motion[slot].prevLinearVel, 1e-6f, what + " previous velocity");
                }
            }
        }

        // ---- the oracle step against the reference solver -----------------------------------------------------

        [Test]
        public void Oracle_Step_MatchesTheReferenceSolver_OnTheRope()
        {
            var solver = new ReferenceSolver();
            AvbdScenes.Rope(solver);
            CheckOracleOverSteps(solver, 30, "rope");
        }

        [Test]
        public void Oracle_Step_MatchesTheReferenceSolver_OnTheSpring()
        {
            var solver = new ReferenceSolver();
            AvbdScenes.Spring(solver);
            CheckOracleOverSteps(solver, 120, "spring");
        }

        static void CheckOracleOverSteps(ReferenceSolver reference, int steps, string what)
        {
            AvbdSolverSettings settings = Settings();
            ConfigureReference(reference, settings);

            SceneArrays scene = ConstraintTestScenes.FromReference(reference, Capacity);
            ConstraintTopology topology = scene.BuildTopology();

            for (int step = 1; step <= steps; step++)
            {
                reference.Step();
                Assert.AreEqual(0, reference.Stats.contacts, what + ": the scene must not touch anything (step " + step + ")");

                int failures = CpuStep.Run(scene.bodies, scene.constraints, scene.constraintCount, topology, scene.bodyCount, settings);
                Assert.AreEqual(reference.Stats.solveFailures, failures, what + ": failed solves in step " + step);

                for (int s = 0; s < scene.bodyCount; s++)
                    AssertBodyMatches(reference.Bodies[s], scene.bodies.State[s], scene.bodies.Motion[s],
                        OraclePositionTolerance, OracleVelocityTolerance, what + ", step " + step + ", body " + s);
            }
        }

        // ---- the GPU against the reference solver -------------------------------------------------------------

        [Test]
        public void Gpu_Step_MatchesTheReferenceSolver_OnTheRope()
        {
            RequireCompute();
            var solver = new ReferenceSolver();
            AvbdScenes.Rope(solver);
            float moved = CheckGpuOverSteps(solver, 30, "rope");
            Assert.Greater(moved, 1f, "the rope must swing, or the comparison proves nothing");
        }

        [Test]
        public void Gpu_Step_MatchesTheReferenceSolver_OnTheSpring()
        {
            RequireCompute();
            var solver = new ReferenceSolver();
            AvbdScenes.Spring(solver);
            float moved = CheckGpuOverSteps(solver, 120, "spring");
            Assert.Greater(moved, 0.5f, "the block must bounce on its spring, or the comparison proves nothing");
        }

        // Runs the reference solver and the GPU solver in lockstep and compares every body after every step. Returns the largest distance
        // any dynamic body got from its starting position.
        static float CheckGpuOverSteps(ReferenceSolver reference, int steps, string what)
        {
            AvbdSolverSettings settings = Settings();
            ConfigureReference(reference, settings);
            SceneArrays scene = ConstraintTestScenes.FromReference(reference, Capacity);
            var start = new float3[scene.bodyCount];
            for (int s = 0; s < scene.bodyCount; s++)
                start[s] = reference.Bodies[s].PositionLin;

            float moved = 0f;
            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, Capacity))
            {
                // The records go in as the reference scene made them: penalties and multipliers start at zero.
                for (int i = 0; i < scene.descriptors.Count; i++)
                    Assert.AreEqual(i, buffers.Spawn(scene.descriptors[i]).Slot);
                for (int i = 0; i < scene.constraintCount; i++)
                    Assert.AreEqual(i, constraints.AddRecord(scene.constraints[i]));

                var solver = new AvbdSolver(buffers, settings, null, constraints);
                var state = new BodyStateRecord[Capacity];
                var motion = new BodyMotionRecord[Capacity];
                var previousVelocity = new float4[Capacity];

                for (int step = 1; step <= steps; step++)
                {
                    reference.Step();
                    Assert.AreEqual(0, reference.Stats.contacts, what + ": the scene must not touch anything (step " + step + ")");
                    solver.Step();

                    Assert.AreEqual(0, constraints.ReadSolveFailures(), what + ": GPU solve failures in step " + step);
                    buffers.State.GetData(state);
                    buffers.Motion.GetData(motion);

                    for (int s = 0; s < scene.bodyCount; s++)
                    {
                        string name = what + ", step " + step + ", body " + s;
                        AvbdBody b = reference.Bodies[s];
                        AssertBodyMatches(b, state[s], motion[s], GpuPositionTolerance, GpuVelocityTolerance, name);

                        if (!b.IsDynamic)
                            continue;

                        // Stage U itself, exactly: the velocity is the displacement of this step over dt, and the previous velocity is last step's.
                        float dt = settings.dt;
                        AssertNear((state[s].position.xyz - state[s].startPosition.xyz) / dt, motion[s].linearVel.xyz, 1e-4f, name + " velocity = displacement / dt");
                        AssertNear(previousVelocity[s], motion[s].prevLinearVel, 1e-6f, name + " previous velocity");
                        previousVelocity[s] = motion[s].linearVel;

                        moved = math.max(moved, math.length(state[s].position.xyz - start[s]));
                    }
                }
            }
            return moved;
        }

        // ---- the step without constraints, and with a changing set of bodies --------------------------------

        [Test]
        public void Gpu_Step_WithoutConstraints_MatchesTheReferenceSolver()
        {
            RequireCompute();
            AvbdSolverSettings settings = Settings();

            // Two free boxes, one spinning and drifting. No ground, so nothing touches.
            var reference = new ReferenceSolver();
            reference.Clear();
            AvbdBody a = reference.AddBody(new float3(1f, 2f, 0.5f), 1f, 0.5f, new float3(0f, 0f, 10f), new float3(1f, 0.5f, 2f));
            a.VelocityAng = new float3(1f, -2f, 0.5f);
            reference.AddBody(new float3(0.5f), 2f, 0.5f, new float3(20f, 0f, 10f));
            ConfigureReference(reference, settings);

            SceneArrays scene = ConstraintTestScenes.FromReference(reference, Capacity);

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var solver = new AvbdSolver(buffers, settings)) // no constraints: the solver makes its own empty set
            {
                foreach (BodyDescriptor d in scene.descriptors)
                    buffers.Spawn(d);

                var state = new BodyStateRecord[Capacity];
                var motion = new BodyMotionRecord[Capacity];
                for (int step = 1; step <= 30; step++)
                {
                    reference.Step();
                    solver.Step();
                    buffers.State.GetData(state);
                    buffers.Motion.GetData(motion);
                    for (int s = 0; s < scene.bodyCount; s++)
                        AssertBodyMatches(reference.Bodies[s], state[s], motion[s], 1e-4f, 1e-2f, "free body " + s + ", step " + step);
                }
            }
        }

        [Test]
        public void Gpu_Step_PicksUpBodiesSpawnedAndRemovedBetweenSteps()
        {
            RequireCompute();
            AvbdSolverSettings settings = AvbdSolverSettings.Default; // gravity (0, -10, 0)

            using (var buffers = new AvbdBodyBuffers(16))
            using (var solver = new AvbdSolver(buffers, settings))
            {
                BodyHandle first = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 10f, 0f)));
                solver.Step();

                // A body added after the first step must be scheduled by the next one.
                BodyHandle second = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(5f, 10f, 0f)));
                solver.Step();

                var state = new BodyStateRecord[16];
                buffers.State.GetData(state);
                Assert.Less(state[second.Slot].position.y, 10f - 1e-3f, "the new body falls");
                Assert.Less(state[first.Slot].position.y, 10f - 1e-3f);

                // A removed body is left alone, and the others keep falling.
                float firstY = state[first.Slot].position.y;
                float secondY = state[second.Slot].position.y;
                buffers.Remove(first);
                solver.Step();
                buffers.State.GetData(state);
                Assert.AreEqual(firstY, state[first.Slot].position.y, "a removed body is not updated");
                Assert.Less(state[second.Slot].position.y, secondY - 1e-4f, "the remaining body keeps falling");
            }
        }

        // ---- helpers -----------------------------------------------------------------------------------------

        static AvbdSolverSettings Settings()
        {
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            settings.gravity = new float3(0f, 0f, -10f); // the upstream scenes are z-up
            return settings; // 10 iterations, like the reference solver's default
        }

        static void ConfigureReference(ReferenceSolver reference, in AvbdSolverSettings settings)
        {
            reference.Params.dt = settings.dt;
            reference.Params.gravity = settings.gravity;
            reference.Params.iterations = settings.iterations;
            reference.Params.mode = GaussSeidelMode.Colored;
        }

        static void AssertBodyMatches(AvbdBody expected, BodyStateRecord state, BodyMotionRecord motion, float tolerance, float velocityTolerance, string what)
        {
            AssertNear(expected.PositionLin, state.position.xyz, tolerance, what + " position");
            AssertNear(expected.PositionAng.value, state.rotation, tolerance, what + " rotation");
            AssertNear(expected.VelocityLin, motion.linearVel.xyz, velocityTolerance, what + " linear velocity");
            AssertNear(expected.VelocityAng, motion.angularVel.xyz, velocityTolerance, what + " angular velocity");
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
