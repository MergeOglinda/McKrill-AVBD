using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Stage V. The CPU oracle is checked against the reference solver (one iteration in the colored order runs exactly predict, constraint
    /// warm start and one primal sweep, as far as the body poses go), and the GPU kernel is checked against the oracle.
    /// </summary>
    public class PrimalTests
    {
        const int Capacity = 64;
        const float PositionTolerance = 2e-3f;
        const float RotationTolerance = 2e-4f;

        // ---- the oracle against the reference solver --------------------------------------------------------

        [Test]
        public void Oracle_MatchesTheReferenceSolver_OnTheRope()
        {
            var solver = new ReferenceSolver();
            AvbdScenes.Rope(solver);
            ConstraintTestScenes.RandomizeJointState(solver, new System.Random(3));
            CheckOracleAgainstReference(solver, "rope");
        }

        [Test]
        public void Oracle_MatchesTheReferenceSolver_OnTheSpring()
        {
            var solver = new ReferenceSolver();
            AvbdScenes.Spring(solver);
            CheckOracleAgainstReference(solver, "spring");
        }

        [Test]
        public void Oracle_MatchesTheReferenceSolver_OnRandomScenes()
        {
            float biggestMove = 0f;
            for (int seed = 1; seed <= 12; seed++)
            {
                var solver = new ReferenceSolver();
                ConstraintTestScenes.RandomLattice(solver, seed);
                biggestMove = math.max(biggestMove, CheckOracleAgainstReference(solver, "random scene " + seed));
            }
            Assert.Greater(biggestMove, 1e-3f, "the primal sweep must move bodies, or the comparison proves nothing");
        }

        // Returns the largest distance a body moved in the primal sweep (after the predict stage).
        static float CheckOracleAgainstReference(ReferenceSolver solver, string what)
        {
            AvbdSolverSettings settings = ZUpSettings();
            solver.Params.dt = settings.dt;
            solver.Params.gravity = settings.gravity;
            solver.Params.iterations = 1;
            solver.Params.mode = GaussSeidelMode.Colored;

            SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);
            solver.Step();

            CpuBodyArrays oracle = scene.bodies;
            CpuConstraintInit.Run(oracle, scene.constraints, scene.constraintCount, settings); // the start pose is still the pose
            CpuPredict.Run(oracle, settings, scene.bodyCount);
            ConstraintTopology topology = scene.BuildTopology();

            var predicted = new float3[scene.bodyCount];
            for (int s = 0; s < scene.bodyCount; s++)
                predicted[s] = oracle.State[s].position.xyz;

            int failures = CpuPrimal.Sweep(oracle, scene.constraints, topology, settings);
            Assert.AreEqual(solver.Stats.solveFailures, failures, what + ": failed solves");

            float biggestMove = 0f;
            for (int s = 0; s < scene.bodyCount; s++)
            {
                AvbdBody b = solver.Bodies[s];
                string name = what + ", body " + s + (b.IsDynamic ? "" : " (static)");
                AssertNear(b.PositionLin, oracle.State[s].position.xyz, PositionTolerance, name + " position");
                AssertNear(b.PositionAng.value, oracle.State[s].rotation, RotationTolerance, name + " rotation");
                biggestMove = math.max(biggestMove, math.length(oracle.State[s].position.xyz - predicted[s]));
            }

            // The warm-started constraint state matches too (the reference leaves it after the dual update, so only C0 is comparable).
            int ci = 0;
            foreach (AvbdForce force in solver.Forces)
            {
                if (!(force is AvbdJoint j))
                {
                    ci += force is AvbdSpring ? 1 : 0;
                    continue;
                }
                AssertNear(j.C0Lin, scene.constraints[ci].c0Lin.xyz, 1e-4f, what + ", joint " + ci + " C0Lin");
                AssertNear(j.C0Ang, scene.constraints[ci].c0Ang.xyz, 1e-4f, what + ", joint " + ci + " C0Ang");
                ci++;
            }
            return biggestMove;
        }

        // ---- GPU --------------------------------------------------------------------------------------------

        [Test]
        public void Gpu_MatchesTheOracle_OnTheRope()
        {
            RequireCompute();
            var solver = new ReferenceSolver();
            AvbdScenes.Rope(solver);
            ConstraintTestScenes.RandomizeJointState(solver, new System.Random(8));
            CheckGpuAgainstOracle(solver, "rope");
        }

        [Test]
        public void Gpu_MatchesTheOracle_OnTheSpring()
        {
            RequireCompute();
            var solver = new ReferenceSolver();
            AvbdScenes.Spring(solver);
            CheckGpuAgainstOracle(solver, "spring");
        }

        [Test]
        public void Gpu_MatchesTheOracle_OnRandomScenes()
        {
            RequireCompute();
            for (int seed = 1; seed <= 6; seed++)
            {
                var solver = new ReferenceSolver();
                ConstraintTestScenes.RandomLattice(solver, 100 + seed);
                CheckGpuAgainstOracle(solver, "random scene " + seed);
            }
        }

        static void CheckGpuAgainstOracle(ReferenceSolver reference, string what)
        {
            AvbdSolverSettings settings = ZUpSettings();
            SceneArrays scene = ConstraintTestScenes.FromReference(reference, Capacity);

            // Oracle: constraint warm start, predict, one primal sweep.
            CpuBodyArrays oracle = scene.bodies;
            CpuConstraintInit.Run(oracle, scene.constraints, scene.constraintCount, settings);
            ConstraintTopology expectedTopology = scene.BuildTopology();

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 128))
            {
                for (int i = 0; i < scene.descriptors.Count; i++)
                    Assert.AreEqual(i, buffers.Spawn(scene.descriptors[i]).Slot);
                buffers.ApplyEdits();

                // The initialized records go to the GPU as they are: the warm start kernel belongs to stage D.
                for (int i = 0; i < scene.constraintCount; i++)
                    Assert.AreEqual(i, constraints.AddRecord(scene.constraints[i]));
                constraints.Rebuild();

                ConstraintTopology topology = constraints.Topology;
                Assert.AreEqual(expectedTopology.ColorCount, topology.ColorCount, what + ": colors");
                CollectionAssert.AreEqual(
                    new ArraySegment<uint>(expectedTopology.ColorBodies, 0, expectedTopology.ColorBodyCount),
                    new ArraySegment<uint>(topology.ColorBodies, 0, topology.ColorBodyCount), what + ": color buckets");

                var gpuRanges = new BodyRangeRecord[Capacity];
                buffers.Range.GetData(gpuRanges);
                for (int s = 0; s < scene.bodyCount; s++)
                    Assert.IsTrue(math.all(expectedTopology.Ranges[s].range == gpuRanges[s].range), what + ": range of slot " + s);

                var solver = new AvbdSolver(buffers, settings, null, constraints);
                solver.DispatchPredict();
                solver.DispatchPrimalSweep();

                CpuPredict.Run(oracle, settings, scene.bodyCount);
                int failures = CpuPrimal.Sweep(oracle, scene.constraints, expectedTopology, settings);
                Assert.AreEqual(0, failures, what + ": the oracle must not fail a solve");
                Assert.AreEqual(0, constraints.ReadSolveFailures(), what + ": GPU solve failures");

                var state = new BodyStateRecord[Capacity];
                buffers.State.GetData(state);
                for (int s = 0; s < scene.bodyCount; s++)
                {
                    string name = what + ", slot " + s;
                    AssertNear(oracle.State[s].position, state[s].position, PositionTolerance, name + " position");
                    AssertNear(oracle.State[s].rotation, state[s].rotation, RotationTolerance, name + " rotation");
                    AssertNear(oracle.State[s].startPosition, state[s].startPosition, 1e-5f, name + " start position");
                }
            }
        }

        // ---- readable GPU cases -------------------------------------------------------------------------------

        [Test]
        public void Gpu_AddJointAndSpring_CompleteTheRecordsAndTheSchedule()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var constraints = new AvbdConstraintBuffers(buffers, 8))
            {
                BodyHandle fixedBody = buffers.Spawn(BodyDescriptor.Box(new float3(1f, 2f, 3f), 0f, new float3(0f, 0f, 0f)));
                BodyHandle a = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(4f, 0f, 0f)));
                BodyHandle b = buffers.Spawn(BodyDescriptor.Box(new float3(2f, 1f, 1f), 1f, new float3(4f, 3f, 4f)));

                int joint = constraints.AddJoint(fixedBody, a, new float3(1f, 0f, 0f), new float3(-1f, 0f, 0f));
                int pin = constraints.AddJoint(BodyHandle.Invalid, b, new float3(4f, 3f, 5f), float3.zero, 1000f, 0f, 50f);
                int spring = constraints.AddSpring(a, b, float3.zero, float3.zero, 20f); // rest = current distance
                Assert.AreEqual(new[] { 0, 1, 2 }, new[] { joint, pin, spring });
                Assert.IsTrue(constraints.HasPendingChanges);

                constraints.Rebuild();
                Assert.IsFalse(constraints.HasPendingChanges);

                var records = new ConstraintRecord[8];
                constraints.Records.GetData(records);

                // Joint: torque arm is the squared length of the summed sizes: (1,2,3) + (1,1,1) = (2,3,4).
                Assert.AreEqual(29f, records[0].c0Lin.w, 1e-5f);
                Assert.AreEqual((uint)fixedBody.Slot, records[0].ident.x);
                Assert.AreEqual((uint)a.Slot, records[0].ident.y);
                Assert.AreEqual(ConstraintFlags.HardLinear, ConstraintFlagLayout.FlagsOf(records[0].ident.z));

                // World joint: no body A, torque arm from body B alone, finite stiffness and a fracture threshold.
                Assert.AreEqual(AvbdConst.Invalid, records[1].ident.x);
                Assert.AreEqual(6f, records[1].c0Lin.w, 1e-5f);
                Assert.AreEqual(ConstraintFlags.Fracturable, ConstraintFlagLayout.FlagsOf(records[1].ident.z));
                Assert.AreEqual(1000f, records[1].anchorA.w);
                Assert.AreEqual(50f, records[1].c0Ang.w);

                // Spring: the rest length is the distance between the anchors: (0,3,4) -> 5.
                Assert.AreEqual(ConstraintKind.Spring, ConstraintFlagLayout.KindOf(records[2].ident.z));
                Assert.AreEqual(5f, records[2].anchorB.w, 1e-5f);
                Assert.AreEqual(20f, records[2].anchorA.w);

                // Schedule: the fixed body is not solved; a and b are joined by the spring, so they get two colors.
                Assert.AreEqual(2, constraints.ColorCount);
                var ranges = new BodyRangeRecord[16];
                buffers.Range.GetData(ranges);
                Assert.AreEqual(0u, ranges[fixedBody.Slot].range.y);
                Assert.AreEqual(2u, ranges[a.Slot].range.y);
                Assert.AreEqual(2u, ranges[b.Slot].range.y);
            }
        }

        [Test]
        public void Gpu_Step_BalancesInertiaAgainstAPenaltyJoint_AndUpdatesItsDualVariables()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var constraints = new AvbdConstraintBuffers(buffers, 8))
            {
                AvbdSolverSettings settings = AvbdSolverSettings.Default; // gravity (0, -10, 0), dt = 1/60
                settings.iterations = 1;

                // A unit box (mass 1) at the pin point, hard joint with a starting penalty of 1e4. With rB = 0 the joint has no torque
                // arm, so the problem is a 1D spring on y.
                BodyHandle ball = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 5f, 0f)));
                buffers.ApplyEdits();
                ConstraintRecord pin = ConstraintRecordFactory.Joint(
                    AvbdConst.Invalid, (uint)ball.Slot, new float3(0f, 5f, 0f), float3.zero,
                    float.PositiveInfinity, 0f, float.PositiveInfinity, 1f);
                pin.penaltyLin = new float4(1e4f, 1e4f, 1e4f, 0f);
                constraints.AddRecord(pin);

                new AvbdSolver(buffers, settings, null, constraints).Step();

                var state = new BodyStateRecord[16];
                buffers.State.GetData(state);
                var records = new ConstraintRecord[8];
                constraints.Records.GetData(records);

                // The warm start scales the penalty by gamma: K = 1e4 * 0.999 = 9990.
                // Minimizing m/(2 dt^2) (y - yInertial)^2 + K/2 (y - 5)^2 with yInertial = 5 - g dt^2 gives
                // y = 5 - (m/dt^2)(g dt^2) / (m/dt^2 + K) = 5 - 10 / 13590. Falling freely it would be 5 - 10 / 3600.
                float expectedY = 5f - 10f / 13590f;
                Assert.AreEqual(expectedY, state[ball.Slot].position.y, 1e-4f);
                Assert.AreEqual(0f, state[ball.Slot].position.x, 1e-4f);
                Assert.AreEqual(0f, state[ball.Slot].position.z, 1e-4f);
                Assert.AreEqual(0, constraints.ReadSolveFailures());

                // The dual update afterwards: C = pin - position = (0, 10 / 13590, 0), no stabilization (C0 = 0),
                // lambda = K C, and the penalty grows by |C| * betaLin on the axis that has an error.
                float c = 10f / 13590f;
                Assert.AreEqual(9990f * c, records[0].lambdaLin.y, 3e-2f); // C is a difference of numbers near 5: fp32 error ~5e-7 times K
                Assert.AreEqual(0f, records[0].lambdaLin.x, 1e-3f);
                Assert.AreEqual(9990f + c * 1e4f, records[0].penaltyLin.y, 0.1f);
                Assert.AreEqual(9990f, records[0].penaltyLin.x, 0.1f);
                Assert.AreEqual(0f, records[0].penaltyAng.x, "no angle constraint");
            }
        }

        [Test]
        public void Gpu_StaticBodies_AreNeverMovedByThePrimalStage()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var constraints = new AvbdConstraintBuffers(buffers, 8))
            {
                BodyHandle anchor = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 0f, new float3(0f, 10f, 0f)));
                BodyHandle block = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 6f, 0f)));
                constraints.AddSpring(anchor, block, float3.zero, float3.zero, 100f, 2f);

                new AvbdSolver(buffers, AvbdSolverSettings.Default, null, constraints).Step();

                var state = new BodyStateRecord[16];
                buffers.State.GetData(state);
                Assert.AreEqual(10f, state[anchor.Slot].position.y);
                Assert.Greater(state[block.Slot].position.y, 6f - 1e-6f, "the spring is stretched (rest 2, length 4) and pulls the block up");
            }
        }

        // ---- helpers --------------------------------------------------------------------------------------------

        static AvbdSolverSettings ZUpSettings()
        {
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            settings.gravity = new float3(0f, 0f, -10f); // the upstream scenes are z-up
            settings.iterations = 1;
            return settings;
        }

        static void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }

        // Absolute tolerance scaled by the magnitude, so that large coordinates are not held to a tiny absolute error.
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
