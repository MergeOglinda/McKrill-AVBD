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

        // A chain of bodies, created in a random order so that the coloring takes few rounds, has three colors, of which the biggest holds more than a
        // third of the bodies: with 12 000 links a color has the 4 lanes of a color of 4096 to 32767 bodies, with 100 000 the 2 lanes of 32768 and up.
        // The lanes of a body share the work differently in each case (entries l, l + lanes, ... of the body's list, and the sums over the lanes).
        [TestCase(12000, 4)]
        [TestCase(100000, 2)]
        public void Gpu_MatchesTheOracle_OnAChainWithColorsThatGetFewerLanes(int links, int lanes)
        {
            RequireCompute();
            var rng = new System.Random(links);
            var solver = new ReferenceSolver();
            solver.Clear();

            var order = new int[links];
            for (int i = 0; i < links; i++)
                order[i] = i;
            for (int i = links - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }

            var bodies = new AvbdBody[links];
            for (int k = 0; k < links; k++)
            {
                int i = order[k]; // body k of the creation order is link i of the chain
                float3 jitter = new float3((float)rng.NextDouble() - 0.5f, (float)rng.NextDouble() - 0.5f, (float)rng.NextDouble() - 0.5f) * 0.3f;
                bodies[i] = solver.AddBody(new float3(1f), 1f, 0.5f, new float3(i * 2f, 0f, 0f) + jitter, float3.zero);
            }
            for (int i = 1; i < links; i++)
                solver.AddJoint(bodies[i - 1], bodies[i], new float3(1f, 0f, 0f), new float3(-1f, 0f, 0f));
            solver.AddJoint(null, bodies[0], new float3(0f, 0f, 0f), new float3(-1f, 0f, 0f)); // the end is pinned
            ConstraintTestScenes.RandomizeJointState(solver, rng);

            CheckGpuAgainstOracle(solver, links + " links", links + 16, links + 16, lanes);
        }

        // With expectedLanes above zero, also checks that the biggest color is of a size the primal sweep gives that many lanes per body. A scene of more than
        // Capacity bodies needs its own capacities, and the reference solver's coloring order (the default order colors differently).
        static void CheckGpuAgainstOracle(ReferenceSolver reference, string what, int capacity = Capacity, int constraintCapacity = 128, int expectedLanes = 0)
        {
            AvbdSolverSettings settings = ZUpSettings();
            SceneArrays scene = ConstraintTestScenes.FromReference(reference, capacity);

            // Oracle: constraint warm start, predict, one primal sweep.
            CpuBodyArrays oracle = scene.bodies;
            CpuConstraintInit.Run(oracle, scene.constraints, scene.constraintCount, settings);
            ConstraintTopology expectedTopology = scene.BuildTopology();

            using (var buffers = new AvbdBodyBuffers(capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, constraintCapacity))
            {
                for (int i = 0; i < scene.descriptors.Count; i++)
                    Assert.AreEqual(i, buffers.Spawn(scene.descriptors[i]).Slot);
                buffers.ApplyEdits();

                if (expectedLanes > 0)
                    constraints.ColoringOrder = GraphColoringOrder.SlotOrder;

                // The initialized records go to the GPU as they are: the warm start kernel belongs to stage D.
                for (int i = 0; i < scene.constraintCount; i++)
                    Assert.AreEqual(i, constraints.AddRecord(scene.constraints[i]));
                constraints.Rebuild();

                // Stage C ran on the GPU: the same colors, and the same bodies in each color (their order inside a color is unspecified).
                Assert.AreEqual(expectedTopology.ColorCount, constraints.ColorCount, what + ": colors");
                uint[] gpuColorBodies = constraints.ReadColorBodies();
                for (int c = 0; c < expectedTopology.ColorCount; c++)
                {
                    Assert.AreEqual(expectedTopology.ColorSize(c), constraints.ColorSize(c), what + ": size of color " + c);
                    var expectedSlots = new uint[expectedTopology.ColorSize(c)];
                    var gpuSlots = new uint[constraints.ColorSize(c)];
                    Array.Copy(expectedTopology.ColorBodies, expectedTopology.ColorOffsets[c], expectedSlots, 0, expectedSlots.Length);
                    Array.Copy(gpuColorBodies, constraints.ColorOffset(c), gpuSlots, 0, gpuSlots.Length);
                    Array.Sort(expectedSlots);
                    Array.Sort(gpuSlots);
                    CollectionAssert.AreEqual(expectedSlots, gpuSlots, what + ": bodies of color " + c);
                }

                var gpuRanges = new BodyRangeRecord[capacity];
                buffers.Range.GetData(gpuRanges);
                for (int s = 0; s < scene.bodyCount; s++)
                    Assert.IsTrue(math.all(expectedTopology.Ranges[s].range == gpuRanges[s].range), what + ": range of slot " + s);

                if (expectedLanes > 0)
                {
                    int largest = 0;
                    for (int c = 0; c < constraints.ColorCount; c++)
                        largest = Math.Max(largest, constraints.ColorSize(c));
                    Assert.AreEqual(expectedLanes, AvbdGraphColoring.PrimalLanesFor(largest), what + ": the biggest color has " + largest + " bodies");
                }

                var solver = new AvbdSolver(buffers, settings, null, constraints);
                solver.DispatchPredict();
                solver.DispatchSolveHeader();
                solver.DispatchPrimalSweep();

                CpuPredict.Run(oracle, settings, scene.bodyCount);
                int failures = CpuPrimal.Sweep(oracle, scene.constraints, expectedTopology, settings);
                Assert.AreEqual(0, failures, what + ": the oracle must not fail a solve");
                Assert.AreEqual(0, constraints.ReadSolveFailures(), what + ": GPU solve failures");

                var state = new BodyStateRecord[capacity];
                buffers.State.GetData(state);
                PoseMirror.Assert(buffers, "after a primal sweep");
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
                constraints.ReadRecords(records);

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
                PoseMirror.Assert(buffers, "after a primal sweep");
                var records = new ConstraintRecord[8];
                constraints.ReadRecords(records);

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
                PoseMirror.Assert(buffers, "after a primal sweep");
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
