using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Stage D (constraint warm start and dual update). The CPU oracles are checked against the reference solver over a whole step
    /// (predict, warm start, N iterations of primal sweep and dual update, as far as body poses and constraint variables go), and the
    /// GPU solver is checked against the oracles.
    /// </summary>
    public class DualTests
    {
        const int Capacity = 64;
        const float PositionTolerance = 2e-3f; // per iteration
        const float RotationTolerance = 2e-4f; // per iteration
        const float DualTolerance = 1e-2f;     // per iteration, relative to max(1, |value|)

        // ---- the oracles against the reference solver ----------------------------------------------------------

        [TestCase(1)]
        [TestCase(3)]
        public void Oracle_MatchesTheReferenceSolver_OnTheRope(int iterations)
        {
            var solver = new ReferenceSolver();
            AvbdScenes.Rope(solver);
            ConstraintTestScenes.RandomizeJointState(solver, new System.Random(5));
            CheckOracleAgainstReference(solver, iterations, "rope");
        }

        [TestCase(1)]
        [TestCase(3)]
        public void Oracle_MatchesTheReferenceSolver_OnRandomScenes(int iterations)
        {
            for (int seed = 1; seed <= 8; seed++)
            {
                var solver = new ReferenceSolver();
                ConstraintTestScenes.RandomLattice(solver, 200 + seed);
                CheckOracleAgainstReference(solver, iterations, "random scene " + seed);
            }
        }

        [Test]
        public void Oracle_BreaksAFracturingJoint_LikeTheReferenceSolver()
        {
            var solver = new ReferenceSolver();
            FractureScene(solver, out AvbdJoint weak, out AvbdJoint strong);
            CheckOracleAgainstReference(solver, 3, "fracture scene");

            Assert.IsTrue(weak.broken, "the scenario must break the weak joint");
            Assert.IsFalse(strong.broken, "the strong joint must hold");
            Assert.AreEqual(0f, math.lengthsq(weak.lambdaAng) + math.lengthsq(weak.lambdaLin) + math.lengthsq(weak.penaltyLin) + math.lengthsq(weak.penaltyAng));
        }

        [Test]
        public void Oracle_UpdatesTheDualVariablesOfAHardJointByHand()
        {
            // One hard world joint at the body's position, penalty 100, no error in the pose: lambda stays, the penalty grows by |C| beta.
            var bodies = new CpuBodyArrays(4);
            var queue = new BodyEditQueue(4);
            var commands = new List<EditCommand>();
            BodyHandle b = queue.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(1f, 2f, 3f)));
            queue.Drain(commands);
            bodies.Apply(commands);

            // The pin is 0.5 above the body in y. The body is at (1,2,3) so C = pin - body = (0, 0.5, 0).
            ConstraintRecord c = ConstraintRecordFactory.Joint(AvbdConst.Invalid, (uint)b.Slot, new float3(1f, 2.5f, 3f), float3.zero,
                float.PositiveInfinity, 0f, float.PositiveInfinity, 1f);
            c.penaltyLin = new float4(100f, 100f, 100f, 0f);
            c.lambdaLin = new float4(2f, 3f, 4f, 0f);
            c.c0Lin = new float4(0f, 0.5f, 0f, c.c0Lin.w); // C0 equals today's error

            AvbdSolverSettings s = AvbdSolverSettings.Default;
            CpuDual.Update(bodies, ref c, s);

            // Stabilized error: C - C0 * alpha = (0, 0.5 - 0.5 * 0.99, 0) = (0, 0.005, 0).
            float3 stab = new float3(0f, 0.5f - 0.5f * s.alpha, 0f);
            Assert.AreEqual((double)(100f * stab.y + 3f), c.lambdaLin.y, 1e-4);
            Assert.AreEqual(2f, c.lambdaLin.x, 1e-6);
            // The penalty ramps with the stabilized |C|.
            Assert.AreEqual(100f + stab.y * s.betaLin, c.penaltyLin.y, 1e-3);
            Assert.AreEqual(100f, c.penaltyLin.x, 1e-6);
        }

        // Three boxes in a row: a static one, then two dynamic ones, joined by two hard joints (linear and angular). The first joint breaks
        // at the first dual update (fracture 0.01 against a multiplier of about 5), the second never does.
        static void FractureScene(ReferenceSolver s, out AvbdJoint weak, out AvbdJoint strong)
        {
            s.Clear();
            AvbdBody a = s.AddBody(new float3(1f), 0f, 0.5f, new float3(0f, 0f, 5f));
            AvbdBody b = s.AddBody(new float3(1f), 1f, 0.5f, new float3(2f, 0f, 5f));
            AvbdBody c = s.AddBody(new float3(1f), 1f, 0.5f, new float3(4f, 0f, 5f));

            weak = s.AddJoint(a, b, new float3(1f, 0f, 0f), new float3(-1f, 0f, 0f), float.PositiveInfinity, float.PositiveInfinity, 0.01f);
            strong = s.AddJoint(b, c, new float3(1f, 0f, 0f), new float3(-1f, 0f, 0f), float.PositiveInfinity, float.PositiveInfinity, 1e6f);
            foreach (AvbdJoint j in new[] { weak, strong })
            {
                j.penaltyLin = new float3(1000f);
                j.penaltyAng = new float3(100f);
                j.lambdaLin = new float3(0f, 1f, 2f);
                j.lambdaAng = new float3(5f, 0f, 0f);
            }
        }

        static void CheckOracleAgainstReference(ReferenceSolver solver, int iterations, string what)
        {
            AvbdSolverSettings settings = Settings(iterations);
            solver.Params.dt = settings.dt;
            solver.Params.gravity = settings.gravity;
            solver.Params.iterations = iterations;
            solver.Params.mode = GaussSeidelMode.Colored;

            SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);
            solver.Step();

            CpuBodyArrays oracle = scene.bodies;
            RunOracleStep(oracle, scene, settings);

            for (int s = 0; s < scene.bodyCount; s++)
            {
                AvbdBody b = solver.Bodies[s];
                string name = what + ", body " + s;
                AssertNear(b.PositionLin, oracle.State[s].position.xyz, PositionTolerance * iterations, name + " position");
                AssertNear(b.PositionAng.value, oracle.State[s].rotation, RotationTolerance * iterations, name + " rotation");
            }

            int ci = 0;
            foreach (AvbdForce force in solver.Forces)
            {
                if (!(force is AvbdJoint j))
                {
                    if (force is AvbdSpring)
                        ci++;
                    continue;
                }
                ConstraintRecord r = scene.constraints[ci];
                string name = what + ", joint " + ci;
                Assert.AreEqual(j.broken, (ConstraintFlagLayout.FlagsOf(r.ident.z) & ConstraintFlags.Broken) != 0, name + " broken");
                AssertNear(j.C0Lin, r.c0Lin.xyz, 1e-4f, name + " C0Lin");
                AssertNear(j.C0Ang, r.c0Ang.xyz, 1e-4f, name + " C0Ang");
                float tol = DualTolerance * iterations;
                AssertNear(j.penaltyLin, r.penaltyLin.xyz, tol, name + " penaltyLin");
                AssertNear(j.penaltyAng, r.penaltyAng.xyz, tol, name + " penaltyAng");
                AssertNear(j.lambdaLin, r.lambdaLin.xyz, tol, name + " lambdaLin");
                AssertNear(j.lambdaAng, r.lambdaAng.xyz, tol, name + " lambdaAng");
                ci++;
            }
        }

        // The step the GPU runs, on the CPU: warm start (the start pose is still the pose), predict, then per iteration a primal sweep
        // and the dual update. Returns the number of failed solves.
        static int RunOracleStep(CpuBodyArrays oracle, SceneArrays scene, in AvbdSolverSettings settings)
        {
            CpuConstraintInit.Run(oracle, scene.constraints, scene.constraintCount, settings);
            CpuPredict.Run(oracle, settings, scene.bodyCount);
            ConstraintTopology topology = scene.BuildTopology();

            int failures = 0;
            for (int it = 0; it < settings.iterations; it++)
            {
                failures += CpuPrimal.Sweep(oracle, scene.constraints, topology, settings);
                CpuDual.Run(oracle, scene.constraints, scene.constraintCount, settings);
            }
            return failures;
        }

        // ---- GPU ----------------------------------------------------------------------------------------------

        [TestCase(1)]
        [TestCase(3)]
        public void Gpu_Step_MatchesTheOracle_OnTheRope(int iterations)
        {
            RequireCompute();
            var solver = new ReferenceSolver();
            AvbdScenes.Rope(solver);
            ConstraintTestScenes.RandomizeJointState(solver, new System.Random(6));
            CheckGpuAgainstOracle(solver, iterations, "rope");
        }

        [TestCase(1)]
        [TestCase(3)]
        public void Gpu_Step_MatchesTheOracle_OnTheFractureScene(int iterations)
        {
            RequireCompute();
            var solver = new ReferenceSolver();
            FractureScene(solver, out _, out _);
            CheckGpuAgainstOracle(solver, iterations, "fracture scene");
        }

        [TestCase(1)]
        [TestCase(3)]
        public void Gpu_Step_MatchesTheOracle_OnRandomScenes(int iterations)
        {
            RequireCompute();
            for (int seed = 1; seed <= 5; seed++)
            {
                var solver = new ReferenceSolver();
                ConstraintTestScenes.RandomLattice(solver, 300 + seed);
                CheckGpuAgainstOracle(solver, iterations, "random scene " + seed);
            }
        }

        static void CheckGpuAgainstOracle(ReferenceSolver reference, int iterations, string what)
        {
            AvbdSolverSettings settings = Settings(iterations);
            SceneArrays scene = ConstraintTestScenes.FromReference(reference, Capacity);

            // The GPU gets the records as the scene made them, with the random penalties and multipliers but without the warm start:
            // the ConstraintInit kernel does that.
            var raw = (ConstraintRecord[])scene.constraints.Clone();

            int oracleFailures = RunOracleStep(scene.bodies, scene, settings);
            Assert.AreEqual(0, oracleFailures, what + ": the oracle must not fail a solve");

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 128))
            {
                for (int i = 0; i < scene.descriptors.Count; i++)
                    Assert.AreEqual(i, buffers.Spawn(scene.descriptors[i]).Slot);
                for (int i = 0; i < scene.constraintCount; i++)
                    Assert.AreEqual(i, constraints.AddRecord(raw[i]));

                new AvbdSolver(buffers, settings, null, constraints).Step();
                Assert.AreEqual(0, constraints.ReadSolveFailures(), what + ": GPU solve failures");

                var state = new BodyStateRecord[Capacity];
                buffers.State.GetData(state);
                var records = new ConstraintRecord[128];
                constraints.Records.GetData(records);

                for (int s = 0; s < scene.bodyCount; s++)
                {
                    string name = what + ", slot " + s;
                    AssertNear(scene.bodies.State[s].position, state[s].position, PositionTolerance * iterations, name + " position");
                    AssertNear(scene.bodies.State[s].rotation, state[s].rotation, RotationTolerance * iterations, name + " rotation");
                }

                for (int i = 0; i < scene.constraintCount; i++)
                {
                    ConstraintRecord expected = scene.constraints[i];
                    ConstraintRecord actual = records[i];
                    string name = what + ", constraint " + i;

                    Assert.AreEqual(expected.ident, actual.ident, name + " ident (bodies, flags, broken)");
                    float tol = DualTolerance * iterations;
                    AssertNear(expected.c0Lin, actual.c0Lin, 1e-4f, name + " c0Lin");
                    AssertNear(expected.c0Ang, actual.c0Ang, 1e-4f, name + " c0Ang");
                    AssertNear(expected.penaltyLin, actual.penaltyLin, tol, name + " penaltyLin");
                    AssertNear(expected.penaltyAng, actual.penaltyAng, tol, name + " penaltyAng");
                    AssertNear(expected.lambdaLin, actual.lambdaLin, tol, name + " lambdaLin");
                    AssertNear(expected.lambdaAng, actual.lambdaAng, tol, name + " lambdaAng");
                }
            }
        }

        [Test]
        public void Gpu_Springs_AreLeftAloneByTheDualStage()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(8))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4))
            {
                BodyHandle a = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 0f, new float3(0f, 5f, 0f)));
                BodyHandle b = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 2f, 0f)));
                constraints.AddSpring(a, b, float3.zero, float3.zero, 50f, 2f);
                constraints.Rebuild();

                var before = new ConstraintRecord[4];
                constraints.Records.GetData(before);

                new AvbdSolver(buffers, AvbdSolverSettings.Default, null, constraints).Step();

                var after = new ConstraintRecord[4];
                constraints.Records.GetData(after);
                Assert.AreEqual(before[0], after[0], "a spring record never changes");
            }
        }

        // ---- helpers -----------------------------------------------------------------------------------------

        static AvbdSolverSettings Settings(int iterations)
        {
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            settings.gravity = new float3(0f, 0f, -10f); // the upstream scenes are z-up
            settings.iterations = iterations;
            return settings;
        }

        static void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }

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
