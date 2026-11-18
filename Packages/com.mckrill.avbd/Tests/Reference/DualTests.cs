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

        [Test]
        public void Oracle_YieldsAndBreaksPlasticJoints_LikeTheReferenceSolver([Values(1, 3)] int iterations)
        {
            var solver = new ReferenceSolver();
            PlasticScene(solver, out AvbdJoint yields, out AvbdJoint holds, out AvbdJoint bends);
            CheckOracleAgainstReference(solver, iterations, "plastic scene");

            Assert.AreNotEqual(quaternion.identity.value, yields.rest.value, "the first joint must have yielded");
            Assert.LessOrEqual(math.length(yields.lambdaAng), yields.yield * 1.001f, "a yielded joint holds its multiplier at the yield value");
            Assert.IsFalse(yields.broken, "yielding alone does not break a joint");
            Assert.AreEqual(quaternion.identity.value, holds.rest.value, "the joint with a huge yield value stays rigid");
            Assert.IsFalse(holds.broken);
            Assert.IsTrue(bends.broken, "the joint that yielded past its bend limit must break");
        }

        [Test]
        public void Oracle_YieldClampsTheMultiplier_AndKeepsTheBend_ByHand()
        {
            // A world joint to a body that is turned 0.1 rad about z: the angular row has an error of about 0.1, so with a penalty of 1000 the multiplier
            // jumps to about 100 (torque arm 1). The yield value is 10.
            const float angle = 0.1f;
            quaternion bent = quaternion.AxisAngle(new float3(0f, 0f, 1f), angle);
            CpuBodyArrays bodies = OneBodyWithRotation(bent, out BodyHandle b);

            ConstraintRecord c = ConstraintRecordFactory.Joint(AvbdConst.Invalid, (uint)b.Slot, float3.zero, float3.zero,
                float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, 1f);
            PlasticRecord[] plastic = { ConstraintRecordFactory.MakePlastic(ref c, 0u, 10f, float.PositiveInfinity) };
            c.penaltyLin = new float4(1000f, 1000f, 1000f, 0f);
            c.penaltyAng = new float4(1000f, 1000f, 1000f, 0f);

            AvbdSolverSettings s = AvbdSolverSettings.Default;
            CpuDual.Update(bodies, ref c, s, plastic);

            Assert.AreEqual(10f, math.length(c.lambdaAng.xyz), 1e-3f, "the multiplier is held at the yield value");
            Assert.Less(c.lambdaAng.z, 0f, "its direction is the direction of the error (upstream's difference is a minus b, so it points the other way)");
            AssertNear(bent.value, plastic[0].rest, 1e-5f, "rest takes the current relative rotation");
            Assert.AreEqual(float3.zero, c.c0Ang.xyz, "the stabilization reference is cleared, so it cannot pull the bend back");
            Assert.AreEqual(1000f, c.penaltyAng.x, "a yielding row does not ramp its penalty");
            Assert.IsTrue((ConstraintFlagLayout.FlagsOf(c.ident.z) & ConstraintFlags.Broken) == 0);

            // At the new rest pose the error is zero: nothing pulls the joint back, and the multiplier stays.
            float3 held = c.lambdaAng.xyz;
            CpuDual.Update(bodies, ref c, s, plastic);
            AssertNear(held, c.lambdaAng.xyz, 1e-3f, "the held multiplier");
            AssertNear(bent.value, plastic[0].rest, 1e-5f, "rest after a second update");
        }

        [Test]
        public void Oracle_BendLimit_BreaksAYieldedJointOnlyPastTheLimit()
        {
            foreach (float limit in new[] { 0.05f, 0.2f })
            {
                CpuBodyArrays bodies = OneBodyWithRotation(quaternion.AxisAngle(new float3(0f, 0f, 1f), 0.1f), out BodyHandle b);
                ConstraintRecord c = ConstraintRecordFactory.Joint(AvbdConst.Invalid, (uint)b.Slot, float3.zero, float3.zero,
                    float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, 1f);
                PlasticRecord[] plastic = { ConstraintRecordFactory.MakePlastic(ref c, 0u, 10f, limit) };
                c.penaltyLin = new float4(1000f, 1000f, 1000f, 0f);
                c.penaltyAng = new float4(1000f, 1000f, 1000f, 0f);

                CpuDual.Update(bodies, ref c, AvbdSolverSettings.Default, plastic);

                bool broken = (ConstraintFlagLayout.FlagsOf(c.ident.z) & ConstraintFlags.Broken) != 0;
                Assert.AreEqual(limit < 0.1f, broken, "a bend of 0.1 rad against a limit of " + limit);
                if (broken)
                    Assert.AreEqual(0f, math.lengthsq(c.lambdaAng.xyz) + math.lengthsq(c.penaltyAng.xyz), "a broken joint exerts nothing");
            }
        }

        [Test]
        public void Oracle_AMultiplierAboveTheFractureValue_BreaksInsteadOfYielding()
        {
            CpuBodyArrays bodies = OneBodyWithRotation(quaternion.AxisAngle(new float3(0f, 0f, 1f), 0.1f), out BodyHandle b);
            ConstraintRecord c = ConstraintRecordFactory.Joint(AvbdConst.Invalid, (uint)b.Slot, float3.zero, float3.zero,
                float.PositiveInfinity, float.PositiveInfinity, 50f, 1f); // fracture 50, yield 10: the multiplier jumps to about 100
            PlasticRecord[] plastic = { ConstraintRecordFactory.MakePlastic(ref c, 0u, 10f, float.PositiveInfinity) };
            c.penaltyLin = new float4(1000f, 1000f, 1000f, 0f);
            c.penaltyAng = new float4(1000f, 1000f, 1000f, 0f);

            CpuDual.Update(bodies, ref c, AvbdSolverSettings.Default, plastic);

            Assert.IsTrue((ConstraintFlagLayout.FlagsOf(c.ident.z) & ConstraintFlags.Broken) != 0);
            AssertNear(new float4(0f, 0f, 0f, 1f), plastic[0].rest, 0f, "a joint that breaks does not move its rest rotation");
        }

        [Test]
        public void Oracle_APlasticJointThatNeverYields_IsTheRigidJoint()
        {
            // The same scene twice: once rigid, once with a yield value that is never reached. The results are identical.
            var rigid = new ReferenceSolver();
            var plastic = new ReferenceSolver();
            foreach (ReferenceSolver s in new[] { rigid, plastic })
            {
                float yield = s == plastic ? 1e9f : float.PositiveInfinity;
                s.Clear();
                AvbdBody a = s.AddBody(new float3(1f), 0f, 0.5f, new float3(0f, 0f, 5f));
                AvbdBody b = s.AddBody(new float3(1f), 1f, 0.5f, new float3(2f, 0f, 5f));
                AvbdJoint j = s.AddJoint(a, b, new float3(1f, 0f, 0f), new float3(-1f, 0f, 0f), float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, yield);
                j.penaltyLin = new float3(1000f);
                j.penaltyAng = new float3(100f);
                j.lambdaAng = new float3(5f, 0f, 0f);
                s.Params.gravity = new float3(0f, 0f, -10f);
                s.Params.iterations = 3;
                s.Params.mode = GaussSeidelMode.Colored;
                s.Step();
            }
            for (int i = 0; i < 2; i++)
            {
                Assert.AreEqual(rigid.Bodies[i].PositionLin, plastic.Bodies[i].PositionLin, "body " + i + " position");
                Assert.AreEqual(rigid.Bodies[i].PositionAng.value, plastic.Bodies[i].PositionAng.value, "body " + i + " rotation");
            }
        }

        // A static box and three dynamic ones in a row, joined by three hard joints (linear and angular), each with the same load: the first yields (yield
        // value 0.01 against a multiplier of about 5) and bends with no limit, the second has a yield value it never reaches, the third yields and breaks at
        // once, because its bend limit is tiny.
        static void PlasticScene(ReferenceSolver s, out AvbdJoint yields, out AvbdJoint holds, out AvbdJoint bends)
        {
            s.Clear();
            AvbdBody a = s.AddBody(new float3(1f), 0f, 0.5f, new float3(0f, 0f, 5f));
            AvbdBody b = s.AddBody(new float3(1f), 1f, 0.5f, new float3(2f, 0f, 5f));
            AvbdBody c = s.AddBody(new float3(1f), 1f, 0.5f, new float3(4f, 0f, 5f));
            AvbdBody d = s.AddBody(new float3(1f), 1f, 0.5f, new float3(6f, 0f, 5f));

            float3 rA = new float3(1f, 0f, 0f);
            float3 rB = new float3(-1f, 0f, 0f);
            float inf = float.PositiveInfinity;
            yields = s.AddJoint(a, b, rA, rB, inf, inf, inf, 0.01f);
            holds = s.AddJoint(b, c, rA, rB, inf, inf, inf, 1e9f);
            bends = s.AddJoint(c, d, rA, rB, inf, inf, inf, 0.01f, 1e-6f);
            foreach (AvbdJoint j in new[] { yields, holds, bends })
            {
                j.penaltyLin = new float3(1000f);
                j.penaltyAng = new float3(100f);
                j.lambdaLin = new float3(0f, 1f, 2f);
                j.lambdaAng = new float3(5f, 0f, 0f);
            }
        }

        static CpuBodyArrays OneBodyWithRotation(quaternion rotation, out BodyHandle handle)
        {
            var bodies = new CpuBodyArrays(4);
            var queue = new BodyEditQueue(4);
            var commands = new List<EditCommand>();
            BodyDescriptor d = BodyDescriptor.Box(new float3(1f), 1f, float3.zero);
            d.rotation = rotation;
            handle = queue.Spawn(d);
            queue.Drain(commands);
            bodies.Apply(commands);
            return bodies;
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
                if (!float.IsPositiveInfinity(j.yield))
                {
                    Assert.IsTrue((ConstraintFlagLayout.FlagsOf(r.ident.z) & ConstraintFlags.Plastic) != 0, name + " is plastic");
                    AssertNear(j.rest.value, scene.plastic[r.ident.w].rest, RotationTolerance * iterations, name + " rest");
                }
                ci++;
            }
        }

        // The step the GPU runs, on the CPU: warm start (the start pose is still the pose), predict, then per iteration a primal sweep
        // and the dual update. Returns the number of failed solves.
        static int RunOracleStep(CpuBodyArrays oracle, SceneArrays scene, in AvbdSolverSettings settings)
        {
            CpuConstraintInit.Run(oracle, scene.constraints, scene.constraintCount, settings, scene.plastic);
            CpuPredict.Run(oracle, settings, scene.bodyCount);
            ConstraintTopology topology = scene.BuildTopology();

            int failures = 0;
            for (int it = 0; it < settings.iterations; it++)
            {
                failures += CpuPrimal.Sweep(oracle, scene.constraints, topology, settings, scene.plastic);
                CpuDual.Run(oracle, scene.constraints, scene.constraintCount, settings, scene.plastic);
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

        // plasticCapacity: the plastic capacity of the GPU constraint buffers. With 0 the joint kernels run without the plastic code; a scene with plastic
        // joints needs it above 0, and then they run the variants that read the plastic array (also on a scene that has no plastic joint).
        static void CheckGpuAgainstOracle(ReferenceSolver reference, int iterations, string what, int plasticCapacity = 0)
        {
            AvbdSolverSettings settings = Settings(iterations);
            SceneArrays scene = ConstraintTestScenes.FromReference(reference, Capacity);

            // The GPU gets the records as the scene made them, with the random penalties and multipliers but without the warm start:
            // the ConstraintInit kernel does that.
            var raw = (ConstraintRecord[])scene.constraints.Clone();
            var rawPlastic = (PlasticRecord[])scene.plastic.Clone();

            int oracleFailures = RunOracleStep(scene.bodies, scene, settings);
            Assert.AreEqual(0, oracleFailures, what + ": the oracle must not fail a solve");

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 128, null, 0, plasticCapacity))
            {
                for (int i = 0; i < scene.descriptors.Count; i++)
                    Assert.AreEqual(i, buffers.Spawn(scene.descriptors[i]).Slot);
                for (int i = 0; i < scene.constraintCount; i++)
                {
                    bool plastic = (ConstraintFlagLayout.FlagsOf(raw[i].ident.z) & ConstraintFlags.Plastic) != 0;
                    Assert.AreEqual(i, plastic ? constraints.AddRecord(raw[i], rawPlastic[raw[i].ident.w]) : constraints.AddRecord(raw[i]));
                }

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

                if (plasticCapacity > 0)
                {
                    PlasticRecord[] plastic = constraints.ReadPlastic();
                    Assert.AreEqual(scene.plastic.Length, plastic.Length, what + ": plastic records");
                    for (int i = 0; i < plastic.Length; i++)
                    {
                        string name = what + ", plastic " + i;
                        AssertNear(scene.plastic[i].rest, plastic[i].rest, RotationTolerance * iterations, name + " rest");
                        Assert.AreEqual(scene.plastic[i].restStart, plastic[i].restStart, name + " restStart never changes");
                        Assert.AreEqual(scene.plastic[i].limits, plastic[i].limits, name + " limits never change");
                    }
                }
            }
        }

        [TestCase(1)]
        [TestCase(3)]
        public void Gpu_Step_MatchesTheOracle_OnThePlasticScene(int iterations)
        {
            RequireCompute();
            var solver = new ReferenceSolver();
            PlasticScene(solver, out _, out _, out _);
            CheckGpuAgainstOracle(solver, iterations, "plastic scene", 8);
        }

        [Test]
        public void Gpu_TheVariantsWithThePlasticArray_MatchTheOracle_OnScenesWithoutPlasticJoints()
        {
            RequireCompute();
            var rope = new ReferenceSolver();
            AvbdScenes.Rope(rope);
            ConstraintTestScenes.RandomizeJointState(rope, new System.Random(7));
            CheckGpuAgainstOracle(rope, 3, "rope with a plastic array", 4);

            var fracture = new ReferenceSolver();
            FractureScene(fracture, out _, out _);
            CheckGpuAgainstOracle(fracture, 3, "fracture scene with a plastic array", 4);
        }

        // ---- plastic joints in the buffers --------------------------------------------------------------------

        [Test]
        public void Buffers_RejectPlasticJointsThatCannotWork()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(8))
            using (var rigidOnly = new AvbdConstraintBuffers(buffers, 8))
            using (var constraints = new AvbdConstraintBuffers(buffers, 8, null, 0, 2))
            {
                BodyHandle a = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 0f, new float3(0f, 5f, 0f)));
                BodyHandle b = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 2f, 0f)));
                float inf = float.PositiveInfinity;

                Assert.Throws<System.InvalidOperationException>(() => rigidOnly.AddJoint(a, b, float3.zero, float3.zero, inf, inf, inf, 10f),
                    "buffers made without a plastic capacity cannot hold a plastic joint");
                Assert.Throws<System.ArgumentException>(() => constraints.AddJoint(a, b, float3.zero, float3.zero, inf, 500f, inf, 10f), "a plastic joint needs a hard angular row");
                Assert.Throws<System.ArgumentException>(() => constraints.AddJoint(a, b, float3.zero, float3.zero, inf, inf, inf, 0f), "yield must be above zero");
                Assert.Throws<System.ArgumentException>(() => constraints.AddJoint(a, b, float3.zero, float3.zero, inf, inf, inf, float.NaN));
                Assert.Throws<System.ArgumentException>(() => constraints.AddJoint(a, b, float3.zero, float3.zero, inf, inf, inf, inf, 0.5f), "a bend limit needs a yield value");
                Assert.Throws<System.ArgumentException>(() => constraints.AddJoint(a, b, float3.zero, float3.zero, inf, inf, inf, 10f, 0f), "a bend limit must be above zero");
                Assert.Throws<System.ArgumentException>(() => constraints.AddJoint(a, b, float3.zero, float3.zero, inf, inf, inf, 10f, 4f), "a bend limit is at most pi");
                Assert.AreEqual(0, constraints.Count, "nothing was added");
                Assert.Throws<System.ArgumentOutOfRangeException>(() => new AvbdConstraintBuffers(buffers, 4, null, 0, 5), "more plastic joints than joints");
            }
        }

        [Test]
        public void Buffers_BuildThePlasticRecords_InTheOrderTheJointsWereAdded_AndCountTheOverflow()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(8))
            using (var constraints = new AvbdConstraintBuffers(buffers, 8, null, 0, 2))
            {
                BodyHandle a = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 0f, new float3(0f, 5f, 0f)));
                BodyHandle b = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 2f, 0f)));
                float inf = float.PositiveInfinity;

                int rigid = constraints.AddJoint(a, b, float3.zero, float3.zero, inf, inf);
                int first = constraints.AddJoint(a, b, float3.zero, float3.zero, inf, inf, inf, 10f);
                int second = constraints.AddJoint(a, b, float3.zero, float3.zero, inf, inf, inf, 20f, 0.5f);
                int third = constraints.AddJoint(a, b, float3.zero, float3.zero, inf, inf, inf, 30f);
                Assert.AreEqual(new[] { 0, 1, 2, -1 }, new[] { rigid, first, second, third }, "the plastic capacity is 2: the third plastic joint finds no room");
                Assert.AreEqual(1, constraints.OverflowCount);
                Assert.AreEqual(2, constraints.PlasticCount);
                constraints.Rebuild();

                var records = new ConstraintRecord[8];
                constraints.Records.GetData(records);
                Assert.IsTrue((ConstraintFlagLayout.FlagsOf(records[rigid].ident.z) & ConstraintFlags.Plastic) == 0, "a rigid joint takes no plastic slot");
                Assert.AreEqual(0u, records[first].ident.w);
                Assert.AreEqual(1u, records[second].ident.w);
                Assert.IsTrue((ConstraintFlagLayout.FlagsOf(records[second].ident.z) & ConstraintFlags.BendBreak) != 0);
                Assert.IsTrue((ConstraintFlagLayout.FlagsOf(records[first].ident.z) & ConstraintFlags.BendBreak) == 0);

                PlasticRecord[] plastic = constraints.ReadPlastic();
                Assert.AreEqual(2, plastic.Length);
                Assert.AreEqual(10f, plastic[0].limits.x);
                Assert.AreEqual(20f, plastic[1].limits.x);
                Assert.AreEqual(0.5f, plastic[1].limits.y);
                Assert.AreEqual(new float4(0f, 0f, 0f, 1f), plastic[1].rest);
                Assert.AreEqual(new float4(0f, 0f, 0f, 1f), plastic[1].restStart);
            }
        }

        // A static box with a dynamic one hanging off its side on a hard joint (linear and angular). The joint holds the dynamic box's weight as a torque:
        // mass 1 * gravity 10 * a lever of 1, which is a multiplier of about 10 / 12 (the torque arm of this joint is 12). Rigid, the box hangs nearly
        // level; plastic with a yield value of 0.2 (a torque of 2.4, far below the load), it bends down and stays there; with a bend limit as well, the
        // joint breaks.
        static void HangingBox(bool plastic, float bendLimit, out float tilt, out bool broken, out float4 rest)
        {
            using (var buffers = new AvbdBodyBuffers(8))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4, null, 0, plastic ? 1 : 0))
            {
                BodyHandle a = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 0f, new float3(0f, 5f, 0f)));
                BodyHandle b = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(2f, 5f, 0f)));
                float inf = float.PositiveInfinity;
                constraints.AddJoint(a, b, new float3(1f, 0f, 0f), new float3(-1f, 0f, 0f), inf, inf, inf, plastic ? 0.2f : inf, bendLimit);

                AvbdSolverSettings settings = AvbdSolverSettings.Default; // y is up, so the box swings about z
                var solver = new AvbdSolver(buffers, settings, null, constraints);
                for (int i = 0; i < 120; i++)
                    solver.Step();

                var state = new BodyStateRecord[8];
                buffers.State.GetData(state);
                float3 hanging = state[(int)b.Slot].position.xyz - state[(int)a.Slot].position.xyz;
                tilt = math.atan2(-hanging.y, hanging.x); // how far the box has swung down from the level

                var records = new ConstraintRecord[4];
                constraints.Records.GetData(records);
                broken = (ConstraintFlagLayout.FlagsOf(records[0].ident.z) & ConstraintFlags.Broken) != 0;
                rest = plastic ? constraints.ReadPlastic()[0].rest : new float4(0f, 0f, 0f, 1f);
            }
        }

        [Test]
        public void Gpu_APlasticJoint_BendsUnderALoadTheRigidJointHolds_AndKeepsTheBend()
        {
            RequireCompute();
            HangingBox(false, float.PositiveInfinity, out float rigidTilt, out bool rigidBroken, out _);
            HangingBox(true, float.PositiveInfinity, out float plasticTilt, out bool plasticBroken, out float4 rest);

            Assert.IsFalse(rigidBroken);
            Assert.IsFalse(plasticBroken, "yielding alone does not break a joint");
            Assert.Less(math.abs(rigidTilt), 0.2f, "the rigid joint holds the box nearly level");
            Assert.Greater(plasticTilt, math.abs(rigidTilt) + 0.3f, "the plastic joint gives way under the same load");
            Assert.Greater(math.length(rest.xyz), 0.05f, "and its rest rotation has moved to the bent pose: it does not spring back");
        }

        [Test]
        public void Gpu_APlasticJoint_BreaksOnceItIsBentPastItsLimit()
        {
            RequireCompute();
            HangingBox(true, 0.05f, out _, out bool broken, out _);
            Assert.IsTrue(broken, "a limit of 0.05 rad is passed long before the box has finished sagging");

            HangingBox(true, 3f, out _, out bool brokenFar, out _);
            Assert.IsFalse(brokenFar, "a limit of 3 rad is not reached by a box hanging off a hinge");
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
