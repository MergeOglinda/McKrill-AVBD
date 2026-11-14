using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Stage B (broadphase). The CPU oracle (<see cref="CpuBroadphase"/>) is checked against the reference solver's own pair loop; the
    /// GPU must then find exactly the oracle's set of pairs (the order is unspecified), on the body state it holds.
    /// </summary>
    public class BroadphaseTests
    {
        const int Capacity = 256;
        const int BigBin = 1 << 16;

        // ---- the oracle against the reference solver -------------------------------------------------------------------

        [Test]
        public void Oracle_FindsTheReferenceSolversPairs_OnTheUpstreamScenes()
        {
            var scenes = new Dictionary<string, Action<ReferenceSolver>>
            {
                { "ground", s => AvbdScenes.Ground(s) },
                { "stack", s => AvbdScenes.Stack(s) },
                { "pyramid", s => AvbdScenes.Pyramid(s, 8) },
                { "rope", s => AvbdScenes.Rope(s) },
                { "spring", s => AvbdScenes.Spring(s) },
            };

            foreach (KeyValuePair<string, Action<ReferenceSolver>> scene in scenes)
            {
                var solver = new ReferenceSolver();
                scene.Value(solver);
                AssertOracleMatchesReference(solver, scene.Key);
            }
        }

        [Test]
        public void Oracle_FindsTheReferenceSolversPairs_OnDenseRandomScenes()
        {
            for (int seed = 1; seed <= 10; seed++)
            {
                var solver = new ReferenceSolver();
                DenseRandom(solver, seed, 70);
                AssertOracleMatchesReference(solver, "dense scene " + seed);
            }
        }

        static void AssertOracleMatchesReference(ReferenceSolver solver, string what)
        {
            SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);
            SortedSet<ulong> oracle = CpuBroadphase.Pairs(scene.bodies, scene.bodyCount, scene.constraints, scene.constraintCount);
            SortedSet<ulong> expected = ReferencePairs(solver);

            AssertSameSets(expected, oracle, what);
            if (what.StartsWith("dense"))
                Assert.Greater(expected.Count, 20, what + ": the scene should have many pairs");
        }

        // The pair loop of ReferenceSolver.Step, without the pairs of two static bodies (the GPU never generates those).
        static SortedSet<ulong> ReferencePairs(ReferenceSolver solver)
        {
            var result = new SortedSet<ulong>();
            IReadOnlyList<AvbdBody> bodies = solver.Bodies;
            for (int i = bodies.Count - 1; i >= 0; i--)
                for (int j = i - 1; j >= 0; j--)
                {
                    AvbdBody a = bodies[i];
                    AvbdBody b = bodies[j];
                    if (!a.IsDynamic && !b.IsDynamic)
                        continue;
                    float3 dp = a.PositionLin - b.PositionLin;
                    float r = a.Radius + b.Radius;
                    if (math.dot(dp, dp) <= r * r && !a.ConstrainedTo(b))
                        result.Add(CpuBroadphase.Key((uint)i, (uint)j, a.IsDynamic && b.IsDynamic ? PairKind.DynamicDynamic : PairKind.DynamicStatic));
                }
            return result;
        }

        // ---- the GPU against the oracle ----------------------------------------------------------------------------------

        [Test]
        public void Gpu_MatchesTheOracle_OnTheUpstreamScenes()
        {
            RequireCompute();

            var pyramid = new ReferenceSolver();
            AvbdScenes.Pyramid(pyramid, 10);
            CheckScene(pyramid, "pyramid"); // the big ground slab is a static neighbor of every box

            var stack = new ReferenceSolver();
            AvbdScenes.Stack(stack);
            CheckScene(stack, "stack");

            var rope = new ReferenceSolver();
            AvbdScenes.Rope(rope);
            CheckScene(rope, "rope"); // jointed neighbors are not pairs

            var spring = new ReferenceSolver();
            AvbdScenes.Spring(spring);
            CheckScene(spring, "spring");
        }

        [Test]
        public void Gpu_MatchesTheOracle_OnDenseRandomScenes()
        {
            RequireCompute();
            for (int seed = 1; seed <= 8; seed++)
            {
                var solver = new ReferenceSolver();
                DenseRandom(solver, 100 + seed, 120);
                CheckScene(solver, "dense scene " + seed);
            }
        }

        // Bodies far from the origin, and sizes that differ by orders of magnitude: the cell size follows the largest body.
        [Test]
        public void Gpu_MatchesTheOracle_ForFarAwayAndMixedSizes()
        {
            RequireCompute();
            foreach (float3 origin in new[] { new float3(-5000f, 3000f, -7000f), new float3(0.25f, -0.5f, 0.125f), new float3(40000f, 40000f, 40000f) })
            {
                var solver = new ReferenceSolver();
                DenseRandom(solver, 7, 90, origin);
                CheckScene(solver, "origin " + origin);
            }

            var rng = new System.Random(5);
            var mixed = new ReferenceSolver();
            for (int i = 0; i < 100; i++)
            {
                float s = math.exp(Range(rng, math.log(0.05f), math.log(8f))); // sizes from 0.05 to 8
                mixed.AddBody(new float3(s, s * 0.5f, s * 1.5f), i % 7 == 0 ? 0f : 1f, 0.5f, new float3(Range(rng, -10f, 10f), Range(rng, -10f, 10f), Range(rng, -10f, 10f)));
            }
            CheckScene(mixed, "mixed sizes");
        }

        [Test]
        public void Gpu_PairsAreWellFormed()
        {
            RequireCompute();
            var solver = new ReferenceSolver();
            DenseRandom(solver, 31, 100);
            SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 128))
            using (var broadphase = new AvbdBroadphase(RequireShader(), Capacity, BigBin, BigBin))
            {
                Fill(buffers, constraints, scene);
                broadphase.Run(buffers, constraints);
                PairRecord[] pairs = broadphase.ReadPairs();
                Assert.Greater(pairs.Length, 20);

                CpuBodyArrays state = ReadBodies(buffers);
                var seen = new HashSet<ulong>();
                foreach (PairRecord pair in pairs)
                {
                    Assert.Greater(pair.BodyA, pair.BodyB, "body A is the higher slot");
                    Assert.AreEqual(0u, pair.pair.w);
                    bool solvedA = CpuBroadphase.Classify(state, (int)pair.BodyA) == CpuBroadphase.BodyClass.Solved;
                    bool solvedB = CpuBroadphase.Classify(state, (int)pair.BodyB) == CpuBroadphase.BodyClass.Solved;
                    Assert.AreEqual(solvedA && solvedB ? PairKind.DynamicDynamic : PairKind.DynamicStatic, pair.Kind, "kind of " + CpuBroadphase.Describe(CpuBroadphase.Key(pair)));
                    Assert.IsTrue(solvedA || solvedB, "a pair of two static bodies");
                    Assert.IsTrue(seen.Add(CpuBroadphase.Key(pair)), "duplicate pair " + CpuBroadphase.Describe(CpuBroadphase.Key(pair)));
                }

                Assert.AreEqual(pairs.Length, broadphase.DynamicPairCount + broadphase.StaticPairCount);
                Assert.IsFalse(broadphase.Overflowed);
            }
        }

        [Test]
        public void Gpu_LeavesOutKinematicStaticAndRemovedBodiesCorrectly()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(32))
            using (var broadphase = new AvbdBroadphase(RequireShader(), 32, 256, 256))
            {
                BodyHandle a = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 0f, 0f)));                // slot 0, solved
                BodyHandle b = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0.5f, 0f, 0f)));              // slot 1, solved
                BodyDescriptor kinematic = BodyDescriptor.Box(new float3(1f), 1f, new float3(0.2f, 0.2f, 0f));
                kinematic.kinematic = true;
                BodyHandle c = buffers.Spawn(kinematic);                                                                      // slot 2, kinematic
                BodyHandle d = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 0f, new float3(0.1f, 0f, 0.3f)));            // slot 3, static
                BodyHandle e = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0.3f, 0.1f, 0f)));            // slot 4, removed below
                BodyHandle g = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(50f, 0f, 0f)));               // slot 5, alone
                Assert.IsTrue(buffers.Remove(e));

                broadphase.Run(buffers, null);
                SortedSet<ulong> got = ToSet(broadphase.ReadPairs());

                var expected = new SortedSet<ulong>
                {
                    CpuBroadphase.Key(1, 0, PairKind.DynamicDynamic),
                    CpuBroadphase.Key(2, 0, PairKind.DynamicStatic),
                    CpuBroadphase.Key(2, 1, PairKind.DynamicStatic),
                    CpuBroadphase.Key(3, 0, PairKind.DynamicStatic),
                    CpuBroadphase.Key(3, 1, PairKind.DynamicStatic),
                };
                // Not in the list: kinematic-static (2, 3), anything with the removed slot 4, and the lone body 5.
                AssertSameSets(expected, got, "hand-computed");
                AssertSameSets(CpuBroadphase.Pairs(ReadBodies(buffers), buffers.HighWaterMark, new ConstraintRecord[0], 0), got, "oracle");

                // A kinematic body can become dynamic again and a body can be moved: edits are applied by Run.
                buffers.SetKinematic(c, false);
                buffers.SetPose(g, new float3(0.4f, 0.4f, 0f), quaternion.identity);
                broadphase.Run(buffers, null);
                got = ToSet(broadphase.ReadPairs());
                AssertSameSets(CpuBroadphase.Pairs(ReadBodies(buffers), buffers.HighWaterMark, new ConstraintRecord[0], 0), got, "after the edits");
                Assert.IsTrue(got.Contains(CpuBroadphase.Key(2, 1, PairKind.DynamicDynamic)), "slot 2 is solved now");
                Assert.IsTrue(got.Contains(CpuBroadphase.Key(5, 2, PairKind.DynamicDynamic)), "slot 5 moved next to slot 2");
                Assert.IsTrue(got.Contains(CpuBroadphase.Key(3, 2, PairKind.DynamicStatic)), "slot 3 (static) and slot 2 (solved) are a pair now");
            }
        }

        [Test]
        public void Gpu_SkipsPairsJoinedByAJointOrSpring_EvenABrokenOne_ButNotByAContact()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var constraints = new AvbdConstraintBuffers(buffers, 16))
            using (var broadphase = new AvbdBroadphase(RequireShader(), 16, 256, 256))
            {
                BodyHandle a = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 0f, 0f)));
                BodyHandle b = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0.3f, 0f, 0f)));
                BodyHandle c = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 0.3f, 0f)));
                BodyHandle ground = buffers.Spawn(BodyDescriptor.Box(new float3(10f, 10f, 1f), 0f, new float3(0f, 0f, 0f)));

                constraints.AddJoint(a, b, float3.zero, float3.zero);                        // 0: dynamic-dynamic
                constraints.AddSpring(b, c, float3.zero, float3.zero, 100f, 1f);             // 1
                constraints.AddJoint(a, ground, float3.zero, float3.zero);                   // 2: to a static body
                constraints.AddJoint(BodyHandle.Invalid, c, new float3(0f, 0.3f, 0f), float3.zero); // 3: pinned to the world, joins nothing
                constraints.Rebuild();

                broadphase.Run(buffers, constraints);
                SortedSet<ulong> got = ToSet(broadphase.ReadPairs());
                var expected = new SortedSet<ulong>
                {
                    CpuBroadphase.Key(2, 0, PairKind.DynamicDynamic),
                    CpuBroadphase.Key(3, 1, PairKind.DynamicStatic),
                    CpuBroadphase.Key(3, 2, PairKind.DynamicStatic),
                };
                AssertSameSets(expected, got, "joined pairs are left out");

                // A fractured joint still excludes its pair, like the narrowphase's _PairExcluded (nothing removes a broken joint yet).
                var records = new ConstraintRecord[constraints.Records.count];
                constraints.Records.GetData(records);
                records[0].ident.z |= (uint)ConstraintFlags.Broken;
                constraints.Records.SetData(records);
                broadphase.Run(buffers, constraints);
                AssertSameSets(expected, ToSet(broadphase.ReadPairs()), "after the joint broke");

                // A contact between a and b (the narrowphase writes them) is in the lists but does not exclude: the pair has to stay a
                // candidate for as long as the boxes touch. Here the joint is also gone from the exclusion by making it a contact.
                records[0].ident.z = ConstraintFlagLayout.Pack(ConstraintFlags.Stick, ConstraintKind.Contact);
                constraints.Records.SetData(records);
                broadphase.Run(buffers, constraints);
                got = ToSet(broadphase.ReadPairs());
                expected.Add(CpuBroadphase.Key(1, 0, PairKind.DynamicDynamic));
                AssertSameSets(expected, got, "a contact does not exclude");
                AssertSameSets(CpuBroadphase.Pairs(ReadBodies(buffers), 4, records, 4), got, "oracle");
            }
        }

        [Test]
        public void Gpu_CountsPairsBeyondTheCapacity_AndStoresOnlyRealPairs()
        {
            RequireCompute();
            var solver = new ReferenceSolver();
            AvbdScenes.Pyramid(solver, 10); // 55 boxes and a ground
            SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);
            SortedSet<ulong> all = CpuBroadphase.Pairs(scene.bodies, scene.bodyCount, scene.constraints, scene.constraintCount);

            int dynamicTotal = 0, staticTotal = 0;
            foreach (ulong key in all)
                if ((PairKind)(key & 3u) == PairKind.DynamicDynamic) dynamicTotal++; else staticTotal++;
            Assert.Greater(dynamicTotal, 10);
            Assert.AreEqual(55, staticTotal);

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 8))
            using (var broadphase = new AvbdBroadphase(RequireShader(), Capacity, 10, 20))
            {
                Fill(buffers, constraints, scene);
                broadphase.Run(buffers, constraints);
                PairRecord[] stored = broadphase.ReadPairs();

                Assert.AreEqual(dynamicTotal, broadphase.DynamicPairCount, "the count includes the dropped pairs");
                Assert.AreEqual(staticTotal, broadphase.StaticPairCount);
                Assert.IsTrue(broadphase.Overflowed);
                Assert.AreEqual(10 + 20, stored.Length, "each bin is full");

                var seen = new HashSet<ulong>();
                foreach (PairRecord pair in stored)
                {
                    Assert.IsTrue(all.Contains(CpuBroadphase.Key(pair)), "stored a pair that is not one: " + CpuBroadphase.Describe(CpuBroadphase.Key(pair)));
                    Assert.IsTrue(seen.Add(CpuBroadphase.Key(pair)), "stored a pair twice");
                }
            }
        }

        [Test]
        public void Gpu_RunsAgainWithTheSameResult_AndFollowsMovingBodies()
        {
            RequireCompute();
            var solver = new ReferenceSolver();
            DenseRandom(solver, 55, 100);
            SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 128))
            using (var broadphase = new AvbdBroadphase(RequireShader(), Capacity, BigBin, BigBin))
            {
                List<BodyHandle> handles = Fill(buffers, constraints, scene);

                broadphase.Run(buffers, constraints);
                SortedSet<ulong> first = ToSet(broadphase.ReadPairs());
                for (int i = 0; i < 3; i++)
                {
                    broadphase.Run(buffers, constraints);
                    AssertSameSets(first, ToSet(broadphase.ReadPairs()), "run " + (i + 2));
                }

                // Pull every body into a small cube: nearly every pair overlaps now, and nothing is left over from the first run.
                var rng = new System.Random(9);
                for (int s = 0; s < scene.bodyCount; s++)
                    buffers.SetPose(handles[s], new float3(Range(rng, -0.3f, 0.3f), Range(rng, -0.3f, 0.3f), Range(rng, -0.3f, 0.3f)), quaternion.identity);
                broadphase.Run(buffers, constraints);
                SortedSet<ulong> packed = ToSet(broadphase.ReadPairs());
                AssertSameSets(CpuBroadphase.Pairs(ReadBodies(buffers), scene.bodyCount, scene.constraints, scene.constraintCount), packed, "packed");
                Assert.GreaterOrEqual(packed.Count, first.Count);
                Assert.IsFalse(first.SetEquals(packed), "the bodies moved, the pairs must change");
            }
        }

        [Test]
        public void Gpu_HandlesNoBodiesAndNoSolvedBodies()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var broadphase = new AvbdBroadphase(RequireShader(), 16, 16, 16))
            {
                broadphase.Run(buffers, null);
                Assert.AreEqual(0, broadphase.ReadPairs().Length);

                buffers.Spawn(BodyDescriptor.Box(new float3(1f), 0f, float3.zero));
                buffers.Spawn(BodyDescriptor.Box(new float3(1f), 0f, new float3(0.2f, 0f, 0f)));
                broadphase.Run(buffers, null);
                Assert.AreEqual(0, broadphase.ReadPairs().Length, "two static bodies make no pair");
                Assert.AreEqual(0f, broadphase.ReadMaxRadius(), "the radius is that of the solved bodies");

                buffers.Spawn(BodyDescriptor.Box(new float3(2f, 0.01f, 0.01f), 1f, new float3(0.1f, 0f, 0f))); // a thin rod: radius 1
                broadphase.Run(buffers, null);
                Assert.AreEqual(2, broadphase.ReadPairs().Length);
                Assert.AreEqual(1f, broadphase.ReadMaxRadius(), 1e-4f);
            }
        }

        // ---- the solver --------------------------------------------------------------------------------------------------

        [Test]
        public void Solver_RunsTheBroadphaseWhenEnabled_OnThePosesOfTheStepStart()
        {
            RequireCompute();
            var reference = new ReferenceSolver();
            AvbdScenes.Pyramid(reference, 6);
            SceneArrays scene = ConstraintTestScenes.FromReference(reference, Capacity);

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 8))
            using (var solver = new AvbdSolver(buffers, AvbdSolverSettings.Default, null, constraints))
            {
                Fill(buffers, constraints, scene);
                Assert.IsFalse(solver.BroadphaseEnabled);
                Assert.IsNull(solver.Broadphase);

                solver.EnableBroadphase(BigBin, BigBin);
                Assert.IsTrue(solver.BroadphaseEnabled);

                SortedSet<ulong> expected = CpuBroadphase.Pairs(ReadBodies(buffers), scene.bodyCount, scene.constraints, scene.constraintCount);
                Assert.Greater(expected.Count, 20);
                solver.Step();
                AssertSameSets(expected, ToSet(solver.Broadphase.ReadPairs()), "first step");

                // The next step sees the poses the first one left.
                expected = CpuBroadphase.Pairs(ReadBodies(buffers), scene.bodyCount, scene.constraints, scene.constraintCount);
                solver.Step();
                AssertSameSets(expected, ToSet(solver.Broadphase.ReadPairs()), "second step");

                solver.DisableBroadphase();
                Assert.IsFalse(solver.BroadphaseEnabled);
                solver.Step(); // still runs
            }
        }

        // The narrowphase of main tests every pair itself. Whatever it makes a manifold of must be a pair of the broadphase, in every step:
        // the pair list is what it will read instead.
        [Test]
        public void Solver_Broadphase_CoversEveryPairThatHasContacts()
        {
            RequireCompute();
            var reference = new ReferenceSolver();
            AvbdScenes.Pyramid(reference, 6);
            SceneArrays scene = ConstraintTestScenes.FromReference(reference, Capacity);

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4, null, scene.bodyCount))
            using (var solver = new AvbdSolver(buffers, AvbdSolverSettings.Default, null, constraints))
            {
                Fill(buffers, constraints, scene);
                solver.EnableBroadphase(BigBin, BigBin);

                int contactPairsSeen = 0;
                for (int step = 0; step < 8; step++)
                {
                    solver.Step();
                    SortedSet<ulong> candidates = ToSet(solver.Broadphase.ReadPairs());
                    uint[] counts = constraints.PairCounts;
                    CpuBodyArrays held = ReadBodies(buffers);
                    for (int pair = 0; pair < ContactLayout.PairCount(scene.bodyCount); pair++)
                    {
                        if (counts[pair] == 0u)
                            continue;
                        contactPairsSeen++;
                        ContactLayout.BodiesOfPair(pair, out int hi, out int lo);
                        bool solvedHi = CpuBroadphase.Classify(held, hi) == CpuBroadphase.BodyClass.Solved;
                        PairKind kind = solvedHi && CpuBroadphase.Classify(held, lo) == CpuBroadphase.BodyClass.Solved
                            ? PairKind.DynamicDynamic : PairKind.DynamicStatic;
                        Assert.IsTrue(candidates.Contains(CpuBroadphase.Key((uint)hi, (uint)lo, kind)),
                            "step " + step + ": the pair (" + hi + ", " + lo + ") has " + counts[pair] + " contacts but is not a broadphase pair");
                    }
                }
                Assert.Greater(contactPairsSeen, 0, "the pyramid should touch the ground and itself");
            }
        }

        // ---- helpers ---------------------------------------------------------------------------------------------------------

        static void CheckScene(ReferenceSolver reference, string what)
        {
            SceneArrays scene = ConstraintTestScenes.FromReference(reference, Capacity);

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 256))
            using (var broadphase = new AvbdBroadphase(RequireShader(), Capacity, BigBin, BigBin))
            {
                Fill(buffers, constraints, scene);
                broadphase.Run(buffers, constraints);

                // The oracle works on what the GPU holds, so the check also covers the conversion of the scene to records.
                CpuBodyArrays held = ReadBodies(buffers);
                SortedSet<ulong> expected = CpuBroadphase.Pairs(held, scene.bodyCount, scene.constraints, scene.constraintCount);
                SortedSet<ulong> got = ToSet(broadphase.ReadPairs());
                AssertSameSets(expected, got, what);
                Assert.IsFalse(broadphase.Overflowed, what);

                // The oracle's pair set is the reference solver's, so the GPU's is too.
                AssertSameSets(ReferencePairs(reference), got, what + " (reference solver)");

                Assert.AreEqual(CpuBroadphase.MaxRadius(held, scene.bodyCount), broadphase.ReadMaxRadius(), 1e-5f, what + ": largest radius");
            }
        }

        // Spawns the scene's bodies (slot = reference body index) and adds its constraints. Returns the handles in slot order.
        static List<BodyHandle> Fill(AvbdBodyBuffers buffers, AvbdConstraintBuffers constraints, SceneArrays scene)
        {
            var handles = new List<BodyHandle>();
            foreach (BodyDescriptor d in scene.descriptors)
                handles.Add(buffers.Spawn(d));
            for (int i = 0; i < scene.constraintCount; i++)
                constraints.AddRecord(scene.constraints[i]);
            constraints.Rebuild();
            return handles;
        }

        static CpuBodyArrays ReadBodies(AvbdBodyBuffers buffers)
        {
            var bodies = new CpuBodyArrays(buffers.Capacity);
            buffers.ApplyEdits();
            buffers.State.GetData(bodies.State);
            buffers.Mass.GetData(bodies.Mass);
            buffers.Shape.GetData(bodies.Shape);
            buffers.Meta.GetData(bodies.Meta);
            return bodies;
        }

        static SortedSet<ulong> ToSet(PairRecord[] pairs)
        {
            var set = new SortedSet<ulong>();
            foreach (PairRecord pair in pairs)
                set.Add(CpuBroadphase.Key(pair));
            return set;
        }

        static void AssertSameSets(SortedSet<ulong> expected, SortedSet<ulong> actual, string what)
        {
            var missing = new List<string>();
            var extra = new List<string>();
            foreach (ulong key in expected)
                if (!actual.Contains(key) && missing.Count < 6) missing.Add(CpuBroadphase.Describe(key));
            foreach (ulong key in actual)
                if (!expected.Contains(key) && extra.Count < 6) extra.Add(CpuBroadphase.Describe(key));

            Assert.IsTrue(missing.Count == 0 && extra.Count == 0,
                what + ": expected " + expected.Count + " pairs, got " + actual.Count + ". Missing e.g. " + string.Join(" ", missing) + "; extra e.g. " + string.Join(" ", extra));
        }

        // Boxes of mixed sizes in a small volume around `origin` (negative coordinates included), one in six static, with joints, springs and
        // world pins between random pairs, some of them overlapping pairs.
        static void DenseRandom(ReferenceSolver s, int seed, int count, float3 origin = default)
        {
            var rng = new System.Random(seed);
            s.Clear();
            var bodies = new List<AvbdBody>();
            float extent = math.pow(count, 1f / 3f) * 1.4f;
            for (int i = 0; i < count; i++)
            {
                bool big = i % 10 == 3;
                float3 size = big ? new float3(Range(rng, 3f, 5f), Range(rng, 3f, 5f), Range(rng, 3f, 5f))
                                  : new float3(Range(rng, 0.3f, 1.4f), Range(rng, 0.3f, 1.4f), Range(rng, 0.3f, 1.4f));
                float density = i % 6 == 0 ? 0f : 1f;
                float3 position = origin + new float3(Range(rng, -extent, extent), Range(rng, -extent, extent), Range(rng, -extent, extent)) * 0.5f;
                bodies.Add(s.AddBody(size, density, 0.5f, position));
            }

            for (int k = 0; k < 18; k++)
            {
                int i = rng.Next(count);
                int j = rng.Next(count);
                if (i == j)
                    continue;
                if (k % 3 == 2)
                    s.AddSpring(bodies[i], bodies[j], float3.zero, float3.zero, 100f, 1f);
                else
                    s.AddJoint(bodies[i], bodies[j], float3.zero, float3.zero);
            }
            for (int k = 0; k < 3; k++)
                s.AddJoint(null, bodies[rng.Next(count)], origin, float3.zero);
        }

        static float Range(System.Random rng, float min, float max) => min + (float)rng.NextDouble() * (max - min);

        static ComputeShader RequireShader()
        {
            AvbdShaderSet set = AvbdShaderSet.Load();
            Assert.IsNotNull(set, "The AvbdShaderSet asset was not found.");
            Assert.IsNotNull(set.Broadphase, "The AvbdShaderSet has no broadphase shader.");
            return set.Broadphase;
        }

        static void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }
    }
}
