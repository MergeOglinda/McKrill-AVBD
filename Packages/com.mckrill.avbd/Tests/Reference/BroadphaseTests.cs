using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Stage B (broadphase). The CPU oracle (<see cref="CpuBroadphase"/>) is checked against the reference solver's own pair loop: it finds
    /// the same pairs except the ones whose boxes are visibly apart, and those must be pairs without a contact. The GPU must then find
    /// exactly the oracle's set of pairs (the order is unspecified), on the body state it holds.
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

        [Test]
        public void Oracle_FindsTheReferenceSolversPairs_OnRotatedBoxes()
        {
            int dropped = 0;
            for (int seed = 1; seed <= 10; seed++)
            {
                var solver = new ReferenceSolver();
                DenseRandom(solver, 300 + seed, 70, default, true);
                dropped += AssertOracleMatchesReference(solver, "dense rotated scene " + seed);
            }
            Assert.Greater(dropped, 0, "bounding spheres of rotated boxes overlap more often than the boxes: some pairs must be dropped");
        }

        // The oracle finds the reference solver's pairs except the ones the box test drops, and every dropped pair is a pair without a contact.
        // Returns the number of dropped pairs.
        static int AssertOracleMatchesReference(ReferenceSolver solver, string what)
        {
            SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);
            SortedSet<ulong> oracle = CpuBroadphase.Pairs(scene.bodies, scene.bodyCount, scene.constraints, scene.constraintCount);
            SortedSet<ulong> expected = ReferencePairs(solver);

            int dropped = AssertOnlyApartPairsDropped(expected, oracle, solver, what);
            if (what.StartsWith("dense"))
                Assert.Greater(expected.Count, 20, what + ": the scene should have many pairs");
            return dropped;
        }

        // `got` is a subset of the reference solver's pairs (spheres only), and every pair that is missing is one whose boxes the narrow phase
        // finds no contact for, so leaving it out changes no manifold. Returns the number of missing pairs.
        static int AssertOnlyApartPairsDropped(SortedSet<ulong> referencePairs, SortedSet<ulong> got, ReferenceSolver solver, string what)
        {
            foreach (ulong key in got)
                Assert.IsTrue(referencePairs.Contains(key), what + ": " + CpuBroadphase.Describe(key) + " is not a pair of the reference solver");

            var contacts = new AvbdManifold.Contact[BoxCollision.MaxContacts];
            float3x3 basis = float3x3.identity;
            int dropped = 0;
            foreach (ulong key in referencePairs)
            {
                if (got.Contains(key))
                    continue;
                dropped++;
                int hi = (int)(key >> 34);
                int lo = (int)((key >> 2) & 0xFFFFFFFFu);
                int count = BoxCollision.Collide(solver.Bodies[hi], solver.Bodies[lo], contacts, ref basis);
                Assert.AreEqual(0, count, what + ": the pair " + CpuBroadphase.Describe(key) + " was dropped but its boxes have " + count + " contacts");
            }
            return dropped;
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

        // ---- the box test against the narrow phase ------------------------------------------------------------------------

        // The reason the test may drop a pair: whenever it separates two boxes, the reference narrow phase finds no contact for them.
        [Test]
        public void BoxesSeparated_NeverDropsAPairThatHasContacts()
        {
            var rng = new System.Random(11);
            var contacts = new AvbdManifold.Contact[BoxCollision.MaxContacts];
            float3x3 basis = float3x3.identity;
            int separated = 0, colliding = 0, nearMisses = 0;

            for (int i = 0; i < 40000; i++)
            {
                float3 sizeA = new float3(Range(rng, 0.1f, 4f), Range(rng, 0.1f, 4f), Range(rng, 0.1f, 4f));
                float3 sizeB = new float3(Range(rng, 0.1f, 4f), Range(rng, 0.1f, 4f), Range(rng, 0.1f, 4f));
                var a = new AvbdBody(sizeA, 1f, 0.5f, float3.zero, float3.zero);
                var b = new AvbdBody(sizeB, 1f, 0.5f, float3.zero, float3.zero);
                a.PositionAng = quaternion.EulerXYZ(Range(rng, 0f, 6f), Range(rng, 0f, 6f), Range(rng, 0f, 6f));
                b.PositionAng = i % 4 == 0 ? quaternion.identity : quaternion.EulerXYZ(Range(rng, 0f, 6f), Range(rng, 0f, 6f), Range(rng, 0f, 6f));

                // Within the reach of the bounding spheres, so the pair is a candidate; a quarter of the pairs are pushed close to touching.
                float reach = a.Radius + b.Radius;
                float3 direction = math.normalize(new float3(Range(rng, -1f, 1f), Range(rng, -1f, 1f), Range(rng, -1f, 1f)) + 1e-4f);
                float distance = Range(rng, 0f, reach);
                if (i % 4 == 1)
                    distance = reach * Range(rng, 0.3f, 0.7f);
                b.PositionLin = direction * distance;

                bool apart = CpuBroadphase.BoxesSeparated(a.PositionLin, a.PositionAng, sizeA, b.PositionLin, b.PositionAng, sizeB);
                int count = BoxCollision.Collide(a, b, contacts, ref basis);
                if (apart)
                {
                    separated++;
                    Assert.AreEqual(0, count, "case " + i + ": separated boxes with " + count + " contacts (sizes " + sizeA + ", " + sizeB + ", offset " + b.PositionLin + ")");
                }
                if (count > 0)
                    colliding++;
                else if (!apart)
                    nearMisses++;
            }

            // The test should also be worth having: most apart pairs of this kind are caught, and plenty of pairs do collide.
            Assert.Greater(separated, 5000, "separated pairs");
            Assert.Greater(colliding, 2000, "colliding pairs");
            Debug.Log("BoxesSeparated: " + separated + " dropped, " + colliding + " colliding, " + nearMisses + " apart pairs the test keeps (edge axes, or within the pad)");
        }

        [Test]
        public void BoxesSeparated_UsesThePad()
        {
            float3 size = new float3(1f, 1f, 1f);
            quaternion none = quaternion.identity;
            float pad = AvbdBroadphase.SeparationPad;

            Assert.IsFalse(CpuBroadphase.BoxesSeparated(float3.zero, none, size, new float3(1f + 0.5f * pad, 0f, 0f), none, size), "inside the pad");
            Assert.IsTrue(CpuBroadphase.BoxesSeparated(float3.zero, none, size, new float3(1f + 2f * pad, 0f, 0f), none, size), "beyond the pad");
            Assert.IsFalse(CpuBroadphase.BoxesSeparated(float3.zero, none, size, new float3(0.9f, 0f, 0f), none, size), "overlapping");

            // A turned box: its face axis separates what the other box's axes alone would not.
            quaternion turned = quaternion.AxisAngle(math.normalize(new float3(1f, 1f, 0f)), 0.7f);
            Assert.IsFalse(CpuBroadphase.BoxesSeparated(float3.zero, none, size, new float3(0.2f, 0.2f, 0.2f), turned, size), "overlapping, turned");
            Assert.IsTrue(CpuBroadphase.BoxesSeparated(float3.zero, none, size, new float3(0f, 0f, 3f), turned, size), "apart, turned");

            // A NaN pose does not separate: the pair stays a candidate.
            Assert.IsFalse(CpuBroadphase.BoxesSeparated(float3.zero, none, size, new float3(float.NaN, 0f, 0f), none, size), "NaN");
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

        [Test]
        public void Gpu_MatchesTheOracle_OnRotatedBoxes()
        {
            RequireCompute();
            for (int seed = 1; seed <= 8; seed++)
            {
                var solver = new ReferenceSolver();
                DenseRandom(solver, 400 + seed, 120, default, true);
                CheckScene(solver, "dense rotated scene " + seed);
            }
        }

        // Bars whose bounding spheres all overlap while the boxes mostly do not. Slots: 0 and 1 and 2 are bars of 4 x 0.2 x 0.2 on top of each other
        // (spacings 0.6 and 0.2005 above bar 0), 3 is a big static slab below (a "large" static body, tested against every solved body), 4 a small
        // static cube above. A gap of 0.0005 is inside the pad, so those pairs stay candidates; every other pair is far enough apart to go.
        [Test]
        public void Gpu_DropsPairsOfBoxesThatAreApart_AndKeepsTheOnesWithinThePad()
        {
            RequireCompute();
            var reference = new ReferenceSolver();
            reference.AddBody(new float3(4f, 0.2f, 0.2f), 1f, 0.5f, new float3(0f, 0f, 0f));         // 0
            reference.AddBody(new float3(4f, 0.2f, 0.2f), 1f, 0.5f, new float3(0f, 0.6f, 0f));       // 1: 0.4 above bar 0
            reference.AddBody(new float3(4f, 0.2f, 0.2f), 1f, 0.5f, new float3(0f, 0.2005f, 0f));    // 2: 0.0005 above bar 0
            reference.AddBody(new float3(20f, 1f, 20f), 0f, 0.5f, new float3(0f, -0.6005f, 0f));     // 3: slab, its top is 0.0005 below bar 0
            reference.AddBody(new float3(1f, 1f, 1f), 0f, 0.5f, new float3(0f, 1.5f, 0f));           // 4: cube, 0.3 above bar 1

            SceneArrays scene = ConstraintTestScenes.FromReference(reference, Capacity);
            SortedSet<ulong> expected = new SortedSet<ulong>
            {
                CpuBroadphase.Key(2, 0, PairKind.DynamicDynamic),
                CpuBroadphase.Key(3, 0, PairKind.DynamicStatic),
            };

            // The spheres of all five overlap (the sphere test alone would give many more pairs).
            Assert.Greater(ReferencePairs(reference).Count, expected.Count + 4);
            AssertSameSets(expected, CpuBroadphase.Pairs(scene.bodies, scene.bodyCount, scene.constraints, scene.constraintCount), "oracle");

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 256))
            using (var broadphase = new AvbdBroadphase(RequireShader(), Capacity, BigBin, BigBin))
            {
                Fill(buffers, constraints, scene);
                broadphase.Run(buffers, constraints);
                AssertSameSets(expected, ToSet(broadphase.ReadPairs()), "gpu");
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

        // Static bodies that fit a grid cell are found through the grid, the ones larger than every solved body through the list: a floor of small
        // static tiles, a big slab, moving kinematic bodies and solved bodies in a range of sizes.
        [Test]
        public void Gpu_MatchesTheOracle_WithManyStaticBodiesOfDifferentSizes()
        {
            RequireCompute();
            const int capacity = 2048;
            using (var buffers = new AvbdBodyBuffers(capacity))
            using (var broadphase = new AvbdBroadphase(RequireShader(), capacity, 1 << 16, 1 << 16))
            {
                for (int x = 0; x < 20; x++)
                    for (int z = 0; z < 20; z++)
                        buffers.Spawn(BodyDescriptor.Box(new float3(1f), 0f, new float3(x * 0.9f, 0f, z * 0.9f)));            // 400 small static tiles
                buffers.Spawn(BodyDescriptor.Box(new float3(200f, 1f, 200f), 0f, new float3(9f, -1f, 9f)));                   // a large static slab
                buffers.Spawn(BodyDescriptor.Box(new float3(6f, 1f, 1f), 0f, new float3(4f, 3f, 4f)));                        // a static body a bit larger than the solved ones

                var rng = new System.Random(11);
                for (int i = 0; i < 300; i++)
                {
                    float s = Range(rng, 0.4f, 1.2f);
                    buffers.Spawn(BodyDescriptor.Box(new float3(s, s * 0.7f, s * 1.3f), 1f, new float3(Range(rng, -1f, 19f), Range(rng, 0.2f, 3f), Range(rng, -1f, 19f))));
                }

                var kinematic = new BodyHandle[4];
                for (int i = 0; i < kinematic.Length; i++)
                {
                    BodyDescriptor descriptor = BodyDescriptor.Box(new float3(1.5f), 1f, new float3(3f + 4f * i, 0.8f, 5f));
                    descriptor.kinematic = true;
                    kinematic[i] = buffers.Spawn(descriptor);
                }

                broadphase.Run(buffers, null);
                SortedSet<ulong> got = ToSet(broadphase.ReadPairs());
                AssertSameSets(CpuBroadphase.Pairs(ReadBodies(buffers), buffers.HighWaterMark, new ConstraintRecord[0], 0), got, "static scene");
                Assert.Greater(broadphase.StaticPairCount, 300, "the solved bodies should touch many static ones");
                Assert.Greater(broadphase.DynamicPairCount, 50, "and each other");

                // The kinematic bodies move: the static grid is rebuilt by every run, so the pairs follow them.
                for (int i = 0; i < kinematic.Length; i++)
                    buffers.SetPose(kinematic[i], new float3(15f - 3f * i, 1.2f, 12f - i), quaternion.identity);
                broadphase.Run(buffers, null);
                got = ToSet(broadphase.ReadPairs());
                AssertSameSets(CpuBroadphase.Pairs(ReadBodies(buffers), buffers.HighWaterMark, new ConstraintRecord[0], 0), got, "after the kinematic bodies moved");
            }
        }

        [Test]
        public void Gpu_SkipsPairsJoinedByAJointOrSpring_UntilTheListsDropABrokenOne_ButNotByAContact()
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

                // A joint that fractured on the GPU still excludes its pair until the lists are rebuilt without it: the broadphase reads the
                // lists. With contacts the step does that (AvbdConstraintBuffers reads the broken flags back and RefreshContactTopology
                // drops the joint), which is the reference solver's timing (it removes a broken joint one step after it broke).
                var records = new ConstraintRecord[constraints.Records.count];
                constraints.ReadRecords(records);
                records[0].ident.z |= (uint)ConstraintFlags.Broken;
                constraints.WriteRecords(records);
                broadphase.Run(buffers, constraints);
                AssertSameSets(expected, ToSet(broadphase.ReadPairs()), "after the joint broke");

                AssertSameSets(CpuBroadphase.Pairs(ReadBodies(buffers), 4, records, 4), ToSet(broadphase.ReadPairs()), "oracle");
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
                Assert.AreEqual(2, broadphase.ReadClassification().LiveCount);
                Assert.AreEqual(0, broadphase.ReadClassification().SolvedCount);

                buffers.Spawn(BodyDescriptor.Box(new float3(2f, 0.01f, 0.01f), 1f, new float3(0.1f, 0f, 0f))); // a thin rod: radius 1
                broadphase.Run(buffers, null);
                Assert.AreEqual(2, broadphase.ReadPairs().Length);
                float bound = broadphase.ReadMaxRadius(); // the upper edge of the bin of the largest small body: the rod, radius 1
                Assert.GreaterOrEqual(bound, 1f);
                Assert.LessOrEqual(bound, 1.06f);
            }
        }

        // ---- the grid ----------------------------------------------------------------------------------------------------

        // Many static tiles and a few solved bodies that are all about the same size, so the median is large; small solved bodies that move (the fine
        // level); solved bodies as large as the tiles (the coarse level); small static pieces (fine static bodies that coarse movers must still find,
        // through BpPairsFixed); and a slab (large). The pair set is the oracle's, and the classification the model's.
        [Test]
        public void Gpu_TwoLevels_FindsThePairsAcrossLevels()
        {
            RequireCompute();
            const int capacity = 1024;
            using (var buffers = new AvbdBodyBuffers(capacity))
            using (var broadphase = new AvbdBroadphase(RequireShader(), capacity, 1 << 16, 1 << 16))
            {
                var rng = new System.Random(21);
                for (int x = 0; x < 15; x++)
                    for (int z = 0; z < 15; z++)
                        buffers.Spawn(BodyDescriptor.Box(new float3(2f), 0f, new float3(x * 2.1f, 0f, z * 2.1f)));              // 225 static tiles: radius 1.73
                buffers.Spawn(BodyDescriptor.Box(new float3(300f, 1f, 300f), 0f, new float3(15f, -1f, 15f)));                      // a slab: large
                for (int i = 0; i < 60; i++)
                    buffers.Spawn(BodyDescriptor.Box(new float3(0.5f), 1f, new float3(Range(rng, 0f, 30f), Range(rng, 0.5f, 3f), Range(rng, 0f, 30f)))); // small movers: radius 0.43
                for (int i = 0; i < 12; i++)
                    buffers.Spawn(BodyDescriptor.Box(new float3(2f), 1f, new float3(Range(rng, 0f, 30f), Range(rng, 1f, 4f), Range(rng, 0f, 30f))));   // movers as large as the tiles: coarse level
                for (int i = 0; i < 20; i++)
                    buffers.Spawn(BodyDescriptor.Box(new float3(0.5f), 0f, new float3(Range(rng, 0f, 30f), Range(rng, 0.5f, 3f), Range(rng, 0f, 30f)))); // small static pieces: fine level
                // A coarse mover on a fine static piece, which only BpPairsFixed can find, and a fine mover on a coarse static tile.
                buffers.Spawn(BodyDescriptor.Box(new float3(0.5f), 0f, new float3(100f, 0f, 100f)));
                buffers.Spawn(BodyDescriptor.Box(new float3(2f), 1f, new float3(100.2f, 0f, 100f)));
                buffers.Spawn(BodyDescriptor.Box(new float3(2f), 0f, new float3(120f, 0f, 120f)));
                buffers.Spawn(BodyDescriptor.Box(new float3(0.5f), 1f, new float3(120.2f, 0f, 120f)));

                broadphase.Run(buffers, null);
                CpuBodyArrays held = ReadBodies(buffers);
                CpuBroadphase.GridModel model = CpuBroadphase.Classification(held, buffers.HighWaterMark);
                BroadphaseClassification classification = broadphase.ReadClassification();
                Assert.IsTrue(model.TwoLevels, "the scene is meant to have two levels");
                Assert.IsTrue(classification.TwoLevels);
                Assert.AreEqual(model.LargeCount, classification.LargeCount);
                Assert.AreEqual(1, classification.LargeCount, "the slab");
                Assert.Less(classification.FineCell, classification.CoarseCell);
                Assert.AreEqual(model.SmallBin, classification.SmallBin);
                Assert.AreEqual(model.FineBin, classification.FineBin);

                SortedSet<ulong> got = ToSet(broadphase.ReadPairs());
                AssertSameSets(CpuBroadphase.Pairs(held, buffers.HighWaterMark, new ConstraintRecord[0], 0), got, "two levels");
                Assert.IsFalse(broadphase.Overflowed);

                int fineStatic = buffers.HighWaterMark - 4, coarseMoverOnFine = buffers.HighWaterMark - 3;
                int coarseStatic = buffers.HighWaterMark - 2, fineMoverOnCoarse = buffers.HighWaterMark - 1;
                Assert.IsTrue(got.Contains(CpuBroadphase.Key((uint)coarseMoverOnFine, (uint)fineStatic, PairKind.DynamicStatic)), "a coarse mover and a fine static piece");
                Assert.IsTrue(got.Contains(CpuBroadphase.Key((uint)fineMoverOnCoarse, (uint)coarseStatic, PairKind.DynamicStatic)), "a fine mover and a coarse static tile");
            }
        }

        // More large candidates than the large list takes: at most MaxLarge of the largest bodies are large, the rest are small and size the cells.
        [Test]
        public void Gpu_LargeBodies_AreAtMostTheLargestFewAndStillFindTheirPairs()
        {
            RequireCompute();
            const int capacity = 1024;
            using (var buffers = new AvbdBodyBuffers(capacity))
            using (var broadphase = new AvbdBroadphase(RequireShader(), capacity, 1 << 17, 1 << 17))
            {
                var rng = new System.Random(33);
                for (int i = 0; i < 300; i++)
                    buffers.Spawn(BodyDescriptor.Box(new float3(Range(rng, 0.9f, 1.1f)), i % 4 == 0 ? 0f : 1f, new float3(Range(rng, -40f, 40f), Range(rng, 0f, 3f), Range(rng, -40f, 40f))));
                for (int i = 0; i < 90; i++)                                            // 90 bodies far above twice the median, all different
                {
                    float size = 8f * math.pow(1.045f, i);
                    buffers.Spawn(BodyDescriptor.Box(new float3(size, size * 0.2f, size), i % 3 == 0 ? 1f : 0f, new float3(Range(rng, -30f, 30f), Range(rng, 0f, 3f), Range(rng, -30f, 30f))));
                }

                broadphase.Run(buffers, null);
                CpuBodyArrays held = ReadBodies(buffers);
                CpuBroadphase.GridModel model = CpuBroadphase.Classification(held, buffers.HighWaterMark);
                BroadphaseClassification classification = broadphase.ReadClassification();
                Assert.AreEqual(model.LargeCount, classification.LargeCount);
                Assert.Greater(classification.LargeCount, 0);
                Assert.LessOrEqual(classification.LargeCount, AvbdBroadphase.MaxLarge);
                AssertSameSets(CpuBroadphase.Pairs(held, buffers.HighWaterMark, new ConstraintRecord[0], 0), ToSet(broadphase.ReadPairs()), "many large bodies");
            }
        }

        // The classification is made again by every run, from the bodies as they are: a body that grows becomes large, and shrinks back.
        [Test]
        public void Gpu_Classification_FollowsTheBodies()
        {
            RequireCompute();
            const int capacity = 256;
            using (var buffers = new AvbdBodyBuffers(capacity))
            using (var broadphase = new AvbdBroadphase(RequireShader(), capacity, BigBin, BigBin))
            {
                var rng = new System.Random(9);
                var handles = new List<BodyHandle>();
                for (int i = 0; i < 80; i++)
                    handles.Add(buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(Range(rng, -6f, 6f), Range(rng, 0f, 3f), Range(rng, -6f, 6f)))));

                broadphase.Run(buffers, null);
                Assert.AreEqual(0, broadphase.ReadClassification().LargeCount);
                AssertSameSets(CpuBroadphase.Pairs(ReadBodies(buffers), buffers.HighWaterMark, new ConstraintRecord[0], 0), ToSet(broadphase.ReadPairs()), "all small");

                buffers.SetShape(handles[5], new float3(60f, 1f, 60f), 1f);
                broadphase.Run(buffers, null);
                Assert.AreEqual(1, broadphase.ReadClassification().LargeCount);
                AssertSameSets(CpuBroadphase.Pairs(ReadBodies(buffers), buffers.HighWaterMark, new ConstraintRecord[0], 0), ToSet(broadphase.ReadPairs()), "one large body");

                buffers.SetShape(handles[5], new float3(1f), 1f);
                broadphase.Run(buffers, null);
                Assert.AreEqual(0, broadphase.ReadClassification().LargeCount);
                AssertSameSets(CpuBroadphase.Pairs(ReadBodies(buffers), buffers.HighWaterMark, new ConstraintRecord[0], 0), ToSet(broadphase.ReadPairs()), "small again");
            }
        }

        // More bodies than the table has cells: bodies of different cells share a hashed cell and the probe must drop them.
        [Test]
        public void Gpu_Grid_HandlesMoreBodiesThanCellsOfTheScanBlocks()
        {
            RequireCompute();
            const int capacity = 3000; // a table of 8192 cells: eight scan blocks
            using (var buffers = new AvbdBodyBuffers(capacity))
            using (var broadphase = new AvbdBroadphase(RequireShader(), capacity, 1 << 18, 1 << 18))
            {
                var rng = new System.Random(77);
                for (int i = 0; i < 2900; i++)
                    buffers.Spawn(BodyDescriptor.Box(new float3(Range(rng, 0.8f, 1.2f)), i % 5 == 0 ? 0f : 1f, new float3(Range(rng, -30f, 30f), Range(rng, 0f, 20f), Range(rng, -30f, 30f))));

                broadphase.Run(buffers, null);
                AssertSameSets(CpuBroadphase.Pairs(ReadBodies(buffers), buffers.HighWaterMark, new ConstraintRecord[0], 0), ToSet(broadphase.ReadPairs()), "a crowded table");
                Assert.IsFalse(broadphase.Overflowed);
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
            using (var solver = new AvbdSolver(buffers, ZUpSettings(), null, constraints))
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
            using (var constraints = new AvbdConstraintBuffers(buffers, 4, null, 512))
            using (var solver = new AvbdSolver(buffers, ZUpSettings(), null, constraints))
            {
                Fill(buffers, constraints, scene);
                solver.EnableBroadphase(BigBin, BigBin);

                int contactPairsSeen = 0;
                for (int step = 0; step < 40; step++) // the boxes start 0.25 above the ground: they touch it after about 13 steps
                {
                    solver.Step();
                    SortedSet<ulong> candidates = ToSet(solver.Broadphase.ReadPairs());
                    ContactSnapshot snapshot = constraints.ReadContacts();
                    CpuBodyArrays held = ReadBodies(buffers);
                    for (int slot = 0; slot < snapshot.Counts.Length; slot++)
                    {
                        if (snapshot.Counts[slot] == 0u)
                            continue;
                        contactPairsSeen++;
                        int hi = (int)snapshot.Bodies[slot].x;
                        int lo = (int)snapshot.Bodies[slot].y;
                        bool solvedHi = CpuBroadphase.Classify(held, hi) == CpuBroadphase.BodyClass.Solved;
                        PairKind kind = solvedHi && CpuBroadphase.Classify(held, lo) == CpuBroadphase.BodyClass.Solved
                            ? PairKind.DynamicDynamic : PairKind.DynamicStatic;
                        Assert.IsTrue(candidates.Contains(CpuBroadphase.Key((uint)hi, (uint)lo, kind)),
                            "step " + step + ": the pair (" + hi + ", " + lo + ") has " + snapshot.Counts[slot] + " contacts but is not a broadphase pair");
                    }
                }
                Assert.Greater(contactPairsSeen, 0, "the pyramid should have touched the ground within 40 steps");
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

                // The oracle's pair set is the reference solver's without the pairs of boxes that are apart, so the GPU's is too.
                AssertOnlyApartPairsDropped(ReferencePairs(reference), got, reference, what + " (reference solver)");

                // The classification is the model's, up to a bin (the rounding of log2 on the GPU can put a radius on a bin edge into the next bin).
                CpuBroadphase.GridModel model = CpuBroadphase.Classification(held, scene.bodyCount);
                BroadphaseClassification classification = broadphase.ReadClassification();
                Assert.AreEqual(model.LiveCount, classification.LiveCount, what + ": live bodies");
                Assert.AreEqual(model.SolvedCount, classification.SolvedCount, what + ": solved bodies");
                Assert.AreEqual(1f, classification.MaxSmallRadius / model.MaxSmallRadius, 0.07f, what + ": largest small radius");
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
            buffers.Flags.GetData(bodies.Flags);
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
        static void DenseRandom(ReferenceSolver s, int seed, int count, float3 origin = default, bool rotate = false)
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
                AvbdBody body = s.AddBody(size, density, 0.5f, position);
                if (rotate)
                    body.PositionAng = quaternion.EulerXYZ(Range(rng, 0f, 6f), Range(rng, 0f, 6f), Range(rng, 0f, 6f));
                bodies.Add(body);
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

        // The upstream scenes are z-up; the default settings have a y-up gravity, under which the pyramid would fall sideways past the ground.
        static AvbdSolverSettings ZUpSettings()
        {
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            settings.gravity = new float3(0f, 0f, -10f);
            return settings;
        }

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
