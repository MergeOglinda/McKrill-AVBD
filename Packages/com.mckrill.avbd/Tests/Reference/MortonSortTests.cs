using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Stage S (Morton sort). The CPU oracle (<see cref="CpuMortonSort"/>) is checked against the definition of the key; the GPU must then produce
    /// a stable order by key that is a permutation of the slots, with the keys of the oracle (except for bodies on a cell boundary, where the GPU's
    /// division may round the other way). The consumers (broadphase pairs, color buckets, a whole step) must give the same results with the order
    /// as without it.
    /// </summary>
    public class MortonSortTests
    {
        const int Capacity = 4096;
        const int BigBin = 1 << 16;

        // ---- the oracle ----------------------------------------------------------------------------------------------------

        // The shader's ExpandBits, with its multiplications and masks.
        static uint ShaderExpandBits(uint v)
        {
            v = (v * 0x00010001u) & 0xFF0000FFu;
            v = (v * 0x00000101u) & 0x0F00F00Fu;
            v = (v * 0x00000011u) & 0xC30C30C3u;
            v = (v * 0x00000005u) & 0x49249249u;
            return v;
        }

        [Test]
        public void Oracle_ExpandBits_EqualsTheShadersMultiplications_ForEveryTenBitValue()
        {
            for (uint v = 0; v < 1024u; v++)
                Assert.AreEqual(CpuMortonSort.ExpandBits(v), ShaderExpandBits(v), "value " + v);
        }

        [Test]
        public void Oracle_Key_InterleavesTheAxes_AndWrapsAfter1024Cells()
        {
            Assert.AreEqual(0u, CpuMortonSort.Key(new int3(0, 0, 0)));
            Assert.AreEqual(4u, CpuMortonSort.Key(new int3(1, 0, 0)), "x is the highest bit of each triple");
            Assert.AreEqual(2u, CpuMortonSort.Key(new int3(0, 1, 0)));
            Assert.AreEqual(1u, CpuMortonSort.Key(new int3(0, 0, 1)));
            Assert.AreEqual(7u, CpuMortonSort.Key(new int3(1, 1, 1)));
            Assert.AreEqual(32u, CpuMortonSort.Key(new int3(2, 0, 0)), "bit 1 of x is bit 5 of the key");
            Assert.AreEqual(0x3FFFFFFFu, CpuMortonSort.Key(new int3(1023, 1023, 1023)));
            Assert.AreEqual(CpuMortonSort.Key(new int3(1023, 1023, 1023)), CpuMortonSort.Key(new int3(-1, -1, -1)), "negative cells wrap");
            Assert.AreEqual(CpuMortonSort.Key(new int3(5, 6, 7)), CpuMortonSort.Key(new int3(5 + 1024, 6 - 2048, 7 + 4096)), "a period of 1024 cells");
            Assert.LessOrEqual(CpuMortonSort.Key(new int3(1000, 20, 300)), CpuMortonSort.InvalidKey);
        }

        [Test]
        public void Oracle_Order_SortsByKey_AndKeepsTheSlotOrderOfEqualKeys()
        {
            var rng = new System.Random(3);
            var keys = new uint[500];
            for (int i = 0; i < keys.Length; i++)
                keys[i] = (uint)rng.Next(0, 40); // many ties

            uint[] order = CpuMortonSort.Order(keys);
            Assert.AreEqual(keys.Length, order.Length);
            var seen = new bool[keys.Length];
            for (int i = 0; i < order.Length; i++)
            {
                Assert.IsFalse(seen[order[i]], "slot " + order[i] + " twice");
                seen[order[i]] = true;
                if (i > 0)
                    Assert.IsTrue(keys[order[i - 1]] < keys[order[i]] || (keys[order[i - 1]] == keys[order[i]] && order[i - 1] < order[i]), "position " + i);
            }
        }

        [Test]
        public void Oracle_Keys_GiveBodiesThatAreNotSolvedTheInvalidKey()
        {
            var solver = new ReferenceSolver();
            BroadphaseTests.DenseRandom(solver, 4, 80);
            SceneArrays scene = ConstraintTestScenes.FromReference(solver, 128);
            uint[] keys = CpuMortonSort.Keys(scene.bodies, scene.bodyCount);

            bool[] solved = scene.Solved();
            int staticBodies = 0;
            for (int s = 0; s < scene.bodyCount; s++)
            {
                if (solved[s])
                    Assert.AreNotEqual(CpuMortonSort.InvalidKey, keys[s], "solved body " + s);
                else
                {
                    Assert.AreEqual(CpuMortonSort.InvalidKey, keys[s], "static body " + s);
                    staticBodies++;
                }
            }
            Assert.Greater(staticBodies, 5, "the scene should have static bodies");

            uint[] order = CpuMortonSort.Order(keys);
            for (int i = 0; i < order.Length - staticBodies; i++)
                Assert.IsTrue(solved[order[i]], "the solved bodies come first (position " + i + ")");
        }

        // ---- the GPU against the oracle ------------------------------------------------------------------------------------

        [Test]
        public void Gpu_SortsLikeTheOracle_OnDenseScenes()
        {
            BroadphaseTests.RequireCompute();
            foreach (int count in new[] { 70, 600, 2500 })
                foreach (float3 origin in new[] { float3.zero, new float3(-5000f, 3000f, -7000f) })
                {
                    var solver = new ReferenceSolver();
                    BroadphaseTests.DenseRandom(solver, 11 + count, count, origin);
                    SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);

                    using (var buffers = new AvbdBodyBuffers(Capacity))
                    using (var sort = new AvbdMortonSort(RequireShader(), Capacity))
                    {
                        Spawn(buffers, scene);
                        sort.Run(buffers);
                        AssertSorted(buffers, sort, count + " bodies around " + origin);
                    }
                }
        }

        // Tile boundaries: the last tile is partial, or full, or has one key.
        [Test]
        public void Gpu_SortsBodyCountsAroundTheTileSize()
        {
            BroadphaseTests.RequireCompute();
            foreach (int count in new[] { 1, 2, 255, 256, 257, 511, 512, 513, 1000 })
                using (var buffers = new AvbdBodyBuffers(1024))
                using (var sort = new AvbdMortonSort(RequireShader(), 1024))
                {
                    var rng = new System.Random(count);
                    for (int i = 0; i < count; i++)
                        buffers.Spawn(BodyDescriptor.Box(new float3(0.5f + (float)rng.NextDouble()), 1f,
                            new float3(Range(rng, -30f, 30f), Range(rng, -30f, 30f), Range(rng, -30f, 30f))));

                    sort.Run(buffers);
                    Assert.AreEqual(count, sort.SortedCount);
                    AssertSorted(buffers, sort, count + " bodies");
                }
        }

        // Equal keys must come out in slot order: every pass keeps the order of equal digits.
        [Test]
        public void Gpu_KeepsTheSlotOrderOfBodiesInTheSameCell()
        {
            BroadphaseTests.RequireCompute();
            using (var buffers = new AvbdBodyBuffers(1024))
            using (var sort = new AvbdMortonSort(RequireShader(), 1024))
            {
                for (int i = 0; i < 700; i++)
                    buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0.1f, 0.2f, 0.3f)));

                sort.Run(buffers);
                (uint[] keys, uint[] order) = sort.ReadSorted();
                for (int i = 0; i < order.Length; i++)
                {
                    Assert.AreEqual(keys[0], keys[i], "one cell, one key");
                    Assert.AreEqual((uint)i, order[i], "slot order at position " + i);
                }
            }
        }

        [Test]
        public void Gpu_PutsStaticKinematicAndRemovedBodiesLast()
        {
            BroadphaseTests.RequireCompute();
            using (var buffers = new AvbdBodyBuffers(64))
            using (var sort = new AvbdMortonSort(RequireShader(), 64))
            {
                BodyHandle[] handles = new BodyHandle[10];
                for (int i = 0; i < handles.Length; i++)
                {
                    BodyDescriptor d = BodyDescriptor.Box(new float3(1f), i == 2 ? 0f : 1f, new float3(i * 3f, 0f, 0f)); // slot 2 is static
                    d.kinematic = i == 4;
                    handles[i] = buffers.Spawn(d);
                }
                Assert.IsTrue(buffers.Remove(handles[6]));

                sort.Run(buffers);
                (uint[] keys, uint[] order) = sort.ReadSorted();
                Assert.AreEqual(10, order.Length);

                var notSolved = new HashSet<uint> { 2u, 4u, 6u };
                for (int i = 0; i < 7; i++)
                    Assert.IsFalse(notSolved.Contains(order[i]), "slot " + order[i] + " at position " + i + " should come after the solved bodies");
                for (int i = 7; i < 10; i++)
                {
                    Assert.IsTrue(notSolved.Contains(order[i]), "position " + i);
                    Assert.AreEqual(AvbdMortonSort.InvalidKey, keys[i]);
                }
                Assert.AreEqual(new uint[] { 2u, 4u, 6u }, new[] { order[7], order[8], order[9] }, "in slot order");
                AssertSorted(buffers, sort, "mixed bodies");
            }
        }

        [Test]
        public void Gpu_FollowsMovedBodies_AndCoversOnlyTheBodyCountItWasMadeFor()
        {
            BroadphaseTests.RequireCompute();
            using (var buffers = new AvbdBodyBuffers(256))
            using (var sort = new AvbdMortonSort(RequireShader(), 256))
            {
                var rng = new System.Random(9);
                var handles = new List<BodyHandle>();
                for (int i = 0; i < 100; i++)
                    handles.Add(buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(Range(rng, -20f, 20f), Range(rng, -20f, 20f), Range(rng, -20f, 20f)))));

                Assert.IsFalse(sort.Covers(100), "not run yet");
                sort.Run(buffers);
                Assert.IsTrue(sort.Covers(100));
                Assert.IsFalse(sort.Covers(101));
                AssertSorted(buffers, sort, "first run");
                int firstVersion = sort.Version;

                foreach (BodyHandle h in handles)
                    buffers.SetPose(h, new float3(Range(rng, -20f, 20f), Range(rng, -20f, 20f), Range(rng, -20f, 20f)), quaternion.identity);
                sort.Run(buffers);
                Assert.AreEqual(firstVersion + 1, sort.Version);
                AssertSorted(buffers, sort, "after moving every body");

                buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, float3.zero));
                buffers.ApplyEdits();
                Assert.IsFalse(sort.Covers(buffers.HighWaterMark), "a new slot: the order is for 100 bodies");
                sort.Run(buffers);
                Assert.IsTrue(sort.Covers(buffers.HighWaterMark));
                AssertSorted(buffers, sort, "after a spawn");
            }
        }

        [Test]
        public void Gpu_NearbyBodiesAreNearInTheOrder()
        {
            BroadphaseTests.RequireCompute();
            // A 12 x 12 x 12 lattice of unit boxes spawned in a scrambled order. Along the Morton curve the average distance between consecutive
            // bodies is close to one cell; in slot order it is several cells.
            const int n = 12;
            var cells = new List<int3>();
            for (int x = 0; x < n; x++)
                for (int y = 0; y < n; y++)
                    for (int z = 0; z < n; z++)
                        cells.Add(new int3(x, y, z));
            var rng = new System.Random(21);
            for (int i = cells.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (cells[i], cells[j]) = (cells[j], cells[i]);
            }

            using (var buffers = new AvbdBodyBuffers(2048))
            using (var sort = new AvbdMortonSort(RequireShader(), 2048))
            {
                foreach (int3 c in cells)
                    buffers.Spawn(BodyDescriptor.Box(new float3(0.9f), 1f, (float3)c * 1.5f + 0.1f));
                sort.Run(buffers);
                AssertSorted(buffers, sort, "lattice");

                (uint[] keys, uint[] order) = sort.ReadSorted();
                float inSlotOrder = 0f, inMortonOrder = 0f;
                for (int i = 1; i < order.Length; i++)
                {
                    inSlotOrder += math.distance((float3)cells[i - 1], (float3)cells[i]);
                    inMortonOrder += math.distance((float3)cells[(int)order[i - 1]], (float3)cells[(int)order[i]]);
                }
                Assert.Less(inMortonOrder, inSlotOrder * 0.5f, "consecutive bodies of the sorted order are much closer than consecutive slots");
                Assert.Less(inMortonOrder / (order.Length - 1), 3f, "mean distance between consecutive bodies, in cells");
            }
        }

        // ---- the consumers -------------------------------------------------------------------------------------------------

        [Test]
        public void Gpu_Broadphase_FindsTheSamePairs_InMortonOrder()
        {
            BroadphaseTests.RequireCompute();
            for (int seed = 1; seed <= 6; seed++)
            {
                var solver = new ReferenceSolver();
                BroadphaseTests.DenseRandom(solver, 200 + seed, 150);
                SceneArrays scene = ConstraintTestScenes.FromReference(solver, 256);

                using (var buffers = new AvbdBodyBuffers(256))
                using (var constraints = new AvbdConstraintBuffers(buffers, 256))
                using (var sort = new AvbdMortonSort(RequireShader(), 256))
                using (var broadphase = new AvbdBroadphase(RequireBroadphaseShader(), 256, BigBin, BigBin))
                {
                    BroadphaseTests.Fill(buffers, constraints, scene);

                    broadphase.Run(buffers, constraints);
                    SortedSet<ulong> plain = BroadphaseTests.ToSet(broadphase.ReadPairs());

                    sort.Run(buffers);
                    broadphase.Run(buffers, constraints, sort);
                    SortedSet<ulong> ordered = BroadphaseTests.ToSet(broadphase.ReadPairs());

                    CpuBodyArrays held = BroadphaseTests.ReadBodies(buffers);
                    SortedSet<ulong> expected = CpuBroadphase.Pairs(held, scene.bodyCount, scene.constraints, scene.constraintCount);
                    Assert.Greater(expected.Count, 20, "seed " + seed + ": the scene should have pairs");
                    BroadphaseTests.AssertSameSets(expected, ordered, "seed " + seed + " in Morton order");
                    BroadphaseTests.AssertSameSets(plain, ordered, "seed " + seed + " with and without the order");
                    Assert.IsFalse(broadphase.Overflowed);
                }
            }
        }

        // A sort made for another body count is ignored, not read out of range.
        [Test]
        public void Gpu_Broadphase_IgnoresASortForAnotherBodyCount()
        {
            BroadphaseTests.RequireCompute();
            var solver = new ReferenceSolver();
            BroadphaseTests.DenseRandom(solver, 77, 100);
            SceneArrays scene = ConstraintTestScenes.FromReference(solver, 256);

            using (var buffers = new AvbdBodyBuffers(256))
            using (var constraints = new AvbdConstraintBuffers(buffers, 256))
            using (var sort = new AvbdMortonSort(RequireShader(), 256))
            using (var broadphase = new AvbdBroadphase(RequireBroadphaseShader(), 256, BigBin, BigBin))
            {
                BroadphaseTests.Fill(buffers, constraints, scene);
                sort.Run(buffers); // for 100 bodies

                // Twenty more bodies close to the others; the sort is now out of date and does not list them.
                for (int i = 0; i < 20; i++)
                    buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(i * 0.2f, 0f, 0f)));
                broadphase.Run(buffers, constraints, sort);

                CpuBodyArrays held = BroadphaseTests.ReadBodies(buffers);
                SortedSet<ulong> expected = CpuBroadphase.Pairs(held, buffers.HighWaterMark, scene.constraints, scene.constraintCount);
                BroadphaseTests.AssertSameSets(expected, BroadphaseTests.ToSet(broadphase.ReadPairs()), "with a stale sort");
            }
        }

        [Test]
        public void Gpu_ColorBuckets_HaveTheSameBodiesInEveryColor_InMortonOrder()
        {
            BroadphaseTests.RequireCompute();
            for (int seed = 1; seed <= 4; seed++)
            {
                var solver = new ReferenceSolver();
                BroadphaseTests.DenseRandom(solver, 300 + seed, 200);
                SceneArrays scene = ConstraintTestScenes.FromReference(solver, 256);

                List<List<uint>> plain = BucketsOf(scene, null);
                using (var sortBuffers = new AvbdBodyBuffers(256))
                using (var sort = new AvbdMortonSort(RequireShader(), 256))
                {
                    Spawn(sortBuffers, scene);
                    sort.Run(sortBuffers);
                    List<List<uint>> ordered = BucketsOf(scene, sort);

                    Assert.Greater(plain.Count, 1, "seed " + seed + ": the scene should need several colors");
                    Assert.AreEqual(plain.Count, ordered.Count, "seed " + seed + ": number of colors");
                    for (int c = 0; c < plain.Count; c++)
                        CollectionAssert.AreEqual(plain[c], ordered[c], "seed " + seed + ": bodies of color " + c);
                }
            }
        }

        // The sorted bucket contents of every color. With a sort, the order is made on its own body buffers (same scene, same slots) and handed to
        // the constraints of this run; the sort only has to cover the same number of bodies.
        static List<List<uint>> BucketsOf(SceneArrays scene, AvbdMortonSort sort)
        {
            using (var buffers = new AvbdBodyBuffers(256))
            using (var constraints = new AvbdConstraintBuffers(buffers, 256))
            {
                Spawn(buffers, scene);
                for (int i = 0; i < scene.constraintCount; i++)
                    constraints.AddRecord(scene.constraints[i]);
                constraints.BodyOrder = sort;
                constraints.Rebuild();

                uint[] bucketed = constraints.ReadColorBodies();
                var result = new List<List<uint>>();
                for (int c = 0; c < constraints.ColorCount; c++)
                {
                    var bodies = new List<uint>();
                    for (int k = 0; k < constraints.ColorSize(c); k++)
                        bodies.Add(bucketed[constraints.ColorOffset(c) + k]);
                    bodies.Sort();
                    result.Add(bodies);
                }
                return result;
            }
        }

        [Test]
        public void Gpu_Step_GivesTheSameResult_WithAndWithoutTheSort()
        {
            BroadphaseTests.RequireCompute();
            BodyStateRecord[] plain = RunPyramid(false, 15, out int plainSorts);
            BodyStateRecord[] sorted = RunPyramid(true, 15, out int sortedSorts);

            Assert.AreEqual(0, plainSorts, "SortBodies = false: the sort never runs");
            Assert.AreEqual(15, sortedSorts, "the sort runs once per step");
            for (int s = 0; s < 22; s++)
            {
                float3 d = plain[s].position.xyz - sorted[s].position.xyz;
                Assert.Less(math.length(d), 1e-3f, "position of slot " + s + " after 15 steps");
            }
        }

        static BodyStateRecord[] RunPyramid(bool sortBodies, int steps, out int sorts)
        {
            var reference = new ReferenceSolver();
            AvbdScenes.Pyramid(reference, 6); // 21 boxes and a ground
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            settings.gravity = new float3(0f, 0f, -10f); // the upstream scenes are z-up
            SceneArrays scene = ConstraintTestScenes.FromReference(reference, 64);

            using (var buffers = new AvbdBodyBuffers(64))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4, null, 256))
            using (var solver = new AvbdSolver(buffers, settings, null, constraints))
            {
                Spawn(buffers, scene);
                solver.SortBodies = sortBodies;
                for (int i = 0; i < steps; i++)
                    solver.Step();

                sorts = solver.MortonSort != null ? solver.MortonSort.Version : 0;
                var state = new BodyStateRecord[64];
                buffers.State.GetData(state);
                return state;
            }
        }

        [Test]
        public void Solver_SortsOnlyWhileThereIsABroadphase_AndOnlyWhenSwitchedOn()
        {
            BroadphaseTests.RequireCompute();
            var reference = new ReferenceSolver();
            AvbdScenes.Stack(reference);
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            settings.gravity = new float3(0f, 0f, -10f);
            SceneArrays scene = ConstraintTestScenes.FromReference(reference, 64);

            // No contacts and no broadphase: nothing reads the order.
            using (var buffers = new AvbdBodyBuffers(64))
            using (var solver = new AvbdSolver(buffers, settings))
            {
                Spawn(buffers, scene);
                solver.SortBodies = true;
                solver.Step();
                Assert.IsNull(solver.MortonSort, "no broadphase, no sort");

                solver.SortBodies = true;
                solver.EnableBroadphase(1024);
                solver.Step();
                Assert.IsNotNull(solver.MortonSort, "a broadphase: the sort runs");
                Assert.AreEqual(1, solver.MortonSort.Version);
            }

            using (var buffers = new AvbdBodyBuffers(64))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4, null, 256))
            using (var solver = new AvbdSolver(buffers, settings, null, constraints))
            {
                Spawn(buffers, scene);
                Assert.IsFalse(solver.SortBodies, "off by default");
                solver.Step();
                Assert.IsNull(solver.MortonSort, "off: the sort is not even made");

                solver.SortBodies = true;
                solver.Step();
                solver.Step();
                Assert.AreEqual(2, solver.MortonSort.Version);
                Assert.IsTrue(solver.MortonSort.Covers(buffers.HighWaterMark));

                solver.SortBodies = false;
                solver.Step();
                Assert.AreEqual(2, solver.MortonSort.Version, "switched off: the sort does not run");

                solver.SortBodies = true;
                solver.Step();
                Assert.AreEqual(3, solver.MortonSort.Version);
            }
        }

        // ---- helpers -------------------------------------------------------------------------------------------------------

        static void Spawn(AvbdBodyBuffers buffers, SceneArrays scene)
        {
            foreach (BodyDescriptor d in scene.descriptors)
                buffers.Spawn(d);
            buffers.ApplyEdits();
        }

        // The result of the last run: a permutation of the slots, sorted by key, equal keys in slot order, with the oracle's keys (up to the cell
        // boundary rounding), and the oracle's largest radius.
        static void AssertSorted(AvbdBodyBuffers buffers, AvbdMortonSort sort, string what)
        {
            CpuBodyArrays held = BroadphaseTests.ReadBodies(buffers);
            int n = buffers.HighWaterMark;
            (uint[] keys, uint[] order) = sort.ReadSorted();
            Assert.AreEqual(n, order.Length, what + ": length");

            var seen = new bool[n];
            var bySlot = new uint[n];
            for (int i = 0; i < n; i++)
            {
                Assert.Less(order[i], (uint)n, what + ": slot at position " + i);
                Assert.IsFalse(seen[order[i]], what + ": slot " + order[i] + " twice");
                seen[order[i]] = true;
                bySlot[order[i]] = keys[i];
                if (i > 0)
                    Assert.IsTrue(keys[i - 1] < keys[i] || (keys[i - 1] == keys[i] && order[i - 1] < order[i]),
                        what + ": position " + i + " (keys " + keys[i - 1] + ", " + keys[i] + ", slots " + order[i - 1] + ", " + order[i] + ")");
            }

            float cellSize = CpuMortonSort.CellSize(held, n);
            uint[] expected = CpuMortonSort.Keys(held, n);
            int compared = 0;
            for (int s = 0; s < n; s++)
            {
                if (expected[s] != CpuMortonSort.InvalidKey && bySlot[s] != expected[s] && CpuMortonSort.NearCellBoundary(held, s, cellSize, 2e-3f))
                    continue; // the GPU rounded position / cell size the other way: either cell is right
                Assert.AreEqual(expected[s], bySlot[s], what + ": key of slot " + s);
                compared++;
            }
            Assert.Greater(compared, n * 9 / 10, what + ": most keys are compared");
            Assert.AreEqual(CpuBroadphase.MaxRadius(held, n), sort.ReadMaxRadius(), 1e-5f, what + ": largest radius");
        }

        static float Range(System.Random rng, float min, float max) => min + (float)rng.NextDouble() * (max - min);

        static ComputeShader RequireShader()
        {
            AvbdShaderSet set = AvbdShaderSet.Load();
            Assert.IsNotNull(set, "The AvbdShaderSet asset was not found.");
            Assert.IsNotNull(set.MortonSort, "The AvbdShaderSet has no Morton sort shader.");
            return set.MortonSort;
        }

        static ComputeShader RequireBroadphaseShader()
        {
            AvbdShaderSet set = AvbdShaderSet.Load();
            Assert.IsNotNull(set, "The AvbdShaderSet asset was not found.");
            Assert.IsNotNull(set.Broadphase, "The AvbdShaderSet has no broadphase shader.");
            return set.Broadphase;
        }
    }
}
