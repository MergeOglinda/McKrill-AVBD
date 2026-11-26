using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Stage C (graph coloring on the GPU). The CPU oracle (<see cref="CpuGraphColoring"/>) is checked against the sequential coloring of
    /// <see cref="ConstraintTopology"/>, which is checked against the reference solver elsewhere. The GPU coloring must equal the oracle
    /// exactly: Jones-Plassmann gives the sequential greedy coloring of its priority order, whatever the thread order is.
    /// </summary>
    public class GraphColoringTests
    {
        const int Capacity = 64;
        const uint NoColor = 0xFFFFFFFFu;

        // ---- the oracle -----------------------------------------------------------------------------------------

        [Test]
        public void Oracle_SlotOrder_IsTheSequentialColoring()
        {
            for (int seed = 1; seed <= 12; seed++)
            {
                var solver = new ReferenceSolver();
                ConstraintTestScenes.RandomLattice(solver, 300 + seed);
                SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);
                ConstraintTopology topology = scene.BuildTopology();

                int[] colors = CpuGraphColoring.Color(scene.constraints, topology, scene.Solved(), scene.bodyCount, GraphColoringOrder.SlotOrder);
                CollectionAssert.AreEqual(topology.ColorOfBody, colors, "random scene " + seed);
            }

            var rope = new ReferenceSolver();
            AvbdScenes.Rope(rope);
            SceneArrays ropeScene = ConstraintTestScenes.FromReference(rope, Capacity);
            ConstraintTopology ropeTopology = ropeScene.BuildTopology();
            CollectionAssert.AreEqual(ropeTopology.ColorOfBody,
                CpuGraphColoring.Color(ropeScene.constraints, ropeTopology, ropeScene.Solved(), ropeScene.bodyCount, GraphColoringOrder.SlotOrder), "rope");
        }

        [Test]
        public void Oracle_Hashed_IsAValidColoring_AndTheOrderIsStrict()
        {
            for (int seed = 1; seed <= 12; seed++)
            {
                var solver = new ReferenceSolver();
                ConstraintTestScenes.RandomLattice(solver, 400 + seed);
                SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);
                ConstraintTopology topology = scene.BuildTopology();
                bool[] solved = scene.Solved();

                int[] colors = CpuGraphColoring.Color(scene.constraints, topology, solved, scene.bodyCount, GraphColoringOrder.Hashed);
                AssertProper(scene.constraints, scene.constraintCount, solved, colors, "random scene " + seed);
            }

            // Of two different slots exactly one colors first.
            for (uint a = 0; a < 40; a++)
                for (uint b = 0; b < 40; b++)
                    if (a != b)
                        Assert.AreNotEqual(CpuGraphColoring.ColorsBefore(a, b, GraphColoringOrder.Hashed), CpuGraphColoring.ColorsBefore(b, a, GraphColoringOrder.Hashed));
        }

        [TestCase(1)]
        [TestCase(4)]
        public void Oracle_Reduce_IsAProperColoring_WithNoMoreColors(int passes)
        {
            for (int seed = 1; seed <= 12; seed++)
            {
                var solver = new ReferenceSolver();
                ConstraintTestScenes.RandomLattice(solver, 600 + seed);
                SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);
                ConstraintTopology topology = scene.BuildTopology();
                bool[] solved = scene.Solved();

                int[] before = CpuGraphColoring.Color(scene.constraints, topology, solved, scene.bodyCount, GraphColoringOrder.Hashed);
                int[] after = CpuGraphColoring.Reduce(scene.constraints, topology, solved, scene.bodyCount, before, passes);

                AssertProper(scene.constraints, scene.constraintCount, solved, after, "random scene " + seed);
                Assert.LessOrEqual(Max(after) + 1, Max(before) + 1, "random scene " + seed + ": reduction must not add colors");
            }
        }

        [TestCase(1)]
        [TestCase(3)]
        public void Gpu_Reduction_MatchesTheOracle_OnRandomScenes(int passes)
        {
            RequireCompute();
            for (int seed = 1; seed <= 8; seed++)
            {
                var solver = new ReferenceSolver();
                ConstraintTestScenes.RandomLattice(solver, 700 + seed);
                CheckScene(solver, GraphColoringOrder.Hashed, "random scene " + seed + ", " + passes + " pass(es)", passes);
            }
        }

        static int Max(int[] values)
        {
            int max = -1;
            foreach (int v in values)
                max = Math.Max(max, v);
            return max;
        }

        // ---- the GPU against the oracle ----------------------------------------------------------------------------

        [TestCase(GraphColoringOrder.SlotOrder)]
        [TestCase(GraphColoringOrder.Hashed)]
        public void Gpu_MatchesTheOracle_OnTheUpstreamScenes(GraphColoringOrder order)
        {
            RequireCompute();

            var rope = new ReferenceSolver();
            AvbdScenes.Rope(rope);
            CheckScene(rope, order, "rope");

            var spring = new ReferenceSolver();
            AvbdScenes.Spring(spring);
            CheckScene(spring, order, "spring");
        }

        [TestCase(GraphColoringOrder.SlotOrder)]
        [TestCase(GraphColoringOrder.Hashed)]
        public void Gpu_MatchesTheOracle_OnRandomScenes(GraphColoringOrder order)
        {
            RequireCompute();
            for (int seed = 1; seed <= 8; seed++)
            {
                var solver = new ReferenceSolver();
                ConstraintTestScenes.RandomLattice(solver, 500 + seed);
                CheckScene(solver, order, "random scene " + seed);
            }
        }

        static void CheckScene(ReferenceSolver reference, GraphColoringOrder order, string what, int reductionPasses = 0)
        {
            SceneArrays scene = ConstraintTestScenes.FromReference(reference, Capacity);
            ConstraintTopology topology = scene.BuildTopology();
            bool[] solved = scene.Solved();
            int[] expected = CpuGraphColoring.Color(scene.constraints, topology, solved, scene.bodyCount, order);
            if (reductionPasses > 0)
                expected = CpuGraphColoring.Reduce(scene.constraints, topology, solved, scene.bodyCount, expected, reductionPasses);

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 128))
            {
                foreach (BodyDescriptor d in scene.descriptors)
                    buffers.Spawn(d);
                for (int i = 0; i < scene.constraintCount; i++)
                    constraints.AddRecord(scene.constraints[i]);
                constraints.ColoringOrder = order;
                constraints.ColoringReductionPasses = reductionPasses;
                constraints.Rebuild();

                AssertColoring(buffers, constraints, scene.constraints, scene.constraintCount, solved, expected, scene.bodyCount, what);
            }
        }

        // A long chain: with the slot order every body waits for the one before it, with the hashed order the chains are short.
        [Test]
        public void Gpu_HashedOrder_NeedsFarFewerRoundsOnALongChain()
        {
            RequireCompute();
            const int links = 160;

            int slotRounds, hashedRounds;
            using (var buffers = new AvbdBodyBuffers(256))
            using (var constraints = new AvbdConstraintBuffers(buffers, 256))
            {
                var handles = new List<BodyHandle>();
                for (int i = 0; i < links; i++)
                    handles.Add(buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(i * 2f, 0f, 0f))));
                for (int i = 1; i < links; i++)
                    constraints.AddJoint(handles[i - 1], handles[i], new float3(1f, 0f, 0f), new float3(-1f, 0f, 0f));

                constraints.ColoringOrder = GraphColoringOrder.SlotOrder;
                constraints.Rebuild();
                slotRounds = constraints.LastColoringRounds;
                Assert.AreEqual(2, constraints.ColorCount);

                constraints.ColoringOrder = GraphColoringOrder.Hashed;
                Assert.IsTrue(constraints.NeedsRebuild, "changing the order must recolor");
                constraints.Rebuild();
                hashedRounds = constraints.LastColoringRounds;
                Assert.GreaterOrEqual(constraints.ColorCount, 2);
                Assert.LessOrEqual(constraints.ColorCount, 3, "a chain needs 2 colors; first-fit never needs more than degree + 1 = 3");

                var gpuColors = new uint[256];
                buffers.Color.GetData(gpuColors);
                for (int i = 1; i < links; i++)
                    Assert.AreNotEqual(gpuColors[handles[i - 1].Slot], gpuColors[handles[i].Slot], "neighbors " + i + " share a color");
            }

            Assert.GreaterOrEqual(slotRounds, links, "the slot order is a chain of dependencies");
            Assert.Less(hashedRounds, slotRounds / 2, "the hashed order is " + hashedRounds + " rounds against " + slotRounds);
        }

        // ---- edits between rebuilds ------------------------------------------------------------------------------------

        [Test]
        public void Gpu_Recolors_WhenBodiesAndConstraintsChange()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var constraints = new AvbdConstraintBuffers(buffers, 16))
            {
                BodyHandle a = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 0f, 0f)));
                BodyHandle b = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(2f, 0f, 0f)));
                BodyHandle fixedBody = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 0f, new float3(4f, 0f, 0f)));
                constraints.AddJoint(a, b, new float3(1f, 0f, 0f), new float3(-1f, 0f, 0f));
                constraints.AddJoint(b, fixedBody, new float3(1f, 0f, 0f), new float3(-1f, 0f, 0f));
                constraints.Rebuild();

                var gpuColors = new uint[16];
                buffers.Color.GetData(gpuColors);
                Assert.AreEqual(2, constraints.ColorCount);
                Assert.AreNotEqual(gpuColors[a.Slot], gpuColors[b.Slot]);
                Assert.AreEqual(NoColor, gpuColors[fixedBody.Slot], "a fixed body has no color");
                Assert.AreEqual(2, constraints.ColorSize(0) + constraints.ColorSize(1), "only the two solved bodies are bucketed");
                Assert.IsFalse(constraints.NeedsRebuild);

                // A new body between a and the fixed body, joined to both ends of the chain: a triangle needs a third color.
                BodyHandle c = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(1f, 2f, 0f)));
                Assert.IsTrue(buffers.HasPendingEdits);
                constraints.AddJoint(a, c, new float3(0f, 1f, 0f), new float3(-1f, 0f, 0f));
                constraints.AddJoint(b, c, new float3(0f, 1f, 0f), new float3(1f, 0f, 0f));
                constraints.Rebuild();

                buffers.Color.GetData(gpuColors);
                Assert.AreEqual(3, constraints.ColorCount);
                Assert.AreNotEqual(gpuColors[a.Slot], gpuColors[c.Slot]);
                Assert.AreNotEqual(gpuColors[b.Slot], gpuColors[c.Slot]);

                // A body without constraints is solved too (it needs the inertia solve) and takes the first color.
                BodyHandle loose = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(10f, 0f, 0f)));
                Assert.IsFalse(constraints.NeedsRebuild, "the edit is still queued");
                buffers.ApplyEdits();
                Assert.IsTrue(constraints.NeedsRebuild, "a spawned body changes the set of solved bodies");
                constraints.Rebuild();
                buffers.Color.GetData(gpuColors);
                Assert.AreEqual(0u, gpuColors[loose.Slot]);
                Assert.AreEqual(3, constraints.ColorCount);
                Assert.AreEqual(4, constraints.ColorSize(0) + constraints.ColorSize(1) + constraints.ColorSize(2));

                // A body that becomes kinematic leaves the graph.
                buffers.SetKinematic(c, true);
                buffers.ApplyEdits();
                constraints.Rebuild();
                buffers.Color.GetData(gpuColors);
                Assert.AreEqual(NoColor, gpuColors[c.Slot]);
                Assert.AreEqual(2, constraints.ColorCount, "a and b are alone again");
            }
        }

        [Test]
        public void Gpu_BodiesWithoutConstraints_GetTheFirstColor_AndNothingSolvedGivesNoColors()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4))
            {
                constraints.Rebuild();
                Assert.AreEqual(0, constraints.ColorCount, "no bodies");

                buffers.Spawn(BodyDescriptor.Box(new float3(1f), 0f, float3.zero));
                constraints.Rebuild();
                Assert.AreEqual(0, constraints.ColorCount, "a fixed body is not colored");

                for (int i = 0; i < 5; i++)
                    buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(i * 3f, 0f, 0f)));
                buffers.ApplyEdits();
                constraints.Rebuild();
                Assert.AreEqual(1, constraints.ColorCount);
                Assert.AreEqual(5, constraints.ColorSize(0));
            }
        }

        [Test]
        public void Gpu_ReportsAGraphThatNeedsTooManyColors()
        {
            RequireCompute();
            const int clique = AvbdGraphColoring.MaxColors + 1; // every body joined to every other one
            using (var buffers = new AvbdBodyBuffers(clique + 1))
            using (var constraints = new AvbdConstraintBuffers(buffers, clique * (clique - 1) / 2))
            {
                var handles = new List<BodyHandle>();
                for (int i = 0; i < clique; i++)
                    handles.Add(buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(i * 3f, 0f, 0f))));
                for (int i = 0; i < clique; i++)
                    for (int j = i + 1; j < clique; j++)
                        Assert.GreaterOrEqual(constraints.AddJoint(handles[i], handles[j], float3.zero, float3.zero), 0);

                Assert.Throws<InvalidOperationException>(() => constraints.Rebuild());
            }
        }

        // ---- persistent coloring -----------------------------------------------------------------------------------------

        [Test]
        public void Oracle_Persistent_KeepsAProperColoring_AndRepairsATamperedOne()
        {
            for (int seed = 1; seed <= 12; seed++)
            {
                var solver = new ReferenceSolver();
                ConstraintTestScenes.RandomLattice(solver, 700 + seed);
                SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);
                ConstraintTopology topology = scene.BuildTopology();
                bool[] solved = scene.Solved();
                GraphColoringOrder order = GraphColoringOrder.Hashed;
                int[] exact = CpuGraphColoring.Color(scene.constraints, topology, solved, scene.bodyCount, order);
                int cap = AvbdGraphColoring.MaxColors;

                // Nothing changed: nothing moves.
                int[] same = CpuGraphColoring.Persistent(scene.constraints, topology, solved, scene.bodyCount, order, exact, cap, out int forced);
                CollectionAssert.AreEqual(exact, same, "seed " + seed + ": a proper coloring is kept");
                Assert.AreEqual(0, forced);

                // Everything colored 0, or nothing colored: the result is proper, and from nothing it is the coloring from scratch.
                var zeros = new int[scene.bodyCount];
                var none = new int[scene.bodyCount];
                for (int s = 0; s < scene.bodyCount; s++)
                    none[s] = -1;
                int[] repaired = CpuGraphColoring.Persistent(scene.constraints, topology, solved, scene.bodyCount, order, zeros, cap, out forced);
                AssertProper(scene.constraints, scene.constraintCount, solved, repaired, "seed " + seed + " repaired");
                int[] scratch = CpuGraphColoring.Persistent(scene.constraints, topology, solved, scene.bodyCount, order, none, cap, out forced);
                CollectionAssert.AreEqual(exact, scratch, "seed " + seed + ": no colors kept is the coloring from scratch");

                // One body gets the color of a neighbor: only bodies that clash with a neighbor may change their color.
                int[] bumped = (int[])exact.Clone();
                int victim = -1;
                for (int s = 0; s < scene.bodyCount && victim < 0; s++)
                    if (solved[s] && topology.Ranges[s].range.y > 0u)
                        victim = s;
                if (victim < 0)
                    continue;
                ConstraintRecord first = scene.constraints[topology.List[topology.Ranges[victim].range.x]];
                int neighbor = (int)(first.ident.x == (uint)victim ? first.ident.y : first.ident.x);
                if (!solved[neighbor])
                    continue;
                bumped[victim] = bumped[neighbor];
                int[] fixedUp = CpuGraphColoring.Persistent(scene.constraints, topology, solved, scene.bodyCount, order, bumped, cap, out forced);
                AssertProper(scene.constraints, scene.constraintCount, solved, fixedUp, "seed " + seed + " bumped");
                // Only a body that had a neighbor of its color may have changed it.
                for (int s = 0; s < scene.bodyCount; s++)
                {
                    if (!solved[s] || fixedUp[s] == bumped[s])
                        continue;
                    bool clashed = false;
                    foreach (int o in CpuGraphColoring.Neighbors(scene.constraints, topology, solved, (uint)s))
                        clashed |= bumped[o] == bumped[s];
                    Assert.IsTrue(clashed, "seed " + seed + ": body " + s + " changed its color without a clash");
                }
            }
        }

        [Test]
        public void Oracle_Persistent_ForcesTheLastColor_WhenTheCapIsTooSmall()
        {
            var solver = new ReferenceSolver();
            ConstraintTestScenes.RandomLattice(solver, 731);
            SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);
            ConstraintTopology topology = scene.BuildTopology();
            bool[] solved = scene.Solved();
            int[] exact = CpuGraphColoring.Color(scene.constraints, topology, solved, scene.bodyCount, GraphColoringOrder.Hashed);
            int colors = 0;
            foreach (int c in exact)
                colors = Math.Max(colors, c + 1);
            Assume.That(colors, Is.GreaterThan(1));

            int[] capped = CpuGraphColoring.Persistent(scene.constraints, topology, solved, scene.bodyCount, GraphColoringOrder.Hashed, exact, colors - 1, out int forced);
            foreach (int c in capped)
                Assert.Less(c, colors - 1);
            Assert.Greater(forced, 0, "the bodies of the highest color have nowhere to go");
        }

        // From a coloring that is exact, the first persistent step colors from scratch (so it is the oracle's coloring again) and so does every one
        // that finds no clash.
        [Test]
        public void Gpu_PersistentStep_MatchesTheOracle_WhenNothingChanged()
        {
            RequireCompute();
            for (int seed = 1; seed <= 6; seed++)
            {
                var solver = new ReferenceSolver();
                ConstraintTestScenes.RandomLattice(solver, 800 + seed);
                SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);
                ConstraintTopology topology = scene.BuildTopology();
                bool[] solved = scene.Solved();
                int[] expected = CpuGraphColoring.Color(scene.constraints, topology, solved, scene.bodyCount, GraphColoringOrder.Hashed);

                using (var buffers = new AvbdBodyBuffers(Capacity))
                using (var constraints = new AvbdConstraintBuffers(buffers, 128))
                {
                    foreach (BodyDescriptor d in scene.descriptors)
                        buffers.Spawn(d);
                    for (int i = 0; i < scene.constraintCount; i++)
                        constraints.AddRecord(scene.constraints[i]);
                    constraints.Rebuild();
                    int exactColors = constraints.ColorCount;
                    constraints.ColoringFreshInterval = 1000;

                    for (int step = 0; step < 3; step++)
                    {
                        constraints.ColorStepForTests();
                        AssertPersistent(buffers, constraints, scene, solved, expected, "seed " + seed + " step " + step);
                    }
                    Assert.GreaterOrEqual(constraints.ColorCount, exactColors, "the cap is never below the colors in use");
                }
            }
        }

        [TestCase(0)]  // the rounds the tuner chose
        [TestCase(2)]  // too few: the extra rounds color what is left
        [TestCase(1)]
        public void Gpu_PersistentStep_RepairsClashes_LikeTheOracle_WhateverTheRounds(int rounds)
        {
            RequireCompute();
            for (int seed = 1; seed <= 6; seed++)
            {
                var solver = new ReferenceSolver();
                ConstraintTestScenes.RandomLattice(solver, 900 + seed);
                SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);
                ConstraintTopology topology = scene.BuildTopology();
                bool[] solved = scene.Solved();

                using (var buffers = new AvbdBodyBuffers(Capacity))
                using (var constraints = new AvbdConstraintBuffers(buffers, 128))
                {
                    foreach (BodyDescriptor d in scene.descriptors)
                        buffers.Spawn(d);
                    for (int i = 0; i < scene.constraintCount; i++)
                        constraints.AddRecord(scene.constraints[i]);
                    constraints.Rebuild();
                    constraints.ColoringFreshInterval = 1000;
                    constraints.ColorStepForTests(); // the step from scratch that follows a rebuild

                    // Give every solved body the color 0 or 1 (by slot): a great many clashes.
                    var gpuColors = new uint[buffers.Capacity];
                    buffers.Color.GetData(gpuColors);
                    var previous = new int[scene.bodyCount];
                    for (int s = 0; s < scene.bodyCount; s++)
                    {
                        previous[s] = solved[s] ? s & 1 : -1;
                        if (solved[s])
                            gpuColors[s] = (uint)previous[s];
                    }
                    buffers.Color.SetData(gpuColors);

                    int cap = constraints.ColorCount;
                    constraints.PinColoring(cap, rounds);
                    constraints.ColorStepForTests();
                    int[] expected = CpuGraphColoring.Persistent(scene.constraints, topology, solved, scene.bodyCount, GraphColoringOrder.Hashed, previous, cap, out int forced);
                    AssertPersistent(buffers, constraints, scene, solved, expected, "seed " + seed + " rounds " + rounds);

                    constraints.FlushColoringStatistics();
                    Assert.Greater(constraints.ColoringSamplesSeen, 0, "the statistics come back");
                    Assert.AreEqual((uint)forced, constraints.LastColoringSample.forced, "bodies without a free color below the cap");
                    if (rounds == 1)
                        Assert.Greater(constraints.LastColoringSample.leftover, 0u, "one round cannot color a lattice: the extra rounds have work");
                }
            }
        }

        [Test]
        public void Gpu_PersistentStep_ForcesTheLastColor_WhenTheCapIsTooSmall()
        {
            RequireCompute();
            const int clique = 5; // every body joined to every other one: needs 5 colors
            using (var buffers = new AvbdBodyBuffers(16))
            using (var constraints = new AvbdConstraintBuffers(buffers, 16))
            {
                var handles = new List<BodyHandle>();
                for (int i = 0; i < clique; i++)
                    handles.Add(buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(i * 3f, 0f, 0f))));
                for (int i = 0; i < clique; i++)
                    for (int j = i + 1; j < clique; j++)
                        constraints.AddJoint(handles[i], handles[j], float3.zero, float3.zero);
                constraints.Rebuild();
                Assert.AreEqual(clique, constraints.ColorCount);

                constraints.PinColoring(3, 0);
                constraints.ColorStepForTests();
                Assert.AreEqual(3, constraints.ColorCount, "the cap");
                var gpuColors = new uint[16];
                buffers.Color.GetData(gpuColors);
                int onLast = 0;
                foreach (BodyHandle h in handles)
                {
                    Assert.Less(gpuColors[h.Slot], 3u);
                    if (gpuColors[h.Slot] == 2u)
                        onLast++;
                }
                Assert.AreEqual(3, onLast, "the third body takes color 2, and the other two have none free below the cap: forced onto it");

                constraints.FlushColoringStatistics();
                Assert.AreEqual(2u, constraints.LastColoringSample.forced);
            }
        }

        // The first persistent step after a rebuild, and one in FreshInterval, color from scratch: the colors that drifted up come down.
        [Test]
        public void Gpu_PersistentStep_ColorsFromScratch_EveryFreshInterval()
        {
            RequireCompute();
            var solver = new ReferenceSolver();
            ConstraintTestScenes.RandomLattice(solver, 950);
            SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);
            ConstraintTopology topology = scene.BuildTopology();
            bool[] solved = scene.Solved();
            int[] expected = CpuGraphColoring.Color(scene.constraints, topology, solved, scene.bodyCount, GraphColoringOrder.Hashed);

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 128))
            {
                foreach (BodyDescriptor d in scene.descriptors)
                    buffers.Spawn(d);
                for (int i = 0; i < scene.constraintCount; i++)
                    constraints.AddRecord(scene.constraints[i]);
                constraints.Rebuild();
                constraints.ColoringFreshInterval = 2;

                var gpuColors = new uint[buffers.Capacity];
                for (int step = 0; step < 6; step++)
                {
                    // Spoil the colors after every step: the steps that keep colors repair them, the ones from scratch forget them.
                    constraints.ColorStepForTests();
                    buffers.Color.GetData(gpuColors);
                    bool scratch = step % 3 == 0; // fresh at steps 0 and 3 (the interval is two persistent steps)
                    if (scratch)
                        AssertPersistent(buffers, constraints, scene, solved, expected, "step " + step);
                    else
                        AssertProper(scene.constraints, scene.constraintCount, solved, ReadColors(gpuColors, scene.bodyCount, solved), "step " + step);
                    for (int s = 0; s < scene.bodyCount; s++)
                        if (solved[s])
                            gpuColors[s] = 7u; // drifted
                    buffers.Color.SetData(gpuColors);
                }
            }
        }

        [Test]
        public void Gpu_PersistentColoring_OffColorsFromScratchEveryStep()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var constraints = new AvbdConstraintBuffers(buffers, 16))
            {
                BodyHandle a = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, float3.zero));
                BodyHandle b = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(2f, 0f, 0f)));
                constraints.AddJoint(a, b, new float3(1f, 0f, 0f), new float3(-1f, 0f, 0f));
                constraints.Rebuild();
                Assert.IsTrue(constraints.PersistentColoring, "the default");
                constraints.PersistentColoring = false;
                constraints.ColorStepForTests();
                Assert.AreEqual(2, constraints.ColorCount, "exact: no cap above the colors");
                Assert.AreEqual(0, constraints.ColoringSamplesSeen);
            }
        }

        // The bodies are in the color buckets of the colors they have, and the colors are those of the oracle.
        static void AssertPersistent(AvbdBodyBuffers buffers, AvbdConstraintBuffers constraints, SceneArrays scene, bool[] solved, int[] expected, string what)
        {
            var gpuColors = new uint[buffers.Capacity];
            buffers.Color.GetData(gpuColors);
            int colors = 0;
            for (int s = 0; s < scene.bodyCount; s++)
            {
                if (solved[s])
                {
                    Assert.AreEqual((uint)expected[s], gpuColors[s], what + ": color of body " + s);
                    colors = Math.Max(colors, expected[s] + 1);
                }
                else
                    Assert.AreEqual(NoColor, gpuColors[s], what + ": body " + s + " is not solved and has no color");
            }
            AssertProper(scene.constraints, scene.constraintCount, solved, expected, what + " (oracle)");
            Assert.GreaterOrEqual(constraints.ColorCount, colors, what + ": the sweep dispatches every color in use");

            uint[] bucketed = constraints.ReadColorBodies();
            int solvedCount = 0;
            foreach (bool s in solved)
                if (s) solvedCount++;
            Assert.AreEqual(solvedCount, bucketed.Length, what + ": bucketed bodies");
            for (int c = 0; c < colors; c++)
            {
                var want = new List<uint>();
                for (int s = 0; s < scene.bodyCount; s++)
                    if (expected[s] == c)
                        want.Add((uint)s);
                var got = new List<uint>();
                for (int k = 0; k < constraints.ColorSize(c); k++)
                    got.Add(bucketed[constraints.ColorOffset(c) + k]);
                want.Sort();
                got.Sort();
                CollectionAssert.AreEqual(want, got, what + ": bodies of color " + c);
            }
        }

        static int[] ReadColors(uint[] gpuColors, int bodyCount, bool[] solved)
        {
            var colors = new int[bodyCount];
            for (int s = 0; s < bodyCount; s++)
                colors[s] = solved[s] ? (int)gpuColors[s] : -1;
            return colors;
        }

        // ---- persistent coloring: statistics ---------------------------------------------------------------------------

        [Test]
        public void Gpu_Persistent_ReportsWhatTheStepDid()
        {
            RequireCompute();
            const int links = 120;
            using (var buffers = new AvbdBodyBuffers(256))
            using (var constraints = new AvbdConstraintBuffers(buffers, 256))
            {
                var handles = new List<BodyHandle>();
                for (int i = 0; i < links; i++)
                    handles.Add(buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(i * 2f, 0f, 0f))));
                for (int i = 1; i < links; i++)
                    constraints.AddJoint(handles[i - 1], handles[i], new float3(1f, 0f, 0f), new float3(-1f, 0f, 0f));
                constraints.Rebuild();
                constraints.ColorStepForTests(); // from scratch
                constraints.FlushColoringStatistics();
                Assert.AreEqual(1, constraints.ColoringSamplesSeen);
                Assert.IsTrue(constraints.LastColoringSample.fresh);
                Assert.AreEqual(0u, constraints.LastColoringSample.forced);

                // Every body takes color 0: every second neighbor pair clashes, so about half of the chain is released.
                var gpuColors = new uint[256];
                buffers.Color.GetData(gpuColors);
                for (int i = 0; i < links; i++)
                    gpuColors[handles[i].Slot] = 0u;
                buffers.Color.SetData(gpuColors);
                constraints.PinColoring(0, 1);
                constraints.ColorStepForTests();
                constraints.FlushColoringStatistics();

                ColoringSample sample = constraints.LastColoringSample;
                Assert.IsFalse(sample.fresh);
                Assert.Greater(sample.released, 0u);
                Assert.AreEqual(1, sample.rounds);
                Assert.Greater(sample.leftover, 0u, "one round cannot color a chain that was released (and the extra rounds did the rest)");

                buffers.Color.GetData(gpuColors);
                for (int i = 1; i < links; i++)
                    Assert.AreNotEqual(gpuColors[handles[i - 1].Slot], gpuColors[handles[i].Slot], "neighbors " + i + " share a color");
            }
        }

        // ---- the tuner ------------------------------------------------------------------------------------------------------

        static ColoringSample Sample(bool fresh, int rounds, int cap, uint leftover, uint forced, uint colors, uint lastRound)
        {
            return new ColoringSample { fresh = fresh, rounds = rounds, cap = cap, leftover = leftover, forced = forced, colors = colors, lastRound = lastRound };
        }

        [Test]
        public void Tuner_RaisesRoundsAtOnce_WhenBodiesWereLeftOver()
        {
            var tuner = new ColoringTuner();
            int before = tuner.RoundsFor(false);
            tuner.Observe(Sample(false, before, 8, 5u, 0u, 4u, (uint)before));
            Assert.AreEqual(before * 2, tuner.RoundsFor(false));
            Assert.AreEqual(ColoringTuner.InitialFreshRounds, tuner.RoundsFor(true), "the other kind of step is not touched");

            for (int i = 0; i < 20; i++)
                tuner.Observe(Sample(false, tuner.RoundsFor(false), 8, 5u, 0u, 4u, 1u));
            Assert.AreEqual(ColoringTuner.MaxRounds, tuner.RoundsFor(false));
        }

        [Test]
        public void Tuner_LowersRoundsSlowly_ToWhatTheLastStepsNeeded()
        {
            var tuner = new ColoringTuner();
            for (int i = 0; i < ColoringTuner.Window - 1; i++)
            {
                tuner.Observe(Sample(false, 8, 8, 0u, 0u, 4u, 3u));
                Assert.AreEqual(ColoringTuner.InitialRounds, tuner.RoundsFor(false), "not before a whole window has been seen");
            }
            tuner.Observe(Sample(false, 8, 8, 0u, 0u, 4u, 3u));
            Assert.AreEqual(5, tuner.RoundsFor(false), "the most rounds a step needed, plus two");

            // One step that needs more rounds raises it again as soon as the window holds it.
            for (int i = 0; i < ColoringTuner.Window; i++)
                tuner.Observe(Sample(false, 5, 8, 0u, 0u, 4u, i == 3 ? 7u : 2u));
            Assert.AreEqual(9, tuner.RoundsFor(false));
        }

        [Test]
        public void Tuner_CapFollowsTheColorsUpAtOnce_AndDownOnlyWithAColoringFromScratch()
        {
            var tuner = new ColoringTuner();
            tuner.Reset(5);
            Assert.AreEqual(5 + ColoringTuner.Headroom, tuner.Cap);

            tuner.Observe(Sample(false, 8, tuner.Cap, 0u, 0u, 12u, 2u));
            Assert.AreEqual(12 + ColoringTuner.Headroom, tuner.Cap);

            tuner.Observe(Sample(false, 8, tuner.Cap, 0u, 0u, 6u, 2u));
            Assert.AreEqual(12 + ColoringTuner.Headroom, tuner.Cap, "colors in use drift upward, so a step that keeps colors does not lower the cap");

            tuner.Observe(Sample(true, 16, tuner.Cap, 0u, 0u, 6u, 2u));
            Assert.AreEqual(6 + ColoringTuner.Headroom, tuner.Cap);
        }

        [Test]
        public void Tuner_ForcedBodies_RaiseTheCapBeyondTheColorsSeen()
        {
            var tuner = new ColoringTuner();
            tuner.Reset(3);
            int cap = tuner.Cap;
            tuner.Observe(Sample(false, 8, cap, 0u, 2u, (uint)cap, 2u));
            Assert.GreaterOrEqual(tuner.Cap, cap + 4);

            tuner.Observe(Sample(true, 16, tuner.Cap, 0u, 1u, 3u, 2u));
            Assert.GreaterOrEqual(tuner.Cap, 3 + ColoringTuner.Headroom, "a coloring from scratch that forced bodies does not lower the cap");
            Assert.LessOrEqual(tuner.Cap, AvbdGraphColoring.MaxColors);
        }

        // ---- helpers -----------------------------------------------------------------------------------------------

        // Compares what the GPU left in the body arrays and the bucket buffer with the oracle's coloring.
        static void AssertColoring(AvbdBodyBuffers buffers, AvbdConstraintBuffers constraints, ConstraintRecord[] records, int recordCount,
            bool[] solved, int[] expected, int bodyCount, string what)
        {
            var gpuColors = new uint[buffers.Capacity];
            buffers.Color.GetData(gpuColors);

            int colors = 0;
            for (int s = 0; s < bodyCount; s++)
            {
                if (solved[s])
                {
                    Assert.AreEqual((uint)expected[s], gpuColors[s], what + ": color of body " + s);
                    colors = Math.Max(colors, expected[s] + 1);
                }
                else
                {
                    Assert.AreEqual(NoColor, gpuColors[s], what + ": body " + s + " is not solved and has no color");
                }
            }
            AssertProper(records, recordCount, solved, expected, what + " (oracle)");

            Assert.AreEqual(colors, constraints.ColorCount, what + ": number of colors");

            // The buckets: every solved body once, in its color's range.
            uint[] bucketed = constraints.ReadColorBodies();
            int solvedCount = 0;
            foreach (bool s in solved)
                if (s) solvedCount++;
            Assert.AreEqual(solvedCount, bucketed.Length, what + ": bucketed bodies");

            for (int c = 0; c < colors; c++)
            {
                var want = new List<uint>();
                for (int s = 0; s < bodyCount; s++)
                    if (expected[s] == c)
                        want.Add((uint)s);
                var got = new List<uint>();
                for (int k = 0; k < constraints.ColorSize(c); k++)
                    got.Add(bucketed[constraints.ColorOffset(c) + k]);
                want.Sort();
                got.Sort();
                CollectionAssert.AreEqual(want, got, what + ": bodies of color " + c);
            }
        }

        // No two solved bodies joined by a constraint share a color, and every solved body has one.
        static void AssertProper(ConstraintRecord[] records, int recordCount, bool[] solved, int[] colors, string what)
        {
            for (int s = 0; s < solved.Length; s++)
                Assert.AreEqual(solved[s], colors[s] >= 0, what + ": body " + s + " solved/colored");

            for (int i = 0; i < recordCount; i++)
            {
                uint a = records[i].ident.x;
                uint b = records[i].ident.y;
                if (a == AvbdConst.Invalid || !solved[a] || !solved[b])
                    continue;
                Assert.AreNotEqual(colors[a], colors[b], what + ": constraint " + i + " joins bodies " + a + " and " + b + " of one color");
            }
        }

        static void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }
    }
}
