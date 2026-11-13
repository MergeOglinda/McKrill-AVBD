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

        static void CheckScene(ReferenceSolver reference, GraphColoringOrder order, string what)
        {
            SceneArrays scene = ConstraintTestScenes.FromReference(reference, Capacity);
            ConstraintTopology topology = scene.BuildTopology();
            bool[] solved = scene.Solved();
            int[] expected = CpuGraphColoring.Color(scene.constraints, topology, solved, scene.bodyCount, order);

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 128))
            {
                foreach (BodyDescriptor d in scene.descriptors)
                    buffers.Spawn(d);
                for (int i = 0; i < scene.constraintCount; i++)
                    constraints.AddRecord(scene.constraints[i]);
                constraints.ColoringOrder = order;
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
                Assert.IsTrue(constraints.HasPendingChanges, "changing the order must recolor");
                constraints.Rebuild();
                hashedRounds = constraints.LastColoringRounds;
                Assert.GreaterOrEqual(constraints.ColorCount, 2);
                Assert.LessOrEqual(constraints.ColorCount, 3, "a chain needs 2 colors; first-fit never needs more than degree + 1 = 3");

                var meta = new BodyMetaRecord[256];
                buffers.Meta.GetData(meta);
                for (int i = 1; i < links; i++)
                    Assert.AreNotEqual(meta[handles[i - 1].Slot].state.x, meta[handles[i].Slot].state.x, "neighbors " + i + " share a color");
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

                var meta = new BodyMetaRecord[16];
                buffers.Meta.GetData(meta);
                Assert.AreEqual(2, constraints.ColorCount);
                Assert.AreNotEqual(meta[a.Slot].state.x, meta[b.Slot].state.x);
                Assert.AreEqual(NoColor, meta[fixedBody.Slot].state.x, "a fixed body has no color");
                Assert.AreEqual(2, constraints.ColorSize(0) + constraints.ColorSize(1), "only the two solved bodies are bucketed");
                Assert.IsFalse(constraints.HasPendingChanges);

                // A new body between a and the fixed body, joined to both ends of the chain: a triangle needs a third color.
                BodyHandle c = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(1f, 2f, 0f)));
                Assert.IsTrue(buffers.HasPendingEdits);
                constraints.AddJoint(a, c, new float3(0f, 1f, 0f), new float3(-1f, 0f, 0f));
                constraints.AddJoint(b, c, new float3(0f, 1f, 0f), new float3(1f, 0f, 0f));
                constraints.Rebuild();

                buffers.Meta.GetData(meta);
                Assert.AreEqual(3, constraints.ColorCount);
                Assert.AreNotEqual(meta[a.Slot].state.x, meta[c.Slot].state.x);
                Assert.AreNotEqual(meta[b.Slot].state.x, meta[c.Slot].state.x);

                // A body without constraints is solved too (it needs the inertia solve) and takes the first color.
                BodyHandle loose = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(10f, 0f, 0f)));
                Assert.IsFalse(constraints.HasPendingChanges, "the edit is still queued");
                buffers.ApplyEdits();
                Assert.IsTrue(constraints.HasPendingChanges, "a spawned body changes the set of solved bodies");
                constraints.Rebuild();
                buffers.Meta.GetData(meta);
                Assert.AreEqual(0u, meta[loose.Slot].state.x);
                Assert.AreEqual(3, constraints.ColorCount);
                Assert.AreEqual(4, constraints.ColorSize(0) + constraints.ColorSize(1) + constraints.ColorSize(2));

                // A body that becomes kinematic leaves the graph.
                buffers.SetKinematic(c, true);
                buffers.ApplyEdits();
                constraints.Rebuild();
                buffers.Meta.GetData(meta);
                Assert.AreEqual(NoColor, meta[c.Slot].state.x);
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

        // ---- helpers -----------------------------------------------------------------------------------------------

        // Compares what the GPU left in the body arrays and the bucket buffer with the oracle's coloring.
        static void AssertColoring(AvbdBodyBuffers buffers, AvbdConstraintBuffers constraints, ConstraintRecord[] records, int recordCount,
            bool[] solved, int[] expected, int bodyCount, string what)
        {
            var meta = new BodyMetaRecord[buffers.Capacity];
            buffers.Meta.GetData(meta);

            int colors = 0;
            for (int s = 0; s < bodyCount; s++)
            {
                if (solved[s])
                {
                    Assert.AreEqual((uint)expected[s], meta[s].state.x, what + ": color of body " + s);
                    colors = Math.Max(colors, expected[s] + 1);
                }
                else
                {
                    Assert.AreEqual(NoColor, meta[s].state.x, what + ": body " + s + " is not solved and has no color");
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
