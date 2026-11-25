using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// The primal sweep is dispatched indirectly: stage C leaves the start and the size of every color on the GPU together with the
    /// arguments of one dispatch per color, and the host only knows how many colors there are. These tests check the arguments against the
    /// colors and that a sweep reaches every body of a color that is bigger than one thread group.
    /// </summary>
    public class IndirectSweepTests
    {
        const int ThreadsPerGroup = AvbdConst.GroupSize;

        // ---- the argument arithmetic (the C# mirror of ColorPlan) ----------------------------------------------------------

        [Test]
        public void GroupsFor_CoversEveryLaneOfEveryBodyOfAColorOnce()
        {
            var counts = new long[]
            {
                0, 1, ThreadsPerGroup - 1, ThreadsPerGroup, ThreadsPerGroup + 1, 300,
                (long)AvbdGraphColoring.RowGroups * ThreadsPerGroup - 1,
                (long)AvbdGraphColoring.RowGroups * ThreadsPerGroup,
                (long)AvbdGraphColoring.RowGroups * ThreadsPerGroup + 1,
                3L * AvbdGraphColoring.RowGroups * ThreadsPerGroup + 5,
            };

            foreach (long count in counts)
            {
                (uint x, uint y) = AvbdGraphColoring.GroupsFor(count);
                long rowWidth = (long)AvbdGraphColoring.RowGroups * ThreadsPerGroup;
                long items = count * AvbdGraphColoring.PrimalLanes; // the primal sweep has several threads per body

                Assert.LessOrEqual(x, (uint)AvbdGraphColoring.RowGroups, "count " + count + ": one dimension of a dispatch is at most 65535 groups");
                Assert.GreaterOrEqual(y, 1u, "count " + count + ": at least one row, so the dispatch is a valid (empty) one");

                // Thread (tx, ty) handles item ty * rowWidth + tx (lane item % lanes of body item / lanes). Every row but the last is full, so
                // the items the dispatch reaches are [0, (y - 1) * rowWidth + x * 64), and the last group is the only one that can overshoot.
                long reached = (y - 1) * rowWidth + (long)x * ThreadsPerGroup;
                Assert.GreaterOrEqual(reached, items, "count " + count + ": reaches every lane of every body");
                Assert.Less(reached - items, ThreadsPerGroup + (y > 1 ? rowWidth : 0), "count " + count + ": no whole group or row is wasted");
                if (y > 1)
                    Assert.AreEqual((uint)AvbdGraphColoring.RowGroups, x, "count " + count + ": the rows of a big color are full width");
            }

            Assert.AreEqual(((uint)0, (uint)1), AvbdGraphColoring.GroupsFor(0), "an empty color is a 0 x 1 dispatch");
            Assert.AreEqual(((uint)38, (uint)1), AvbdGraphColoring.GroupsFor(300), "300 bodies x 8 lanes = 2400 threads = 37.5 groups");
            Assert.AreEqual(ThreadsPerGroup % AvbdGraphColoring.PrimalLanes, 0, "the lanes of a body share a thread group");
        }

        // ---- the GPU -------------------------------------------------------------------------------------------------------

        [Test]
        public void Gpu_ColorSchedule_HoldsTheStartSizeAndArgumentsOfEveryColor()
        {
            RequireCompute();
            const int links = 200; // a chain colors with two colors of 100 bodies: 13 thread groups each (8 lanes per body)

            using (var buffers = new AvbdBodyBuffers(256))
            using (var constraints = new AvbdConstraintBuffers(buffers, 256))
            {
                var handles = new List<BodyHandle>();
                for (int i = 0; i < links; i++)
                    handles.Add(buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(i * 2f, 0f, 0f))));
                for (int i = 1; i < links; i++)
                    constraints.AddJoint(handles[i - 1], handles[i], new float3(1f, 0f, 0f), new float3(-1f, 0f, 0f));
                constraints.Rebuild();

                Assert.AreEqual(2, constraints.ColorCount);
                ColorSchedule schedule = constraints.ReadColorSchedule();

                Assert.AreEqual(100, schedule.Counts[0]);
                Assert.AreEqual(100, schedule.Counts[1]);
                Assert.AreEqual(0, schedule.Starts[0]);
                Assert.AreEqual(100, schedule.Starts[1]);
                Assert.AreEqual(links, schedule.Starts[2], "the starts end at the number of colored bodies");
                for (int c = 0; c < 2; c++)
                {
                    Assert.AreEqual(13u, schedule.GroupsX[c], "groups of color " + c + ": 100 bodies x 8 lanes = 12.5 groups");
                    Assert.AreEqual(1u, schedule.GroupsY[c]);
                    Assert.AreEqual(1u, schedule.GroupsZ[c]);
                }

                // The buckets fill [start, start + size) of every color with that color's bodies only.
                uint[] bucketed = constraints.ReadColorBodies();
                var gpuColors = new uint[256];
                buffers.Color.GetData(gpuColors);
                for (int c = 0; c < 2; c++)
                    for (int k = 0; k < schedule.Counts[c]; k++)
                        Assert.AreEqual((uint)c, gpuColors[bucketed[schedule.Starts[c] + k]], "color " + c + ", entry " + k);
            }
        }

        [Test]
        public void Gpu_ColorSchedule_MatchesTheOracleColorSizes_AndTheArgumentMirror()
        {
            RequireCompute();
            for (int seed = 1; seed <= 6; seed++)
            {
                var solver = new ReferenceSolver();
                ConstraintTestScenes.RandomLattice(solver, 500 + seed);
                SceneArrays scene = ConstraintTestScenes.FromReference(solver, 64);
                ConstraintTopology expected = scene.BuildTopology();

                using (var buffers = new AvbdBodyBuffers(64))
                using (var constraints = new AvbdConstraintBuffers(buffers, 128))
                {
                    for (int i = 0; i < scene.descriptors.Count; i++)
                        buffers.Spawn(scene.descriptors[i]);
                    buffers.ApplyEdits();
                    for (int i = 0; i < scene.constraintCount; i++)
                        constraints.AddRecord(scene.constraints[i]);
                    constraints.Rebuild();

                    string what = "random scene " + seed;
                    Assert.AreEqual(expected.ColorCount, constraints.ColorCount, what);
                    ColorSchedule schedule = constraints.ReadColorSchedule();
                    int start = 0;
                    for (int c = 0; c < expected.ColorCount; c++)
                    {
                        Assert.AreEqual(expected.ColorSize(c), schedule.Counts[c], what + ": size of color " + c);
                        Assert.AreEqual(start, schedule.Starts[c], what + ": start of color " + c);
                        (uint x, uint y) = AvbdGraphColoring.GroupsFor(schedule.Counts[c]);
                        Assert.AreEqual(x, schedule.GroupsX[c], what + ": groups x of color " + c);
                        Assert.AreEqual(y, schedule.GroupsY[c], what + ": groups y of color " + c);
                        Assert.AreEqual(1u, schedule.GroupsZ[c]);
                        start += schedule.Counts[c];
                    }
                }
            }
        }

        [Test]
        public void Gpu_Step_SolvesEveryBodyOfAColorBiggerThanOneThreadGroup()
        {
            RequireCompute();
            const int count = 300; // bodies without constraints share one color: five thread groups

            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            using (var buffers = new AvbdBodyBuffers(512))
            using (var solver = new AvbdSolver(buffers, settings))
            {
                for (int i = 0; i < count; i++)
                    buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(i * 2f, 10f, 0f)));

                solver.Step();

                var state = new BodyStateRecord[512];
                buffers.State.GetData(state);
                float drop = 10f - state[0].position.y;
                Assert.Greater(drop, 1e-4f, "gravity moved the first body");
                for (int i = 0; i < count; i++)
                    Assert.AreEqual(drop, 10f - state[i].position.y, 1e-6f, "body " + i + " moved like the first (the sweep reached it)");
            }
        }

        static void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }
    }
}
