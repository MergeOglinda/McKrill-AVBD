using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>The CPU construction of the constraint lists and the coloring, checked against the reference solver's coloring.</summary>
    public class ConstraintTopologyTests
    {
        const int Capacity = 64;

        // ---- against the reference solver -------------------------------------------------------------------

        [Test]
        public void Coloring_MatchesTheReferenceSolver_OnTheUpstreamScenes()
        {
            var rope = new ReferenceSolver();
            AvbdScenes.Rope(rope);
            AssertMatchesReference(rope, "rope");

            var spring = new ReferenceSolver();
            AvbdScenes.Spring(spring);
            AssertMatchesReference(spring, "spring");
        }

        [Test]
        public void Coloring_MatchesTheReferenceSolver_OnRandomScenes()
        {
            for (int seed = 1; seed <= 12; seed++)
            {
                var solver = new ReferenceSolver();
                ConstraintTestScenes.RandomLattice(solver, seed);
                AssertMatchesReference(solver, "random scene " + seed);
            }
        }

        static void AssertMatchesReference(ReferenceSolver solver, string what)
        {
            SceneArrays scene = ConstraintTestScenes.FromReference(solver, Capacity);

            solver.Params.mode = GaussSeidelMode.Colored;
            solver.Params.iterations = 0; // the coloring is built before the iterations
            solver.Step();

            ConstraintTopology topology = scene.BuildTopology();

            Assert.AreEqual(solver.Stats.colors, topology.ColorCount, what + ": color count");
            Assert.AreEqual(solver.ColorGroups.Count, topology.ColorCount, what + ": group count");
            for (int c = 0; c < topology.ColorCount; c++)
            {
                List<AvbdBody> expected = solver.ColorGroups[c];
                Assert.AreEqual(expected.Count, topology.ColorSize(c), what + ": size of color " + c);
                for (int k = 0; k < expected.Count; k++)
                    Assert.AreEqual((uint)expected[k].Index, topology.ColorBodies[topology.ColorOffsets[c] + k], what + ": color " + c + " entry " + k);
            }

            for (int s = 0; s < scene.bodyCount; s++)
                Assert.AreEqual(solver.Bodies[s].Color, topology.ColorOfBody[s], what + ": color of body " + s);

            // Each solved body's list holds its constraints, newest first.
            bool[] solved = scene.Solved();
            for (int s = 0; s < scene.bodyCount; s++)
            {
                var expected = new List<uint>();
                if (solved[s])
                    for (int i = scene.constraintCount - 1; i >= 0; i--)
                        if (scene.constraints[i].ident.x == (uint)s || scene.constraints[i].ident.y == (uint)s)
                            expected.Add((uint)i);

                uint offset = topology.Ranges[s].range.x;
                uint count = topology.Ranges[s].range.y;
                Assert.AreEqual(expected.Count, (int)count, what + ": list length of body " + s);
                for (int e = 0; e < expected.Count; e++)
                    Assert.AreEqual(expected[e], topology.List[offset + e], what + ": list of body " + s + " entry " + e);
            }
        }

        // ---- small cases --------------------------------------------------------------------------------------

        [Test]
        public void Bodies_ThatAreNotSolved_GetNoListAndNoColor()
        {
            // Body 0 static, bodies 1 and 2 dynamic. Constraint 0 joins the static body to body 1, constraint 1 joins bodies 1 and 2,
            // constraint 2 pins body 2 to the world.
            ConstraintRecord[] records =
            {
                ConstraintRecordFactory.Joint(0, 1, float3.zero, float3.zero, float.PositiveInfinity, 0f, float.PositiveInfinity, 1f),
                ConstraintRecordFactory.Spring(1, 2, float3.zero, float3.zero, 10f, 1f),
                ConstraintRecordFactory.Joint(AvbdConst.Invalid, 2, float3.zero, float3.zero, float.PositiveInfinity, 0f, float.PositiveInfinity, 1f),
            };
            var solved = new[] { false, true, true };

            var t = new ConstraintTopology();
            t.Build(records, records.Length, solved, 3);

            Assert.AreEqual(0u, t.Ranges[0].range.y, "the static body has no list");
            Assert.AreEqual(-1, t.ColorOfBody[0]);

            // Newest first: body 1 sees constraint 1 then 0. Body 2 sees constraint 2 then 1.
            Assert.AreEqual(2u, t.Ranges[1].range.y);
            Assert.AreEqual(1u, t.List[t.Ranges[1].range.x]);
            Assert.AreEqual(0u, t.List[t.Ranges[1].range.x + 1]);
            Assert.AreEqual(2u, t.Ranges[2].range.y);
            Assert.AreEqual(2u, t.List[t.Ranges[2].range.x]);
            Assert.AreEqual(1u, t.List[t.Ranges[2].range.x + 1]);

            // Highest slot first: body 2 gets color 0, its neighbor body 1 gets color 1 (the static body creates no edge).
            Assert.AreEqual(2, t.ColorCount);
            Assert.AreEqual(0, t.ColorOfBody[2]);
            Assert.AreEqual(1, t.ColorOfBody[1]);
            Assert.AreEqual(2u, t.ColorBodies[t.ColorOffsets[0]]);
            Assert.AreEqual(1u, t.ColorBodies[t.ColorOffsets[1]]);
            Assert.AreEqual(2, t.ColorBodyCount);
        }

        [Test]
        public void ConstraintsWithoutDynamicNeighbors_AllShareColorZero()
        {
            ConstraintRecord[] records =
            {
                ConstraintRecordFactory.Joint(AvbdConst.Invalid, 0, float3.zero, float3.zero, float.PositiveInfinity, 0f, float.PositiveInfinity, 1f),
                ConstraintRecordFactory.Joint(AvbdConst.Invalid, 1, float3.zero, float3.zero, float.PositiveInfinity, 0f, float.PositiveInfinity, 1f),
            };
            var t = new ConstraintTopology();
            t.Build(records, records.Length, new[] { true, true }, 2);
            Assert.AreEqual(1, t.ColorCount);
            Assert.AreEqual(2, t.ColorSize(0));
        }

        [Test]
        public void NoConstraints_GiveEmptyLists_AndSolvedBodiesColorZero()
        {
            var t = new ConstraintTopology();
            t.Build(new ConstraintRecord[0], 0, new[] { true, false, true }, 3);
            Assert.AreEqual(0, t.ListCount);
            Assert.AreEqual(1, t.ColorCount);
            Assert.AreEqual(2, t.ColorSize(0));
            Assert.AreEqual(-1, t.ColorOfBody[1]);

            t.Build(new ConstraintRecord[0], 0, new bool[0], 0);
            Assert.AreEqual(0, t.ColorCount);
            Assert.AreEqual(0, t.ColorBodyCount);
        }

        [Test]
        public void ConstraintOnAnUnusedSlot_IsRejected()
        {
            ConstraintRecord[] records = { ConstraintRecordFactory.Spring(0, 5, float3.zero, float3.zero, 10f, 1f) };
            var t = new ConstraintTopology();
            Assert.Throws<InvalidOperationException>(() => t.Build(records, 1, new[] { true, true }, 2));
        }

        // ---- the record factory ---------------------------------------------------------------------------------

        [Test]
        public void Factory_EncodesInfiniteStiffnessAndFractureAsFlags()
        {
            ConstraintRecord hard = ConstraintRecordFactory.Joint(3, 4, new float3(1, 2, 3), new float3(4, 5, 6),
                float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, 7f);
            ConstraintFlags flags = ConstraintFlagLayout.FlagsOf(hard.ident.z);
            Assert.AreEqual(ConstraintFlags.HardLinear | ConstraintFlags.HardAngular, flags);
            Assert.AreEqual(ConstraintKind.Joint, ConstraintFlagLayout.KindOf(hard.ident.z));
            Assert.AreEqual(0f, hard.anchorA.w);
            Assert.AreEqual(0f, hard.anchorB.w);
            Assert.AreEqual(0f, hard.c0Ang.w);
            Assert.AreEqual(7f, hard.c0Lin.w);
            Assert.AreEqual(3u, hard.ident.x);
            Assert.AreEqual(4u, hard.ident.y);

            ConstraintRecord soft = ConstraintRecordFactory.Joint(AvbdConst.Invalid, 4, float3.zero, float3.zero, 500f, 0f, 12f, 1f);
            Assert.AreEqual(ConstraintFlags.Fracturable, ConstraintFlagLayout.FlagsOf(soft.ident.z));
            Assert.AreEqual(500f, soft.anchorA.w);
            Assert.AreEqual(0f, soft.anchorB.w);
            Assert.AreEqual(12f, soft.c0Ang.w);
            Assert.AreEqual(AvbdConst.Invalid, soft.ident.x);

            ConstraintRecord spring = ConstraintRecordFactory.Spring(1, 2, float3.zero, float3.zero, 100f, 4f);
            Assert.AreEqual(ConstraintKind.Spring, ConstraintFlagLayout.KindOf(spring.ident.z));
            Assert.AreEqual(100f, spring.anchorA.w);
            Assert.AreEqual(4f, spring.anchorB.w);
        }
    }
}
