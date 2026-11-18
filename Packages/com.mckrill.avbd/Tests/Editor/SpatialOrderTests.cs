using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;

namespace McKrill.Avbd.Tests
{
    /// <summary>The Z-curve order of body positions (<see cref="SpatialOrder"/>) and the batch spawn that uses it (<see cref="BodyEditQueue.SpawnBatch"/>). No GPU.</summary>
    public class SpatialOrderTests
    {
        static BodyDescriptor Box(float3 position) => BodyDescriptor.Box(new float3(1f), 1f, position);

        // A 4 x 4 x 4 lattice, in a scrambled order.
        static List<float3> Lattice()
        {
            var points = new List<float3>();
            for (int x = 0; x < 4; x++)
                for (int y = 0; y < 4; y++)
                    for (int z = 0; z < 4; z++)
                        points.Add(new float3(x, y, z));

            var rng = new System.Random(5);
            for (int i = points.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (points[i], points[j]) = (points[j], points[i]);
            }
            return points;
        }

        [Test]
        public void Interleave_PutsTheAxesInThreeInterleavedBitSequences()
        {
            Assert.AreEqual(0UL, SpatialOrder.Interleave(0, 0, 0));
            Assert.AreEqual(1UL, SpatialOrder.Interleave(1, 0, 0));
            Assert.AreEqual(2UL, SpatialOrder.Interleave(0, 1, 0));
            Assert.AreEqual(4UL, SpatialOrder.Interleave(0, 0, 1));
            Assert.AreEqual(9UL, SpatialOrder.Interleave(3, 0, 0));   // bits 0 and 3
            Assert.AreEqual(63UL, SpatialOrder.Interleave(3, 3, 3));  // the low six bits
            Assert.AreEqual((1UL << 63) - 1UL, SpatialOrder.Interleave(0x1FFFFFu, 0x1FFFFFu, 0x1FFFFFu));
            Assert.AreEqual(SpatialOrder.Interleave(5, 0, 0) | SpatialOrder.Interleave(0, 6, 0) | SpatialOrder.Interleave(0, 0, 7), SpatialOrder.Interleave(5, 6, 7));
        }

        [Test]
        public void Order_OfNothingOrOnePoint()
        {
            Assert.AreEqual(0, SpatialOrder.Order(new List<float3>()).Length);
            CollectionAssert.AreEqual(new[] { 0 }, SpatialOrder.Order(new List<float3> { new float3(3f, 4f, 5f) }));
        }

        [Test]
        public void Order_IsAPermutation_InAscendingKeyOrder()
        {
            List<float3> points = Lattice();
            ulong[] keys = SpatialOrder.Keys(points);
            int[] order = SpatialOrder.Order(points);

            Assert.AreEqual(points.Count, order.Length);
            var seen = new HashSet<int>(order);
            Assert.AreEqual(points.Count, seen.Count, "every point exactly once");
            for (int k = 1; k < order.Length; k++)
                Assert.LessOrEqual(keys[order[k - 1]], keys[order[k]], "rank " + k);
        }

        [Test]
        public void Order_KeepsTheInputOrder_OfPointsWithEqualKeys()
        {
            var same = new List<float3>();
            for (int i = 0; i < 6; i++)
                same.Add(new float3(1f, 2f, 3f));

            CollectionAssert.AreEqual(new[] { 0, 1, 2, 3, 4, 5 }, SpatialOrder.Order(same));
        }

        [Test]
        public void Order_IsTheSameOnEveryCall()
        {
            List<float3> points = Lattice();
            CollectionAssert.AreEqual(SpatialOrder.Order(points), SpatialOrder.Order(points));
        }

        // The point of the order: bodies that are close in space are close in the list. On a lattice of 4 x 4 x 4 points every run of 8
        // consecutive ranks is one 2 x 2 x 2 block, and every run of 2 differs along one axis only.
        [Test]
        public void Order_OfALattice_PutsEachBlockOf2x2x2PointsInARun()
        {
            List<float3> points = Lattice();
            int[] order = SpatialOrder.Order(points);

            for (int group = 0; group < order.Length / 8; group++)
            {
                int3 block = (int3)math.floor(points[order[group * 8]] * 0.5f);
                for (int k = 1; k < 8; k++)
                    Assert.IsTrue(math.all((int3)math.floor(points[order[group * 8 + k]] * 0.5f) == block), "group " + group + ", member " + k);
            }
            for (int pair = 0; pair < order.Length / 2; pair++)
            {
                float3 d = math.abs(points[order[pair * 2]] - points[order[pair * 2 + 1]]);
                Assert.AreEqual(1f, math.csum(d), "pair " + pair + " are neighbors on one axis");
            }
        }

        [Test]
        public void Keys_UseOneScaleForAllAxes()
        {
            // A wide, flat scene: the thin side still has distinct keys, and the scale is the long side's, so equal distances are equal key steps.
            var points = new List<float3> { new float3(0f, 0f, 0f), new float3(100f, 0f, 0f), new float3(0f, 1f, 0f), new float3(0f, 0f, 100f) };
            ulong[] keys = SpatialOrder.Keys(points);

            Assert.AreEqual(0UL, keys[0]);
            Assert.AreNotEqual(keys[0], keys[2], "one meter is above the resolution");
            Assert.AreEqual(SpatialOrder.Interleave(2097151u, 0, 0), keys[1]);
            Assert.AreEqual(SpatialOrder.Interleave(0, 0, 2097151u), keys[3]);
        }

        [Test]
        public void Keys_OfPointsThatAreNotFinite_AreZeroAndDoNotChangeTheOthers()
        {
            var clean = new List<float3> { new float3(0f, 0f, 0f), new float3(2f, 2f, 2f), new float3(1f, 1f, 1f) };
            var dirty = new List<float3> { new float3(float.NaN, 0f, 0f), clean[0], clean[1], new float3(float.PositiveInfinity, 1f, 1f), clean[2] };

            ulong[] a = SpatialOrder.Keys(clean);
            ulong[] b = SpatialOrder.Keys(dirty);

            Assert.AreEqual(0UL, b[0]);
            Assert.AreEqual(0UL, b[3]);
            Assert.AreEqual(a[0], b[1]);
            Assert.AreEqual(a[1], b[2]);
            Assert.AreEqual(a[2], b[4]);
            Assert.AreEqual(0UL, SpatialOrder.Keys(new List<float3> { new float3(float.NaN) })[0]);
        }

        // ---- batch spawn -------------------------------------------------------------------------------------------

        [Test]
        public void SpawnBatch_InSpatialOrder_GivesEachDescriptorItsHandle_AndMortonSlots()
        {
            List<float3> points = Lattice();
            var descriptors = new List<BodyDescriptor>();
            foreach (float3 p in points)
                descriptors.Add(Box(p));
            var queue = new BodyEditQueue(64);
            var handles = new BodyHandle[descriptors.Count];

            queue.SpawnBatch(descriptors, handles, true);
            var commands = new List<EditCommand>();
            queue.Drain(commands);

            int[] order = SpatialOrder.Order(points);
            for (int k = 0; k < order.Length; k++)
                Assert.AreEqual(k, handles[order[k]].Slot, "the point of rank " + k + " is in slot " + k);

            Assert.AreEqual(descriptors.Count, commands.Count);
            foreach (EditCommand command in commands)
            {
                int i = System.Array.FindIndex(handles, h => h.Slot == command.Slot);
                Assert.GreaterOrEqual(i, 0);
                Assert.IsTrue(math.all(command.position.xyz == points[i]), "slot " + command.Slot + " holds the body of descriptor " + i);
            }
        }

        [Test]
        public void SpawnBatch_WithoutSpatialOrder_KeepsTheListOrder()
        {
            var descriptors = new List<BodyDescriptor>();
            foreach (float3 p in Lattice())
                descriptors.Add(Box(p));
            var queue = new BodyEditQueue(64);
            var handles = new BodyHandle[descriptors.Count];

            queue.SpawnBatch(descriptors, handles, false);

            for (int i = 0; i < handles.Length; i++)
                Assert.AreEqual(i, handles[i].Slot);
        }

        [Test]
        public void SpawnBatch_WithABadDescriptor_ThrowsAndUsesNoSlot()
        {
            var descriptors = new List<BodyDescriptor> { Box(float3.zero), Box(new float3(1f, 0f, 0f)), Box(new float3(float.NaN, 0f, 0f)) };
            var queue = new BodyEditQueue(8);
            var handles = new BodyHandle[3];

            Assert.Throws<System.ArgumentException>(() => queue.SpawnBatch(descriptors, handles, true));
            Assert.AreEqual(0, queue.AliveCount);
            Assert.AreEqual(0, queue.PendingSlotCount);
        }

        [Test]
        public void SpawnBatch_NeedsRoomForTheHandles()
        {
            var queue = new BodyEditQueue(8);
            Assert.Throws<System.ArgumentException>(() => queue.SpawnBatch(new List<BodyDescriptor> { Box(float3.zero), Box(float3.zero) }, new BodyHandle[1], true));
            Assert.Throws<System.ArgumentNullException>(() => queue.SpawnBatch(null, new BodyHandle[1], true));
            Assert.Throws<System.ArgumentNullException>(() => queue.SpawnBatch(new List<BodyDescriptor>(), null, true));
            Assert.AreEqual(0, queue.AliveCount);
        }

        [Test]
        public void SpawnBatch_WhenFull_GivesTheInvalidHandleToTheRestAndCountsTheOverflow()
        {
            var descriptors = new List<BodyDescriptor>();
            foreach (float3 p in Lattice())
                descriptors.Add(Box(p));
            var queue = new BodyEditQueue(10);
            var handles = new BodyHandle[descriptors.Count];

            queue.SpawnBatch(descriptors, handles, true);

            int valid = 0;
            foreach (BodyHandle h in handles)
                if (h.IsValid) valid++;
            Assert.AreEqual(10, valid);
            Assert.AreEqual(10, queue.AliveCount);
            Assert.AreEqual(descriptors.Count - 10, queue.OverflowCount);
        }
    }
}
