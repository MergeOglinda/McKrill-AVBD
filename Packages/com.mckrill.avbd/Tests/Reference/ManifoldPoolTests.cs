using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// The manifold pool of stage N: slots are handed to the pairs of the broadphase, kept while a pair has contacts, and given back. What
    /// the manifolds contain is checked against the CPU oracle in <see cref="ContactTests"/>; here it is where they live.
    /// </summary>
    public class ManifoldPoolTests
    {
        const int Capacity = 64;

        // What a step does up to the end of the narrow phase, without solving: the bodies stay where they are.
        static void NarrowphaseRound(AvbdSolver solver, AvbdConstraintBuffers constraints)
        {
            solver.DispatchBroadphase();
            solver.DispatchPredict();
            constraints.PrepareContacts();
            solver.DispatchNarrowphase();
        }

        static AvbdSolverSettings NoGravity()
        {
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            settings.gravity = float3.zero;
            return settings;
        }

        [Test]
        public void Gpu_Pool_GivesASlotBackWhenThePairSeparates()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 1, null, 16))
            using (var solver = new AvbdSolver(buffers, NoGravity(), null, constraints))
            {
                BodyHandle a = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 0f, 0f)));
                BodyHandle b = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0.8f, 0f, 0f)));
                buffers.ApplyEdits();
                constraints.Rebuild();

                ContactSnapshot start = constraints.ReadContacts();
                Assert.AreEqual(16, start.FreeSlots, "all slots start free");
                Assert.AreEqual(0, start.UsedSlots);

                NarrowphaseRound(solver, constraints);
                ContactSnapshot touching = constraints.ReadContacts();
                Assert.AreEqual(1, touching.UsedSlots);
                Assert.AreEqual(15, touching.FreeSlots);
                Assert.Greater(touching.CountOf(a.Slot, b.Slot), 0u);

                buffers.SetPose(b, new float3(20f, 0f, 0f), quaternion.identity);
                NarrowphaseRound(solver, constraints);
                ContactSnapshot apart = constraints.ReadContacts();
                Assert.AreEqual(0, apart.UsedSlots, "the manifold of a pair that no longer touches is removed");
                Assert.AreEqual(16, apart.FreeSlots, "and its slot is free again");
                Assert.AreEqual(0u, apart.CountOf(a.Slot, b.Slot));

                // And the same slot count again when they touch again: nothing leaked.
                buffers.SetPose(b, new float3(0.8f, 0f, 0f), quaternion.identity);
                NarrowphaseRound(solver, constraints);
                ContactSnapshot again = constraints.ReadContacts();
                Assert.AreEqual(1, again.UsedSlots);
                Assert.AreEqual(15, again.FreeSlots);
            }
        }

        [Test]
        public void Gpu_Pool_DropsPairsWhenFull_StoresOnlyRealPairs_AndRecovers()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 1, null, 2))
            using (var solver = new AvbdSolver(buffers, NoGravity(), null, constraints))
            {
                // A row of four boxes: the pairs (1, 0), (2, 1) and (3, 2) touch, and there are two slots.
                var boxes = new List<BodyHandle>();
                for (int i = 0; i < 4; i++)
                    boxes.Add(buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0.8f * i, 0f, 0f))));
                buffers.ApplyEdits();
                constraints.Rebuild();

                NarrowphaseRound(solver, constraints);
                ContactSnapshot full = constraints.ReadContacts();
                Assert.AreEqual(2, full.UsedSlots, "the pool is full");
                Assert.AreEqual(0, full.FreeSlots);
                Assert.GreaterOrEqual(full.Overflow, 1, "the third pair found no slot");
                Assert.AreEqual(0, full.DuplicatePairs);
                for (int slot = 0; slot < 2; slot++)
                {
                    int hi = (int)full.Bodies[slot].x;
                    int lo = (int)full.Bodies[slot].y;
                    Assert.AreEqual(1, hi - lo, "slot " + slot + " holds (" + hi + ", " + lo + "), which does not touch");
                    Assert.Greater(full.Counts[slot], 0u);
                }

                // The last box leaves: only (1, 0) and (2, 1) are left, and two slots are enough. The pair that was dropped has to be
                // picked up once the slot of (3, 2), if it had one, is free: that takes a step.
                buffers.SetPose(boxes[3], new float3(30f, 0f, 0f), quaternion.identity);
                for (int round = 0; round < 3; round++)
                    NarrowphaseRound(solver, constraints);
                ContactSnapshot recovered = constraints.ReadContacts();
                Assert.AreEqual(2, recovered.UsedSlots);
                Assert.Greater(recovered.CountOf(1, 0), 0u, "(1, 0) has its manifold");
                Assert.Greater(recovered.CountOf(2, 1), 0u, "(2, 1) has its manifold");
                Assert.AreEqual(0, recovered.DuplicatePairs);
            }
        }

        [Test]
        public void Gpu_Pool_HandlesMoreBodiesThanTheDenseLayoutAllowed()
        {
            RequireCompute();
            const int nx = 10, ny = 10, nz = 7; // 700 boxes; the dense layout stopped at 512 bodies
            const float spacing = 0.95f;        // 1 wide boxes overlap their neighbors by 0.05

            using (var buffers = new AvbdBodyBuffers(1024))
            using (var constraints = new AvbdConstraintBuffers(buffers, 1, null, 8000))
            using (var solver = new AvbdSolver(buffers, NoGravity(), null, constraints))
            {
                var index = new int[nx, ny, nz];
                int n = 0;
                for (int z = 0; z < nz; z++)
                    for (int y = 0; y < ny; y++)
                        for (int x = 0; x < nx; x++)
                        {
                            buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(x, y, z) * spacing));
                            index[x, y, z] = n++;
                        }
                buffers.ApplyEdits();
                constraints.Rebuild();

                NarrowphaseRound(solver, constraints);
                ContactSnapshot snapshot = constraints.ReadContacts();
                Assert.AreEqual(0, snapshot.Overflow, "no pair may be dropped");
                Assert.AreEqual(0, snapshot.DuplicatePairs);
                Assert.AreEqual(snapshot.SlotsWithContacts, snapshot.UsedSlots, "no used slot without contacts");
                Assert.AreEqual(8000, snapshot.UsedSlots + snapshot.FreeSlots);

                // Every pair of boxes next to each other along an axis overlaps by 0.05 and must have a manifold.
                int faceNeighbors = 0;
                for (int z = 0; z < nz; z++)
                    for (int y = 0; y < ny; y++)
                        for (int x = 0; x < nx; x++)
                        {
                            if (x + 1 < nx) { faceNeighbors++; Assert.Greater(snapshot.CountOf(index[x, y, z], index[x + 1, y, z]), 0u, "x neighbor of " + x + "," + y + "," + z); }
                            if (y + 1 < ny) { faceNeighbors++; Assert.Greater(snapshot.CountOf(index[x, y, z], index[x, y + 1, z]), 0u, "y neighbor of " + x + "," + y + "," + z); }
                            if (z + 1 < nz) { faceNeighbors++; Assert.Greater(snapshot.CountOf(index[x, y, z], index[x, y, z + 1]), 0u, "z neighbor of " + x + "," + y + "," + z); }
                        }
                Assert.AreEqual(1860, faceNeighbors);

                // Nothing else than neighbors (face, edge or corner) may touch: the boxes two cells apart are 0.9 clear of each other.
                for (int slot = 0; slot < snapshot.Counts.Length; slot++)
                {
                    if (snapshot.Counts[slot] == 0u)
                        continue;
                    int a = (int)snapshot.Bodies[slot].x;
                    int b = (int)snapshot.Bodies[slot].y;
                    int ax = a % nx, ay = (a / nx) % ny, az = a / (nx * ny);
                    int bx = b % nx, by = (b / nx) % ny, bz = b / (nx * ny);
                    Assert.LessOrEqual(math.max(math.abs(ax - bx), math.max(math.abs(ay - by), math.abs(az - bz))), 1, "boxes " + a + " and " + b + " are not neighbors");
                }
            }
        }

        [Test]
        public void Gpu_Pool_StaysConsistentWhileAPyramidSettles()
        {
            RequireCompute();
            var reference = new ReferenceSolver();
            AvbdScenes.Pyramid(reference, 6);
            SceneArrays scene = ConstraintTestScenes.FromReference(reference, Capacity);

            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            settings.gravity = new float3(0f, 0f, -10f); // the upstream scenes are z-up

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 1, null, 512))
            using (var solver = new AvbdSolver(buffers, settings, null, constraints))
            {
                foreach (BodyDescriptor d in scene.descriptors)
                    buffers.Spawn(d);

                int maxUsed = 0;
                int previousHighWater = 0;
                for (int step = 0; step < 40; step++)
                {
                    solver.Step();
                    ContactSnapshot snapshot = constraints.ReadContacts();
                    Assert.AreEqual(0, snapshot.Overflow, "step " + step);
                    Assert.AreEqual(0, snapshot.DuplicatePairs, "step " + step);
                    Assert.AreEqual(512, snapshot.UsedSlots + snapshot.FreeSlots, "step " + step + ": every slot is used or free");
                    Assert.AreEqual(snapshot.SlotsWithContacts, snapshot.UsedSlots, "step " + step + ": a slot without contacts must be free");
                    for (int slot = 0; slot < snapshot.Counts.Length; slot++)
                        if (snapshot.Counts[slot] != 0u)
                            Assert.Greater(snapshot.Bodies[slot].x, snapshot.Bodies[slot].y, "step " + step + ": body A is the higher slot");
                    maxUsed = math.max(maxUsed, snapshot.UsedSlots);
                    Assert.AreEqual(0, constraints.ReadSolveFailures(), "step " + step);

                    // The high-water mark: above every used slot, never above the capacity, and it never falls.
                    int highestUsed = -1;
                    for (int slot = 0; slot < snapshot.Bodies.Length; slot++)
                        if (snapshot.Bodies[slot].x != AvbdConst.Invalid)
                            highestUsed = slot;
                    Assert.Greater(snapshot.HighWater, highestUsed, "step " + step + ": every used slot is below the high-water mark");
                    Assert.LessOrEqual(snapshot.HighWater, 512, "step " + step);
                    Assert.GreaterOrEqual(snapshot.HighWater, previousHighWater, "step " + step + ": the high-water mark never falls");
                    previousHighWater = snapshot.HighWater;
                }
                Assert.Greater(maxUsed, 10, "the pyramid should touch the ground and itself");
            }
        }

        // ---- the indirect dispatches of stage N and of the dual update of the contacts ---------------------------------------

        [Test]
        public void Gpu_HighWaterMark_StartsAtZero_AndTheArgumentsStartEmpty()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 1, null, 128))
            {
                Assert.AreEqual(0, constraints.ReadContacts().HighWater);
                uint[] args = constraints.ReadNpArgs();
                for (int entry = 0; entry < ContactLayout.PoolArgsCount; entry++)
                {
                    Assert.AreEqual(0u, args[entry * 3], "entry " + entry + ": no groups");
                    Assert.AreEqual(1u, args[entry * 3 + 1]);
                    Assert.AreEqual(1u, args[entry * 3 + 2]);
                }
            }
        }

        [Test]
        public void Gpu_ContactDispatches_FollowTheManifoldsAndPairs_NotTheCapacities()
        {
            RequireCompute();
            const int slots = 20000; // far more than a pyramid of 6 layers needs
            var reference = new ReferenceSolver();
            AvbdScenes.Pyramid(reference, 6);
            SceneArrays scene = ConstraintTestScenes.FromReference(reference, Capacity);

            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            settings.gravity = new float3(0f, 0f, -10f); // the upstream scenes are z-up

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 1, null, slots))
            using (var solver = new AvbdSolver(buffers, settings, null, constraints))
            {
                foreach (BodyDescriptor d in scene.descriptors)
                    buffers.Spawn(d);

                for (int step = 0; step < 5; step++)
                    solver.Step();

                ContactSnapshot snapshot = constraints.ReadContacts();
                Assert.Greater(snapshot.HighWater, 10, "the pyramid should touch the ground and itself");
                Assert.Less(snapshot.HighWater, slots / 10, "the manifolds are packed at the bottom of the pool");

                AvbdBroadphase broadphase = solver.Broadphase;
                broadphase.ReadCounts();
                long dynamicPairs = math.min(broadphase.DynamicPairCount, broadphase.DynamicPairCapacity);
                long staticPairs = math.min(broadphase.StaticPairCount, broadphase.StaticPairCapacity);

                // The planner ran after the last allocation of the step, so every entry has the final counts.
                uint[] args = constraints.ReadNpArgs();
                AssertArgs(args, ContactLayout.PoolArgsRetire, snapshot.HighWater);
                AssertArgs(args, ContactLayout.PoolArgsAllocateDynamic, dynamicPairs);
                AssertArgs(args, ContactLayout.PoolArgsAllocateStatic, staticPairs);
                AssertArgs(args, ContactLayout.PoolArgsNarrowphase, snapshot.HighWater);
                AssertArgs(args, ContactLayout.PoolArgsDual, (long)snapshot.HighWater * ContactLayout.MaxContactsPerPair);

                (uint capacityGroups, _) = AvbdIndirect.GroupsFor((long)slots * ContactLayout.MaxContactsPerPair);
                Assert.Less(args[ContactLayout.PoolArgsDual * 3], capacityGroups / 10,
                    "the dual update runs over the manifolds that exist, not over the " + capacityGroups + " groups of the capacity");
            }
        }

        static void AssertArgs(uint[] args, int entry, long items)
        {
            (uint x, uint y) = AvbdIndirect.GroupsFor(items);
            Assert.AreEqual(x, args[entry * 3], "groups x of entry " + entry + " for " + items + " items");
            Assert.AreEqual(y, args[entry * 3 + 1], "groups y of entry " + entry);
            Assert.AreEqual(1u, args[entry * 3 + 2], "groups z of entry " + entry);
        }

        static void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }
    }
}
