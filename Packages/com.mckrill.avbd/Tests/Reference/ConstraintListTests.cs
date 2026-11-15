using System;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Stage G: the per-body constraint lists built on the GPU (<c>AvbdLists.compute</c>) against the CPU oracle, <see cref="ConstraintTopology"/>.
    /// The lists and the ranges must be identical, entry by entry: the GPU sorts every list into the oracle's order, so the result does not
    /// depend on thread timing or on the slots the manifold pool handed out.
    /// </summary>
    public class ConstraintListTests
    {
        const int Capacity = 64;
        const int ContactSlots = 256;

        [Test]
        public void Gpu_Lists_MatchTheOracle_ForJointsSpringsPinsAndFixedBodies()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 32))
            {
                BodyHandle ground = buffers.Spawn(BodyDescriptor.Box(new float3(10f, 10f, 1f), 0f, new float3(0f, 0f, -1f))); // mass 0
                BodyDescriptor mover = BodyDescriptor.Box(new float3(1f), 1f, new float3(3f, 0f, 0f));
                mover.kinematic = true;
                BodyHandle kinematic = buffers.Spawn(mover);
                var chain = new BodyHandle[6];
                for (int i = 0; i < chain.Length; i++)
                    chain[i] = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(i * 1.5f, 0f, 2f)));

                constraints.AddJoint(BodyHandle.Invalid, chain[0], new float3(0f, 0f, 2f), float3.zero); // pinned to the world
                for (int i = 0; i + 1 < chain.Length; i++)
                    constraints.AddJoint(chain[i], chain[i + 1], new float3(0.75f, 0f, 0f), new float3(-0.75f, 0f, 0f));
                constraints.AddSpring(chain[2], chain[4], float3.zero, float3.zero, 10f);
                constraints.AddJoint(chain[3], ground, float3.zero, float3.zero);       // to a body with no mass: listed for the chain body only
                constraints.AddSpring(chain[5], kinematic, float3.zero, float3.zero, 5f); // to a kinematic body
                constraints.AddSpring(chain[0], chain[1], float3.zero, float3.zero, 3f);  // a second constraint between the same two bodies
                constraints.Rebuild();

                AssertListsMatchOracle(buffers, constraints, "joints and springs");
                Assert.AreEqual(0, constraints.ReadListControl().Invalid);
                Assert.AreEqual(0, constraints.ReadListControl().Broken);
            }
        }

        [Test]
        public void Gpu_Lists_LeaveOutAFracturedJoint_AndCountIt()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 8))
            {
                BodyHandle a = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 0f, 0f)));
                BodyHandle b = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(1.5f, 0f, 0f)));
                BodyHandle c = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(3f, 0f, 0f)));
                constraints.Rebuild();

                ConstraintRecord intact = ConstraintRecordFactory.Joint(
                    (uint)a.Slot, (uint)b.Slot, float3.zero, float3.zero, float.PositiveInfinity, 0f, float.PositiveInfinity, 1f);
                ConstraintRecord broken = ConstraintRecordFactory.Joint(
                    (uint)b.Slot, (uint)c.Slot, float3.zero, float3.zero, float.PositiveInfinity, 0f, float.PositiveInfinity, 1f);
                broken.ident.z |= (uint)ConstraintFlags.Broken;
                constraints.AddRecord(intact);
                constraints.AddRecord(broken);
                constraints.AddRecord(intact);
                constraints.Rebuild();

                AssertListsMatchOracle(buffers, constraints, "a fractured joint");
                ConstraintListControl control = constraints.ReadListControl();
                Assert.AreEqual(1, control.Broken, "one joint is left out");
                Assert.AreEqual(1, constraints.BrokenCount);
                Assert.AreEqual(4, control.Total, "two intact joints, two bodies each");

                // c has only the fractured joint: its list is empty.
                var ranges = new BodyRangeRecord[Capacity];
                buffers.Range.GetData(ranges);
                Assert.AreEqual(0u, ranges[c.Slot].range.y);
            }
        }

        [Test]
        public void Gpu_Lists_WithContacts_MatchTheOracle_EveryStep()
        {
            RequireCompute();
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            settings.gravity = new float3(0f, 0f, -10f);

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 16, null, ContactSlots))
            {
                buffers.Spawn(BodyDescriptor.Box(new float3(100f, 100f, 1f), 0f, new float3(0f, 0f, -0.5f)));
                var boxes = new BodyHandle[8];
                for (int i = 0; i < boxes.Length; i++)
                {
                    // Two stacks of three and two boxes dropped at an angle, so pairs and contact counts vary.
                    float3 p = i < 3 ? new float3(0f, 0f, 0.5f + i * 0.98f)
                             : i < 6 ? new float3(2f, 0f, 0.5f + (i - 3) * 0.98f)
                             : new float3(1f + (i - 6) * 0.3f, 0.4f, 0.5f + (i - 6) * 1.4f);
                    boxes[i] = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, p));
                }
                constraints.AddJoint(boxes[0], boxes[6], new float3(0.5f, 0f, 0f), new float3(-0.5f, 0f, 0f)); // a joint between contacting bodies
                constraints.AddSpring(boxes[2], boxes[5], float3.zero, float3.zero, 20f);
                var solver = new AvbdSolver(buffers, settings, null, constraints);

                for (int step = 0; step < 6; step++)
                {
                    solver.Step();
                    AssertListsMatchOracle(buffers, constraints, "step " + step);
                    Assert.AreEqual(0, constraints.ReadListControl().Invalid, "step " + step);
                }
                Assert.Greater(constraints.ReadContacts().SlotsWithContacts, 3, "the scene has contacts, so the test covers them");
            }
        }

        [Test]
        public void Gpu_Lists_ListAConstraintToAMasslessBodyForItsSolvedSideOnly()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 8))
            {
                BodyHandle a = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, float3.zero));
                BodyHandle b = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(1.5f, 0f, 0f)));
                constraints.AddJoint(a, b, float3.zero, float3.zero);
                constraints.Rebuild();
                AssertListsMatchOracle(buffers, constraints, "before");

                // A spring to a body with no mass: the new constraint is listed for b only, and the solved set grew by a body that is not solved.
                BodyHandle c = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 0f, new float3(3f, 0f, 0f)));
                constraints.AddSpring(b, c, float3.zero, float3.zero, 4f);
                constraints.Rebuild();
                AssertListsMatchOracle(buffers, constraints, "after adding a massless partner");
            }
        }

        // ---- helpers -------------------------------------------------------------------------------------------

        // Builds the lists of the current records, bodies and manifold pool with the CPU oracle and compares them with the GPU's, entry by entry.
        static void AssertListsMatchOracle(AvbdBodyBuffers buffers, AvbdConstraintBuffers constraints, string what)
        {
            int bodyCount = buffers.HighWaterMark;
            var meta = new BodyMetaRecord[buffers.Capacity];
            var mass = new BodyMassRecord[buffers.Capacity];
            buffers.Meta.GetData(meta);
            buffers.Mass.GetData(mass);
            var solved = new bool[bodyCount];
            const uint notSolved = (uint)BodyFlags.Free | (uint)BodyFlags.Kinematic;
            for (int s = 0; s < bodyCount; s++)
                solved[s] = (meta[s].ident.x & notSolved) == 0u && mass[s].momentMass.w > 0f;

            var records = new ConstraintRecord[constraints.Capacity];
            if (constraints.RecordCount > 0)
                constraints.Records.GetData(records, 0, 0, constraints.RecordCount);

            var oracle = new ConstraintTopology();
            if (constraints.HasContacts)
            {
                ContactSnapshot pool = constraints.ReadContacts();
                oracle.BuildWithManifolds(records, constraints.RecordCount, solved, bodyCount, false, pool.Counts, pool.Bodies,
                    constraints.ContactCapacity, constraints.ContactBase);
            }
            else
            {
                oracle.Build(records, constraints.RecordCount, solved, bodyCount, false);
            }

            var ranges = new BodyRangeRecord[buffers.Capacity];
            var list = new uint[constraints.List.count];
            var others = new uint[constraints.List.count];
            buffers.Range.GetData(ranges);
            constraints.List.GetData(list);
            constraints.ListOther.GetData(others);

            Assert.AreEqual(oracle.ListCount, constraints.ReadListControl().Total, what + ": number of entries");
            for (int s = 0; s < bodyCount; s++)
            {
                Assert.AreEqual(oracle.Ranges[s].range.x, ranges[s].range.x, what + ": offset of body " + s);
                Assert.AreEqual(oracle.Ranges[s].range.y, ranges[s].range.y, what + ": count of body " + s);
                Assert.AreEqual(oracle.Ranges[s].range.z, ranges[s].range.z, what + ": contact entries of body " + s);
                for (int e = 0; e < (int)oracle.Ranges[s].range.y; e++)
                {
                    int at = (int)oracle.Ranges[s].range.x + e;
                    Assert.AreEqual(oracle.List[at], list[at], what + ": entry " + e + " of body " + s);
                    Assert.AreEqual(oracle.ListOther[at], others[at], what + ": body on the other side of entry " + e + " of body " + s);
                }
            }
        }

        static void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }
    }
}
