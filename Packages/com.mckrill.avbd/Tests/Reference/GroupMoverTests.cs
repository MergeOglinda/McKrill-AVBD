using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>The GPU group mover (<see cref="AvbdGroupMover"/>): by hand, and against its CPU oracle.</summary>
    public class GroupMoverTests
    {
        const uint Group = 2u;

        [Test]
        public void Translate_MovesTheBodiesOfTheGroup_AndNothingElse()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var mover = new AvbdGroupMover(buffers))
            {
                BodyHandle a = Spawn(buffers, new float3(1f, 2f, 3f), 1f, Group, new float3(4f, 0f, 0f));
                BodyHandle b = Spawn(buffers, new float3(10f, 0f, 0f), 2f, Group | 1u, float3.zero);
                BodyHandle other = Spawn(buffers, new float3(5f, 5f, 5f), 1f, 1u, float3.zero);
                BodyHandle wall = Spawn(buffers, new float3(0f, -1f, 0f), 0f, Group, float3.zero); // static
                buffers.ApplyEdits();
                Snapshot before = Read(buffers);

                mover.Translate(Group, new float3(0f, 7f, -1f));
                Snapshot after = Read(buffers);

                AssertNear(new float3(1f, 9f, 2f), after.State[a.Slot].position.xyz, "a moves");
                AssertNear(new float3(10f, 7f, -1f), after.State[b.Slot].position.xyz, "b moves, it is in the group among others");
                Assert.AreEqual(before.State[other.Slot].position, after.State[other.Slot].position, "another group");
                Assert.AreEqual(before.State[wall.Slot].position, after.State[wall.Slot].position, "a static body is not moved");

                // A teleport: the step-start pose follows, the rotation, the velocity and what makes the inertia are as they were.
                foreach (BodyHandle h in new[] { a, b })
                {
                    Assert.AreEqual(after.State[h.Slot].position, after.State[h.Slot].startPosition, "start pose");
                    Assert.AreEqual(before.State[h.Slot].rotation, after.State[h.Slot].rotation);
                    Assert.AreEqual(before.Motion[h.Slot].linearVel, after.Motion[h.Slot].linearVel, "velocity");
                    Assert.AreEqual(before.Motion[h.Slot].angularVel, after.Motion[h.Slot].angularVel);
                    Assert.AreEqual(before.Mass[h.Slot].momentMass, after.Mass[h.Slot].momentMass, "mass and moments");
                    Assert.AreEqual(before.Shape[h.Slot].sizeFriction, after.Shape[h.Slot].sizeFriction, "shape");
                    Assert.AreEqual(before.Meta[h.Slot].ident, after.Meta[h.Slot].ident, "flags and group mask");
                }
            }
        }

        [Test]
        public void SetVelocity_SetsTheVelocities_AndKeepsThePositions()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var mover = new AvbdGroupMover(buffers))
            {
                BodyHandle a = Spawn(buffers, new float3(1f, 2f, 3f), 1f, Group, new float3(4f, 0f, 0f));
                BodyHandle other = Spawn(buffers, new float3(5f, 5f, 5f), 1f, 1u, new float3(0f, 0f, 9f));
                buffers.ApplyEdits();
                Snapshot before = Read(buffers);

                mover.SetVelocity(Group, new float3(0f, 3f, 0f), new float3(0f, 0f, 2f));
                Snapshot after = Read(buffers);
                AssertNear(new float3(0f, 3f, 0f), after.Motion[a.Slot].linearVel.xyz, "linear");
                AssertNear(new float3(0f, 3f, 0f), after.Motion[a.Slot].prevLinearVel.xyz, "previous linear, for the warm start");
                AssertNear(new float3(0f, 0f, 2f), after.Motion[a.Slot].angularVel.xyz, "angular");
                Assert.AreEqual(before.State[a.Slot], after.State[a.Slot], "the pose is as it was");
                Assert.AreEqual(before.Motion[other.Slot].linearVel, after.Motion[other.Slot].linearVel, "another group");

                // Only the linear velocity: the angular one stays.
                mover.Move(Group, GroupMoveFields.LinearVelocity, float3.zero, new float3(7f, 0f, 0f), new float3(100f));
                after = Read(buffers);
                AssertNear(new float3(7f, 0f, 0f), after.Motion[a.Slot].linearVel.xyz, "linear only");
                AssertNear(new float3(0f, 0f, 2f), after.Motion[a.Slot].angularVel.xyz, "angular unchanged");
            }
        }

        [Test]
        public void MoveWithVelocity_DoesBoth_AndAnEmptyMaskOrNoFieldsDoesNothing()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var mover = new AvbdGroupMover(buffers))
            {
                BodyHandle a = Spawn(buffers, new float3(0f), 1f, Group, float3.zero);
                buffers.ApplyEdits();

                mover.Move(0u, GroupMoveFields.Position, new float3(1f), float3.zero, float3.zero);
                mover.Move(Group, GroupMoveFields.None, new float3(1f), new float3(1f), new float3(1f));
                Snapshot same = Read(buffers);
                AssertNear(float3.zero, same.State[a.Slot].position.xyz, "no mask, no fields");

                mover.MoveWithVelocity(Group, new float3(0f, 0.5f, 0f), new float3(0f, 15f, 0f));
                Snapshot moved = Read(buffers);
                AssertNear(new float3(0f, 0.5f, 0f), moved.State[a.Slot].position.xyz, "position");
                AssertNear(new float3(0f, 15f, 0f), moved.Motion[a.Slot].linearVel.xyz, "velocity");
            }
        }

        [Test]
        public void Gpu_MatchesTheOracle_OnRandomBodiesAndMasks()
        {
            RequireCompute();
            var rng = new System.Random(5);
            const int Capacity = 128;
            var oracle = new CpuBodyArrays(Capacity);
            var applied = new List<EditCommand>();

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var mover = new AvbdGroupMover(buffers))
            {
                for (int i = 0; i < 100; i++)
                {
                    BodyDescriptor d = BodyDescriptor.Box(new float3(0.5f + (float)rng.NextDouble()), i % 7 == 0 ? 0f : 0.5f + (float)rng.NextDouble(),
                        new float3((float)rng.NextDouble() * 20f - 10f, (float)rng.NextDouble() * 20f, (float)rng.NextDouble() * 20f - 10f));
                    d.rotation = quaternion.EulerXYZ((float)rng.NextDouble() * 6f, (float)rng.NextDouble() * 6f, (float)rng.NextDouble() * 6f);
                    d.linearVelocity = new float3((float)rng.NextDouble() * 4f, 0f, 0f);
                    d.angularVelocity = new float3(0f, (float)rng.NextDouble() * 4f, 0f);
                    d.kinematic = i % 11 == 0;
                    d.groupMask = (uint)rng.Next(0, 8);
                    buffers.Spawn(d);
                }
                buffers.Remove(buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, float3.zero))); // a free slot below the high-water mark
                buffers.ApplyEdits(applied);
                oracle.Apply(applied);

                GroupMoveFields[] cases =
                {
                    GroupMoveFields.Position, GroupMoveFields.LinearVelocity, GroupMoveFields.AngularVelocity,
                    GroupMoveFields.Position | GroupMoveFields.LinearVelocity | GroupMoveFields.AngularVelocity,
                };
                uint[] masks = { 1u, 2u, 6u, 0xFFFFFFFFu };
                for (int k = 0; k < cases.Length; k++)
                {
                    float3 t = new float3(k + 0.25f, -2f * k, 0.5f);
                    float3 lin = new float3(3f * k, 1f, -k);
                    float3 ang = new float3(0.5f, k, 2f);
                    mover.Move(masks[k], cases[k], t, lin, ang);
                    CpuGroupMove.Run(oracle, buffers.HighWaterMark, masks[k], cases[k], t, lin, ang);
                }

                Snapshot gpu = Read(buffers);
                for (int slot = 0; slot < Capacity; slot++)
                {
                    Assert.AreEqual(oracle.State[slot], gpu.State[slot], "state of slot " + slot);
                    Assert.AreEqual(oracle.Motion[slot], gpu.Motion[slot], "motion of slot " + slot);
                    Assert.AreEqual(oracle.Mass[slot], gpu.Mass[slot], "mass of slot " + slot);
                }
            }
        }

        [Test]
        public void Move_RejectsBadValues_AndAfterDispose()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var mover = new AvbdGroupMover(buffers))
            {
                Assert.Throws<ArgumentException>(() => mover.Translate(1u, new float3(float.NaN, 0f, 0f)));
                Assert.Throws<ArgumentException>(() => mover.SetVelocity(1u, new float3(0f, float.PositiveInfinity, 0f), float3.zero));
                Assert.Throws<ArgumentException>(() => mover.SetVelocity(1u, float3.zero, new float3(0f, 0f, float.NaN)));
                Assert.DoesNotThrow(() => mover.Move(1u, GroupMoveFields.Position, float3.zero, new float3(float.NaN), new float3(float.NaN)),
                    "values of fields that are not selected are ignored");
                mover.Dispose();
                Assert.Throws<ObjectDisposedException>(() => mover.Translate(1u, float3.zero));
            }
        }

        [Test]
        public void Hold_PutsTheCentroidAtTheAnchorPlusTheOffset_AndZeroesTheVelocities()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var mover = new AvbdGroupMover(buffers))
            {
                BodyHandle a = Spawn(buffers, new float3(0f, 0f, 0f), 1f, Group, new float3(4f, 0f, 0f));
                BodyHandle b = Spawn(buffers, new float3(2f, 0f, 0f), 2f, Group | 1u, new float3(0f, -3f, 0f));
                BodyHandle other = Spawn(buffers, new float3(5f, 5f, 5f), 1f, 1u, new float3(0f, 0f, 9f));
                BodyHandle wall = Spawn(buffers, new float3(0f, -1f, 0f), 0f, Group, float3.zero); // static: not part of the group
                buffers.ApplyEdits();
                Snapshot before = Read(buffers);
                float3 reference = new float3(1f, 0f, 0f);

                mover.Hold(Group, reference, new float3(0f, 1f, 0f));
                Snapshot noAnchor = Read(buffers);
                Assert.AreEqual(before.State[a.Slot].position, noAnchor.State[a.Slot].position, "Hold does nothing before BeginHold");

                mover.BeginHold(Group, reference); // the centroid is (1, 0, 0)
                Snapshot begun = Read(buffers);
                Assert.AreEqual(before.State[a.Slot].position, begun.State[a.Slot].position, "BeginHold moves nothing");
                Assert.AreEqual(before.Motion[a.Slot].linearVel, begun.Motion[a.Slot].linearVel, "BeginHold changes no velocity");

                // The step lets the group sink and gives it a velocity; the hold puts the centroid back at the anchor, plus the offset, and the velocities at zero.
                for (int i = 0; i < 10; i++)
                {
                    mover.Translate(Group, new float3(0f, -0.0625f, 0f)); // a multiple of 1/1024, so that the centroid the hold takes is exact
                    mover.SetVelocity(Group, new float3(0f, -1f, 0f), new float3(1f, 2f, 3f));
                    mover.Hold(Group, reference, new float3(0f, 0f, 3f));
                }
                Snapshot after = Read(buffers);
                AssertNear(new float3(0f, 0f, 3f), after.State[a.Slot].position.xyz, "a: the centroid is at (1, 0, 3) and the group keeps its shape");
                AssertNear(new float3(2f, 0f, 3f), after.State[b.Slot].position.xyz, "b");
                Assert.AreEqual(before.State[other.Slot].position, after.State[other.Slot].position, "another group");
                Assert.AreEqual(before.State[wall.Slot].position, after.State[wall.Slot].position, "a static body is not held");

                foreach (BodyHandle h in new[] { a, b })
                {
                    Assert.AreEqual(float3.zero, after.Motion[h.Slot].linearVel.xyz, "linear velocity is zero");
                    Assert.AreEqual(float3.zero, after.Motion[h.Slot].prevLinearVel.xyz, "previous linear velocity is zero");
                    Assert.AreEqual(float3.zero, after.Motion[h.Slot].angularVel.xyz, "angular velocity is zero");
                    Assert.AreEqual(after.State[h.Slot].position, after.State[h.Slot].startPosition, "start pose follows");
                    Assert.AreEqual(before.State[h.Slot].rotation, after.State[h.Slot].rotation);
                    Assert.AreEqual(before.Mass[h.Slot].momentMass, after.Mass[h.Slot].momentMass, "mass and moments");
                    Assert.AreEqual(before.Meta[h.Slot].ident, after.Meta[h.Slot].ident, "flags and group mask");
                }
                Assert.AreEqual(before.Motion[other.Slot].linearVel, after.Motion[other.Slot].linearVel, "another group keeps its velocity");
            }
        }

        [Test]
        public void Hold_GpuMatchesTheOracle_OnRandomBodies()
        {
            RequireCompute();
            var rng = new System.Random(8);
            const int Capacity = 128;
            var oracle = new CpuBodyArrays(Capacity);
            var applied = new List<EditCommand>();

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var mover = new AvbdGroupMover(buffers))
            {
                for (int i = 0; i < 100; i++)
                {
                    BodyDescriptor d = BodyDescriptor.Box(new float3(0.5f + (float)rng.NextDouble()), i % 7 == 0 ? 0f : 0.5f + (float)rng.NextDouble(),
                        new float3((float)rng.NextDouble() * 6f - 3f, (float)rng.NextDouble() * 6f, (float)rng.NextDouble() * 6f - 3f));
                    d.linearVelocity = new float3((float)rng.NextDouble() * 4f, 0f, 0f);
                    d.angularVelocity = new float3(0f, (float)rng.NextDouble() * 4f, 0f);
                    d.kinematic = i % 11 == 0;
                    d.groupMask = (uint)rng.Next(0, 8);
                    buffers.Spawn(d);
                }
                buffers.Remove(buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, float3.zero))); // a free slot below the high-water mark
                buffers.ApplyEdits(applied);
                oracle.Apply(applied);

                float3 reference = new float3(0f, 3f, 0f);
                const uint Mask = 2u;
                mover.BeginHold(Mask, reference);
                Assert.IsTrue(CpuGroupMove.Centroid(oracle, buffers.HighWaterMark, Mask, reference, out float3 anchor));

                // Drift the group, then hold at an offset: the bodies end up as the oracle puts them.
                float3 drift = new float3(0.3f, -0.2f, 0.1f);
                mover.Translate(Mask, drift);
                CpuGroupMove.Run(oracle, buffers.HighWaterMark, Mask, GroupMoveFields.Position, drift, float3.zero, float3.zero);
                float3 offset = new float3(1.5f, -0.25f, 2f);
                mover.Hold(Mask, reference, offset);
                CpuGroupMove.Hold(oracle, buffers.HighWaterMark, Mask, reference, anchor, offset);

                Snapshot gpu = Read(buffers);
                for (int slot = 0; slot < Capacity; slot++)
                {
                    AssertNear(oracle.State[slot].position.xyz, gpu.State[slot].position.xyz, "position of slot " + slot, 1e-3f);
                    AssertNear(oracle.State[slot].startPosition.xyz, gpu.State[slot].startPosition.xyz, "start position of slot " + slot, 1e-3f);
                    Assert.AreEqual(oracle.Motion[slot], gpu.Motion[slot], "motion of slot " + slot);
                    Assert.AreEqual(oracle.Mass[slot], gpu.Mass[slot], "mass of slot " + slot);
                }
            }
        }

        [Test]
        public void Hold_RejectsBadValues()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var mover = new AvbdGroupMover(buffers))
            {
                Assert.Throws<ArgumentException>(() => mover.BeginHold(1u, new float3(float.NaN, 0f, 0f)));
                Assert.Throws<ArgumentException>(() => mover.Hold(1u, float3.zero, new float3(0f, float.PositiveInfinity, 0f)));
                Assert.DoesNotThrow(() => mover.Hold(0u, float3.zero, float3.zero), "an empty mask does nothing");
                mover.Dispose();
                Assert.Throws<ObjectDisposedException>(() => mover.BeginHold(1u, float3.zero));
            }
        }

        // ---- helpers --------------------------------------------------------------------------------------

        struct Snapshot
        {
            public BodyStateRecord[] State;
            public BodyMotionRecord[] Motion;
            public BodyMassRecord[] Mass;
            public BodyShapeRecord[] Shape;
            public BodyMetaRecord[] Meta;
        }

        static Snapshot Read(AvbdBodyBuffers buffers)
        {
            var s = new Snapshot
            {
                State = new BodyStateRecord[buffers.Capacity],
                Motion = new BodyMotionRecord[buffers.Capacity],
                Mass = new BodyMassRecord[buffers.Capacity],
                Shape = new BodyShapeRecord[buffers.Capacity],
                Meta = new BodyMetaRecord[buffers.Capacity],
            };
            buffers.State.GetData(s.State);
            buffers.Motion.GetData(s.Motion);
            buffers.Mass.GetData(s.Mass);
            buffers.Shape.GetData(s.Shape);
            buffers.Meta.GetData(s.Meta);
            return s;
        }

        // A box of 1 m with the given density (0: static).
        static BodyHandle Spawn(AvbdBodyBuffers buffers, float3 position, float density, uint groupMask, float3 linearVelocity)
        {
            BodyDescriptor d = BodyDescriptor.Box(new float3(1f), density, position);
            d.groupMask = groupMask;
            d.linearVelocity = linearVelocity;
            d.angularVelocity = new float3(0.1f, 0.2f, 0.3f);
            return buffers.Spawn(d);
        }

        static void AssertNear(float3 expected, float3 actual, string what, float tolerance = 1e-5f)
        {
            Assert.IsTrue(math.all(math.abs(expected - actual) < tolerance), what + ": expected " + expected + " but was " + actual);
        }

        static void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }
    }
}
