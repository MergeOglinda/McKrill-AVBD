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
                    Assert.AreEqual(before.Meta[h.Slot].ident, after.Meta[h.Slot].ident, "group mask");
                    Assert.AreEqual(before.Flags[h.Slot], after.Flags[h.Slot], "flags");
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
        public void Hold_PutsEveryBodyAtItsOwnPlaceRelativeToThePoint_AndSetsTheVelocity()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var mover = new AvbdGroupMover(buffers))
            {
                BodyHandle a = Spawn(buffers, new float3(0f, 0f, 0f), 1f, Group, new float3(4f, 0f, 0f));
                BodyHandle b = Spawn(buffers, new float3(2f, 1f, 0f), 2f, Group | 1u, new float3(0f, -3f, 0f));
                BodyHandle other = Spawn(buffers, new float3(5f, 5f, 5f), 1f, 1u, new float3(0f, 0f, 9f));
                BodyHandle wall = Spawn(buffers, new float3(0f, -1f, 0f), 0f, Group, float3.zero); // static: not part of the group
                buffers.ApplyEdits();
                Snapshot before = Read(buffers);
                float3 start = new float3(1f, 0f, 0f);

                mover.Hold(Group, start, float3.zero);
                Snapshot noCapture = Read(buffers);
                Assert.AreEqual(before.State[a.Slot].position, noCapture.State[a.Slot].position, "Hold does nothing before BeginHold");

                mover.BeginHold(Group, start);
                Snapshot begun = Read(buffers);
                Assert.AreEqual(before.State[a.Slot].position, begun.State[a.Slot].position, "BeginHold moves nothing");
                Assert.AreEqual(before.Motion[a.Slot].linearVel, begun.Motion[a.Slot].linearVel, "BeginHold changes no velocity");

                // The steps in between pull the bodies apart (each one its own way), and the hold puts each one back at its own place, relative to the point.
                mover.Translate(Group, new float3(0f, -0.0625f, 0f)); // what a step does to the group
                mover.Hold(Group, new float3(1f, 0f, 3f), new float3(0f, 0f, 10f));
                Snapshot after = Read(buffers);
                AssertNear(new float3(0f, 0f, 3f), after.State[a.Slot].position.xyz, "a: the point (1, 0, 3) plus its offset (-1, 0, 0)");
                AssertNear(new float3(2f, 1f, 3f), after.State[b.Slot].position.xyz, "b: the point plus its offset (1, 1, 0)");
                Assert.AreEqual(before.State[other.Slot].position, after.State[other.Slot].position, "another group");
                Assert.AreEqual(before.State[wall.Slot].position, after.State[wall.Slot].position, "a static body is not held");

                foreach (BodyHandle h in new[] { a, b })
                {
                    AssertNear(new float3(0f, 0f, 10f), after.Motion[h.Slot].linearVel.xyz, "velocity of the point");
                    AssertNear(new float3(0f, 0f, 10f), after.Motion[h.Slot].prevLinearVel.xyz, "previous velocity");
                    Assert.AreEqual(before.Motion[h.Slot].angularVel, after.Motion[h.Slot].angularVel, "angular velocity stays");
                    Assert.AreEqual(after.State[h.Slot].position, after.State[h.Slot].startPosition, "start pose follows");
                    Assert.AreEqual(before.State[h.Slot].rotation, after.State[h.Slot].rotation);
                    Assert.AreEqual(before.Mass[h.Slot].momentMass, after.Mass[h.Slot].momentMass, "mass and moments");
                    Assert.AreEqual(before.Shape[h.Slot].sizeFriction, after.Shape[h.Slot].sizeFriction, "shape");
                    Assert.AreEqual(before.Meta[h.Slot].ident, after.Meta[h.Slot].ident, "group mask");
                    Assert.AreEqual(before.Flags[h.Slot], after.Flags[h.Slot], "flags");
                }
                Assert.AreEqual(before.Motion[other.Slot].linearVel, after.Motion[other.Slot].linearVel, "another group keeps its velocity");
            }
        }

        [Test]
        public void Hold_GpuMatchesTheOracle_OnRandomBodies_AndABodyThatJoinsLaterIsNotHeld()
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

                const uint Mask = 2u;
                float3 start = new float3(0f, 3f, 0f);
                mover.BeginHold(Mask, start);
                float4[] offsets = CpuGroupMove.HoldCapture(oracle, buffers.HighWaterMark, Mask, start);
                Assert.IsTrue(offsets.Length > 0 && Array.Exists(offsets, o => o.w == 1f), "the random group is not empty");

                // Drift the group, then hold at another point: every body ends where the oracle puts it.
                float3 drift = new float3(0.3f, -0.2f, 0.1f);
                mover.Translate(Mask, drift);
                CpuGroupMove.Run(oracle, buffers.HighWaterMark, Mask, GroupMoveFields.Position, drift, float3.zero, float3.zero);
                float3 point = new float3(1.5f, 2.75f, 2f);
                float3 velocity = new float3(0.5f, -1f, 8f);
                mover.Hold(Mask, point, velocity);
                CpuGroupMove.Hold(oracle, buffers.HighWaterMark, Mask, point, velocity, offsets);

                Snapshot gpu = Read(buffers);
                for (int slot = 0; slot < Capacity; slot++)
                {
                    Assert.AreEqual(oracle.State[slot], gpu.State[slot], "state of slot " + slot);
                    Assert.AreEqual(oracle.Motion[slot], gpu.Motion[slot], "motion of slot " + slot);
                    Assert.AreEqual(oracle.Mass[slot], gpu.Mass[slot], "mass of slot " + slot);
                }

                // A body that is spawned after the capture is in the group (the mask) but was not remembered: it stays where it is.
                BodyDescriptor late = BodyDescriptor.Box(new float3(1f), 1f, new float3(9f, 9f, 9f));
                late.groupMask = Mask;
                BodyHandle lateBody = buffers.Spawn(late);
                buffers.ApplyEdits();
                mover.Hold(Mask, point + new float3(5f, 0f, 0f), float3.zero);
                Snapshot afterLate = Read(buffers);
                AssertNear(new float3(9f, 9f, 9f), afterLate.State[lateBody.Slot].position.xyz, "a body that joined later is not held");
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
                Assert.Throws<ArgumentException>(() => mover.Hold(1u, new float3(0f, float.NegativeInfinity, 0f), float3.zero));
                Assert.Throws<ArgumentException>(() => mover.Hold(1u, float3.zero, new float3(0f, 0f, float.NaN)));
                Assert.DoesNotThrow(() => mover.BeginHold(0u, float3.zero), "an empty mask does nothing");
                mover.Dispose();
                Assert.Throws<ObjectDisposedException>(() => mover.BeginHold(1u, float3.zero));
                Assert.Throws<ObjectDisposedException>(() => mover.Hold(1u, float3.zero, float3.zero));
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
            public uint[] Flags;
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
                Flags = new uint[buffers.Capacity],
            };
            buffers.State.GetData(s.State);
            PoseMirror.Assert(buffers, "after a move");
            buffers.Motion.GetData(s.Motion);
            buffers.Mass.GetData(s.Mass);
            buffers.Shape.GetData(s.Shape);
            buffers.Meta.GetData(s.Meta);
            buffers.Flags.GetData(s.Flags);
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

        static void AssertNear(float3 expected, float3 actual, string what)
        {
            Assert.IsTrue(math.all(math.abs(expected - actual) < 1e-5f), what + ": expected " + expected + " but was " + actual);
        }

        static void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }
    }
}
