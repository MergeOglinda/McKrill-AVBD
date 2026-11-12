using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;

namespace McKrill.Avbd.Tests
{
    public class BodyEditQueueTests
    {
        static BodyDescriptor Box(float3 position)
        {
            return BodyDescriptor.Box(new float3(1f, 2f, 3f), 2f, position);
        }

        static List<EditCommand> Drain(BodyEditQueue queue)
        {
            var commands = new List<EditCommand>();
            queue.Drain(commands);
            return commands;
        }

        static BodyHandle SpawnAndApply(BodyEditQueue queue, float3 position = default)
        {
            BodyHandle handle = queue.Spawn(Box(position));
            Drain(queue);
            return handle;
        }

        static void AssertEqual(float4 expected, float4 actual, string what)
        {
            Assert.IsTrue(math.all(expected == actual), what + ": expected " + expected + " but was " + actual);
        }

        // ---- spawn ------------------------------------------------------------------------------------

        [Test]
        public void Spawn_EncodesEveryField()
        {
            var queue = new BodyEditQueue(8);
            BodyDescriptor d = BodyDescriptor.Box(new float3(1f, 2f, 3f), 2f, new float3(4f, 5f, 6f));
            d.linearVelocity = new float3(7f, 8f, 9f);
            d.angularVelocity = new float3(0.5f, 0.25f, 0.125f);
            d.friction = 0.25f;
            d.groupMask = 0xA5u;
            d.appearanceId = 42u;
            d.kinematic = true;

            BodyHandle handle = queue.Spawn(d);
            List<EditCommand> commands = Drain(queue);

            Assert.IsTrue(handle.IsValid);
            Assert.AreEqual(1, commands.Count);
            EditCommand c = commands[0];
            Assert.AreEqual(EditKind.Spawn, c.Kind);
            Assert.AreEqual(handle.Slot, c.Slot);
            Assert.AreEqual(EditFields.All, c.Fields);
            AssertEqual(new float4(4f, 5f, 6f, 0f), c.position, "position");
            AssertEqual(new float4(0f, 0f, 0f, 1f), c.rotation, "rotation");
            AssertEqual(new float4(7f, 8f, 9f, 0f), c.linear, "linear");
            AssertEqual(new float4(0.5f, 0.25f, 0.125f, 0f), c.angular, "angular");
            AssertEqual(new float4(1f, 2f, 3f, 0.25f), c.sizeFriction, "sizeFriction");

            // Volume 6, density 2: mass 12. Moments follow the box formula.
            Assert.AreEqual(12f, c.momentMass.w, 1e-5f);
            Assert.AreEqual(13f, c.momentMass.x, 1e-4f);
            Assert.AreEqual(10f, c.momentMass.y, 1e-4f);
            Assert.AreEqual(5f, c.momentMass.z, 1e-4f);

            Assert.AreEqual(BodyFlagLayout.Pack(BodyFlags.Kinematic, ShapeKind.Box), c.meta.x);
            Assert.AreEqual(0xA5u, c.meta.y);
            Assert.AreEqual(42u, c.meta.z);
        }

        [Test]
        public void Spawn_NormalizesTheRotation()
        {
            var queue = new BodyEditQueue(2);
            BodyDescriptor d = Box(float3.zero);
            d.rotation = new quaternion(0f, 0f, 0f, 2f);
            queue.Spawn(d);
            AssertEqual(new float4(0f, 0f, 0f, 1f), Drain(queue)[0].rotation, "rotation");
        }

        [Test]
        public void Spawn_RejectsBadValues_WithoutUsingASlot()
        {
            var queue = new BodyEditQueue(4);

            BodyDescriptor zeroSize = Box(float3.zero);
            zeroSize.size = new float3(1f, 0f, 1f);
            Assert.Throws<ArgumentException>(() => queue.Spawn(zeroSize));

            BodyDescriptor nanPosition = Box(new float3(float.NaN, 0f, 0f));
            Assert.Throws<ArgumentException>(() => queue.Spawn(nanPosition));

            BodyDescriptor negativeDensity = Box(float3.zero);
            negativeDensity.density = -1f;
            Assert.Throws<ArgumentException>(() => queue.Spawn(negativeDensity));

            BodyDescriptor reservedAppearance = Box(float3.zero);
            reservedAppearance.appearanceId = AvbdConst.Invalid;
            Assert.Throws<ArgumentException>(() => queue.Spawn(reservedAppearance));

            BodyDescriptor zeroRotation = Box(float3.zero);
            zeroRotation.rotation = default;
            Assert.Throws<ArgumentException>(() => queue.Spawn(zeroRotation));

            Assert.AreEqual(0, queue.AliveCount);
            Assert.AreEqual(0, queue.PendingSlotCount);
        }

        [Test]
        public void Spawn_WhenFull_ReturnsInvalidAndCountsTheOverflow()
        {
            var queue = new BodyEditQueue(2);
            Assert.IsTrue(queue.Spawn(Box(float3.zero)).IsValid);
            Assert.IsTrue(queue.Spawn(Box(float3.zero)).IsValid);

            BodyHandle none = queue.Spawn(Box(float3.zero));

            Assert.IsFalse(none.IsValid);
            Assert.AreEqual(1, queue.OverflowCount);
            Assert.AreEqual(2, queue.AliveCount);
            Assert.AreEqual(2, Drain(queue).Count);
        }

        [Test]
        public void Drain_EmptiesTheQueue()
        {
            var queue = new BodyEditQueue(4);
            queue.Spawn(Box(float3.zero));
            Assert.AreEqual(1, queue.PendingSlotCount);
            Assert.AreEqual(1, Drain(queue).Count);
            Assert.AreEqual(0, queue.PendingSlotCount);
            Assert.AreEqual(0, Drain(queue).Count);
        }

        // ---- set --------------------------------------------------------------------------------------

        [Test]
        public void Set_OnALiveBody_EmitsOneSetCommandWithTheTouchedFields()
        {
            var queue = new BodyEditQueue(4);
            BodyHandle h = SpawnAndApply(queue);

            Assert.IsTrue(queue.SetPose(h, new float3(1f, 2f, 3f), quaternion.identity));
            Assert.IsTrue(queue.SetFriction(h, 0.75f));
            List<EditCommand> commands = Drain(queue);

            Assert.AreEqual(1, commands.Count);
            EditCommand c = commands[0];
            Assert.AreEqual(EditKind.Set, c.Kind);
            Assert.AreEqual(h.Slot, c.Slot);
            Assert.AreEqual(EditFields.Pose | EditFields.Friction, c.Fields);
            AssertEqual(new float4(1f, 2f, 3f, 0f), c.position, "position");
            Assert.AreEqual(0.75f, c.sizeFriction.w);
        }

        [Test]
        public void SetShape_ComputesMassFromTheDensity()
        {
            var queue = new BodyEditQueue(4);
            BodyHandle h = SpawnAndApply(queue);

            queue.SetShape(h, new float3(2f, 2f, 2f), 1f);
            EditCommand c = Drain(queue)[0];

            Assert.AreEqual(EditFields.Shape, c.Fields);
            AssertEqual(new float4(2f, 2f, 2f, 0f), new float4(c.sizeFriction.xyz, 0f), "size");
            Assert.AreEqual(8f, c.momentMass.w, 1e-5f);
            AssertEqual(BoxMassProperties.Compute(new float3(2f), 1f), c.momentMass, "mass properties");
        }

        [Test]
        public void SetShape_WithZeroDensity_MakesTheBodyStatic()
        {
            var queue = new BodyEditQueue(4);
            BodyHandle h = SpawnAndApply(queue);
            queue.SetShape(h, new float3(1f), 0f);
            Assert.AreEqual(0f, Drain(queue)[0].momentMass.w);
        }

        [Test]
        public void SetVelocity_SetGroupMaskAndAppearance_AreEncoded()
        {
            var queue = new BodyEditQueue(4);
            BodyHandle h = SpawnAndApply(queue);

            queue.SetVelocity(h, new float3(1f, 0f, 0f), new float3(0f, 2f, 0f));
            queue.SetGroupMask(h, 0x0Fu);
            queue.SetAppearanceId(h, 9u);
            EditCommand c = Drain(queue)[0];

            Assert.AreEqual(EditFields.Velocity | EditFields.GroupMask | EditFields.Appearance, c.Fields);
            AssertEqual(new float4(1f, 0f, 0f, 0f), c.linear, "linear");
            AssertEqual(new float4(0f, 2f, 0f, 0f), c.angular, "angular");
            Assert.AreEqual(0x0Fu, c.meta.y);
            Assert.AreEqual(9u, c.meta.z);
        }

        [Test]
        public void SetKinematic_TogglesOnlyTheKinematicBit()
        {
            var queue = new BodyEditQueue(4);
            BodyHandle h = SpawnAndApply(queue);

            queue.SetKinematic(h, true);
            EditCommand on = Drain(queue)[0];
            Assert.AreEqual(EditFields.Kinematic, on.Fields);
            Assert.AreEqual((uint)BodyFlags.Kinematic, on.meta.x & BodyFlagLayout.UserSettable);

            queue.SetKinematic(h, false);
            EditCommand off = Drain(queue)[0];
            Assert.AreEqual(0u, off.meta.x & BodyFlagLayout.UserSettable);
        }

        [Test]
        public void Set_RejectsBadValues()
        {
            var queue = new BodyEditQueue(4);
            BodyHandle h = SpawnAndApply(queue);

            Assert.Throws<ArgumentException>(() => queue.SetPose(h, new float3(float.PositiveInfinity, 0f, 0f), quaternion.identity));
            Assert.Throws<ArgumentException>(() => queue.SetVelocity(h, new float3(float.NaN), float3.zero));
            Assert.Throws<ArgumentException>(() => queue.SetShape(h, new float3(1f, -1f, 1f), 1f));
            Assert.Throws<ArgumentException>(() => queue.SetFriction(h, -0.1f));
            Assert.Throws<ArgumentException>(() => queue.SetAppearanceId(h, AvbdConst.Invalid));
            Assert.AreEqual(0, Drain(queue).Count, "a rejected edit must leave nothing queued");
        }

        // ---- stale and invalid handles ----------------------------------------------------------------

        [Test]
        public void EditsThroughInvalidOrStaleHandles_ReturnFalseAndQueueNothing()
        {
            var queue = new BodyEditQueue(4);
            BodyHandle h = SpawnAndApply(queue);
            queue.Remove(h);
            Drain(queue);

            Assert.IsFalse(queue.SetPose(h, float3.zero, quaternion.identity));
            Assert.IsFalse(queue.SetVelocity(h, float3.zero, float3.zero));
            Assert.IsFalse(queue.SetShape(h, new float3(1f), 1f));
            Assert.IsFalse(queue.SetFriction(h, 0.5f));
            Assert.IsFalse(queue.SetGroupMask(h, 1u));
            Assert.IsFalse(queue.SetAppearanceId(h, 1u));
            Assert.IsFalse(queue.SetKinematic(h, true));
            Assert.IsFalse(queue.Remove(h));
            Assert.IsFalse(queue.SetPose(BodyHandle.Invalid, float3.zero, quaternion.identity));
            Assert.AreEqual(0, Drain(queue).Count);
        }

        [Test]
        public void StaleHandle_CannotEditABodyThatReusedItsSlot()
        {
            var queue = new BodyEditQueue(4);
            BodyHandle old = SpawnAndApply(queue);
            queue.Remove(old);
            BodyHandle fresh = queue.Spawn(Box(float3.zero));
            Drain(queue);

            Assert.AreEqual(old.Slot, fresh.Slot);
            Assert.IsFalse(queue.SetFriction(old, 0.9f));
            Assert.IsFalse(queue.Remove(old));
            Assert.IsTrue(queue.IsAlive(fresh));
            Assert.IsTrue(queue.SetFriction(fresh, 0.9f));
        }

        // ---- merging inside one batch -------------------------------------------------------------------

        [Test]
        public void SpawnThenSet_IsOneSpawnWithThePatchedValues()
        {
            var queue = new BodyEditQueue(4);
            BodyHandle h = queue.Spawn(Box(new float3(1f, 1f, 1f)));
            queue.SetPose(h, new float3(9f, 8f, 7f), quaternion.identity);
            queue.SetFriction(h, 0.9f);
            queue.SetShape(h, new float3(2f), 1f);
            queue.SetKinematic(h, true);

            List<EditCommand> commands = Drain(queue);

            Assert.AreEqual(1, commands.Count);
            EditCommand c = commands[0];
            Assert.AreEqual(EditKind.Spawn, c.Kind);
            Assert.AreEqual(EditFields.All, c.Fields);
            AssertEqual(new float4(9f, 8f, 7f, 0f), c.position, "position");
            AssertEqual(new float4(2f, 2f, 2f, 0.9f), c.sizeFriction, "size and friction");
            Assert.AreEqual(8f, c.momentMass.w, 1e-5f);
            Assert.AreEqual((uint)BodyFlags.Kinematic, c.meta.x & (uint)BodyFlags.Kinematic);
        }

        [Test]
        public void SpawnThenRemove_InOneBatch_EmitsNothing()
        {
            var queue = new BodyEditQueue(4);
            BodyHandle h = queue.Spawn(Box(float3.zero));
            Assert.IsTrue(queue.Remove(h));

            Assert.AreEqual(0, Drain(queue).Count);
            Assert.AreEqual(0, queue.AliveCount);
        }

        [Test]
        public void RemoveOfALiveBody_EmitsOneRemove()
        {
            var queue = new BodyEditQueue(4);
            BodyHandle h = SpawnAndApply(queue);

            queue.SetFriction(h, 0.1f);
            queue.Remove(h);
            List<EditCommand> commands = Drain(queue);

            Assert.AreEqual(1, commands.Count);
            Assert.AreEqual(EditKind.Remove, commands[0].Kind);
            Assert.AreEqual(h.Slot, commands[0].Slot);
        }

        [Test]
        public void RemoveThenReuseOfTheSlot_IsOneSpawnThatOverwritesEverything()
        {
            var queue = new BodyEditQueue(4);
            BodyHandle old = SpawnAndApply(queue);

            queue.Remove(old);
            BodyHandle fresh = queue.Spawn(Box(new float3(3f, 3f, 3f)));
            List<EditCommand> commands = Drain(queue);

            Assert.AreEqual(old.Slot, fresh.Slot);
            Assert.AreEqual(1, commands.Count);
            Assert.AreEqual(EditKind.Spawn, commands[0].Kind);
            AssertEqual(new float4(3f, 3f, 3f, 0f), commands[0].position, "position");
        }

        [Test]
        public void RemoveReuseRemove_ThenIsOneRemove()
        {
            var queue = new BodyEditQueue(4);
            BodyHandle old = SpawnAndApply(queue);

            queue.Remove(old);
            BodyHandle fresh = queue.Spawn(Box(float3.zero));
            queue.Remove(fresh);
            List<EditCommand> commands = Drain(queue);

            Assert.AreEqual(1, commands.Count);
            Assert.AreEqual(EditKind.Remove, commands[0].Kind, "the GPU slot was live at the start, so it must be freed");
        }

        [Test]
        public void SpawnRemoveSpawn_OnAFreeSlot_IsOneSpawnWithTheLastData()
        {
            var queue = new BodyEditQueue(4);
            BodyHandle first = queue.Spawn(Box(new float3(1f, 0f, 0f)));
            queue.Remove(first);
            BodyHandle second = queue.Spawn(Box(new float3(2f, 0f, 0f)));
            List<EditCommand> commands = Drain(queue);

            Assert.AreEqual(first.Slot, second.Slot);
            Assert.AreEqual(1, commands.Count);
            Assert.AreEqual(EditKind.Spawn, commands[0].Kind);
            AssertEqual(new float4(2f, 0f, 0f, 0f), commands[0].position, "position");
        }

        [Test]
        public void AfterSpawnRemoveInOneBatch_TheNextBatchSpawnIsAFullSpawn()
        {
            var queue = new BodyEditQueue(4);
            BodyHandle h = queue.Spawn(Box(float3.zero));
            queue.Remove(h);
            Drain(queue);

            queue.Spawn(Box(new float3(5f, 0f, 0f)));
            List<EditCommand> commands = Drain(queue);

            Assert.AreEqual(1, commands.Count);
            Assert.AreEqual(EditKind.Spawn, commands[0].Kind);
        }

        [Test]
        public void Commands_AreEmittedInFirstTouchOrder_OneForEachSlot()
        {
            var queue = new BodyEditQueue(8);
            BodyHandle a = queue.Spawn(Box(float3.zero));
            BodyHandle b = queue.Spawn(Box(float3.zero));
            BodyHandle c = queue.Spawn(Box(float3.zero));
            queue.SetFriction(c, 0.1f);
            queue.SetFriction(a, 0.2f);
            queue.SetFriction(b, 0.3f);

            List<EditCommand> commands = Drain(queue);

            Assert.AreEqual(3, commands.Count);
            Assert.AreEqual(a.Slot, commands[0].Slot);
            Assert.AreEqual(b.Slot, commands[1].Slot);
            Assert.AreEqual(c.Slot, commands[2].Slot);
            var slots = new HashSet<int>();
            foreach (EditCommand command in commands)
                Assert.IsTrue(slots.Add(command.Slot), "two commands for one slot");
        }
    }
}
