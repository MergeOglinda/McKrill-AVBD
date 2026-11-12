using NUnit.Framework;
using Unity.Mathematics;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Behavior of stage E as defined by the CPU oracle (<see cref="CpuBodyArrays"/>), which the GPU kernel is
    /// compared against in <see cref="BodyEditsGpuTests"/>. These tests need no GPU.
    /// </summary>
    public class BodyEditsOracleTests
    {
        const int Capacity = 8;

        static void Run(CpuBodyArrays arrays, BodyEditQueue queue)
        {
            var commands = new System.Collections.Generic.List<EditCommand>();
            queue.Drain(commands);
            arrays.Apply(commands);
        }

        static bool Same(float4 a, float4 b) => math.all(a == b);

        // ---- the free state ---------------------------------------------------------------------------

        [Test]
        public void NewArrays_AreAllFree()
        {
            var arrays = new CpuBodyArrays(Capacity);
            for (int slot = 0; slot < Capacity; slot++)
            {
                Assert.IsTrue(arrays.IsFree(slot));
                Assert.AreEqual(AvbdConst.Invalid, arrays.Meta[slot].ident.z, "appearance id of a free slot");
                Assert.AreEqual(AvbdConst.Invalid, arrays.Meta[slot].state.x, "graph color of a free slot");
                Assert.AreEqual(0f, arrays.Mass[slot].momentMass.w);
                Assert.IsTrue(Same(new float4(0f, 0f, 0f, 1f), arrays.State[slot].rotation));
                Assert.IsTrue(Same(new float4(0f, 0f, 0f, 1f), arrays.Inertial[slot].rotation));
            }
        }

        // ---- spawn ------------------------------------------------------------------------------------

        [Test]
        public void Spawn_WritesEveryPersistentField()
        {
            var arrays = new CpuBodyArrays(Capacity);
            var queue = new BodyEditQueue(Capacity);

            BodyDescriptor d = BodyDescriptor.Box(new float3(1f, 2f, 3f), 2f, new float3(4f, 5f, 6f));
            d.linearVelocity = new float3(1f, 1f, 1f);
            d.groupMask = 3u;
            d.appearanceId = 7u;
            BodyHandle h = queue.Spawn(d);
            Run(arrays, queue);

            int s = h.Slot;
            Assert.IsFalse(arrays.IsFree(s));
            Assert.IsTrue(Same(new float4(4f, 5f, 6f, 0f), arrays.State[s].position));
            Assert.IsTrue(Same(arrays.State[s].position, arrays.State[s].startPosition), "spawn sets the step-start pose");
            Assert.IsTrue(Same(arrays.State[s].rotation, arrays.State[s].startRotation));
            Assert.IsTrue(Same(arrays.Motion[s].linearVel, arrays.Motion[s].prevLinearVel), "spawn sets the previous velocity");
            Assert.AreEqual(12f, arrays.Mass[s].momentMass.w, 1e-5f);
            Assert.IsTrue(Same(new float4(1f, 2f, 3f, 0.5f), arrays.Shape[s].sizeFriction));
            Assert.AreEqual(3u, arrays.Meta[s].ident.y);
            Assert.AreEqual(7u, arrays.Meta[s].ident.z);
            Assert.AreEqual(AvbdConst.Invalid, arrays.Meta[s].state.x, "a new body has no color");
            Assert.AreEqual(0u, arrays.Meta[s].state.y, "sleep counter starts at zero");
            Assert.AreEqual(0u, arrays.Range[s].range.y, "no constraints yet");
        }

        [Test]
        public void Spawn_LeavesTheInertialTargetAlone()
        {
            var arrays = new CpuBodyArrays(Capacity);
            var queue = new BodyEditQueue(Capacity);
            BodyHandle h = queue.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(9f, 9f, 9f)));
            Run(arrays, queue);

            Assert.IsTrue(Same(float4.zero, arrays.Inertial[h.Slot].position), "only stage P writes the inertial target");
        }

        // ---- remove -----------------------------------------------------------------------------------

        [Test]
        public void Remove_FreesTheSlot_AndKeepsTheShapeKind()
        {
            var arrays = new CpuBodyArrays(Capacity);
            var queue = new BodyEditQueue(Capacity);
            BodyDescriptor d = BodyDescriptor.Box(new float3(1f), 1f, float3.zero);
            d.kinematic = true;
            d.appearanceId = 3u;
            BodyHandle h = queue.Spawn(d);
            Run(arrays, queue);

            queue.Remove(h);
            Run(arrays, queue);

            int s = h.Slot;
            Assert.IsTrue(arrays.IsFree(s));
            Assert.AreEqual(0f, arrays.Mass[s].momentMass.w, "a free slot must never look dynamic");
            Assert.AreEqual(AvbdConst.Invalid, arrays.Meta[s].ident.z);
            Assert.AreEqual(AvbdConst.Invalid, arrays.Meta[s].state.x);
            Assert.AreEqual(0u, arrays.Meta[s].ident.x & (uint)BodyFlags.Kinematic, "remove clears the other flags");
            Assert.AreEqual(ShapeKind.Box, BodyFlagLayout.KindOf(arrays.Meta[s].ident.x));
        }

        // ---- set --------------------------------------------------------------------------------------

        [Test]
        public void SetPose_ResetsTheStepStartPose()
        {
            var arrays = new CpuBodyArrays(Capacity);
            var queue = new BodyEditQueue(Capacity);
            BodyHandle h = queue.Spawn(BodyDescriptor.Box(new float3(1f), 1f, float3.zero));
            Run(arrays, queue);

            queue.SetPose(h, new float3(5f, 6f, 7f), quaternion.identity);
            Run(arrays, queue);

            Assert.IsTrue(Same(new float4(5f, 6f, 7f, 0f), arrays.State[h.Slot].position));
            Assert.IsTrue(Same(arrays.State[h.Slot].position, arrays.State[h.Slot].startPosition));
        }

        [Test]
        public void SetVelocity_ResetsThePreviousVelocity()
        {
            var arrays = new CpuBodyArrays(Capacity);
            var queue = new BodyEditQueue(Capacity);
            BodyHandle h = queue.Spawn(BodyDescriptor.Box(new float3(1f), 1f, float3.zero));
            Run(arrays, queue);

            queue.SetVelocity(h, new float3(3f, 0f, 0f), new float3(0f, 4f, 0f));
            Run(arrays, queue);

            int s = h.Slot;
            Assert.IsTrue(Same(new float4(3f, 0f, 0f, 0f), arrays.Motion[s].linearVel));
            Assert.IsTrue(Same(new float4(0f, 4f, 0f, 0f), arrays.Motion[s].angularVel));
            Assert.IsTrue(Same(arrays.Motion[s].linearVel, arrays.Motion[s].prevLinearVel));
        }

        [Test]
        public void SetShape_KeepsFriction_AndSetFriction_KeepsSize()
        {
            var arrays = new CpuBodyArrays(Capacity);
            var queue = new BodyEditQueue(Capacity);
            BodyDescriptor d = BodyDescriptor.Box(new float3(1f, 2f, 3f), 1f, float3.zero);
            d.friction = 0.3f;
            BodyHandle h = queue.Spawn(d);
            Run(arrays, queue);

            queue.SetShape(h, new float3(4f, 5f, 6f), 1f);
            Run(arrays, queue);
            Assert.IsTrue(Same(new float4(4f, 5f, 6f, 0.3f), arrays.Shape[h.Slot].sizeFriction));
            Assert.AreEqual(120f, arrays.Mass[h.Slot].momentMass.w, 1e-3f);

            queue.SetFriction(h, 0.9f);
            Run(arrays, queue);
            Assert.IsTrue(Same(new float4(4f, 5f, 6f, 0.9f), arrays.Shape[h.Slot].sizeFriction));
            Assert.AreEqual(120f, arrays.Mass[h.Slot].momentMass.w, 1e-3f, "friction must not touch the mass");
        }

        [Test]
        public void SetKinematic_ChangesOnlyThatBit()
        {
            var arrays = new CpuBodyArrays(Capacity);
            var queue = new BodyEditQueue(Capacity);
            BodyDescriptor d = BodyDescriptor.Box(new float3(1f), 1f, float3.zero);
            d.groupMask = 0x11u;
            d.appearanceId = 5u;
            BodyHandle h = queue.Spawn(d);
            Run(arrays, queue);

            queue.SetKinematic(h, true);
            Run(arrays, queue);
            uint on = arrays.Meta[h.Slot].ident.x;
            Assert.AreNotEqual(0u, on & (uint)BodyFlags.Kinematic);
            Assert.AreEqual(0u, on & (uint)BodyFlags.Free);
            Assert.AreEqual(0x11u, arrays.Meta[h.Slot].ident.y);
            Assert.AreEqual(5u, arrays.Meta[h.Slot].ident.z);

            queue.SetKinematic(h, false);
            Run(arrays, queue);
            Assert.AreEqual(0u, arrays.Meta[h.Slot].ident.x & (uint)BodyFlags.Kinematic);
        }

        [Test]
        public void SetGroupMaskAndAppearance_TouchOnlyTheirWords()
        {
            var arrays = new CpuBodyArrays(Capacity);
            var queue = new BodyEditQueue(Capacity);
            BodyHandle h = queue.Spawn(BodyDescriptor.Box(new float3(1f), 1f, float3.zero));
            Run(arrays, queue);
            uint flagsBefore = arrays.Meta[h.Slot].ident.x;

            queue.SetGroupMask(h, 0x77u);
            queue.SetAppearanceId(h, 12u);
            Run(arrays, queue);

            Assert.AreEqual(0x77u, arrays.Meta[h.Slot].ident.y);
            Assert.AreEqual(12u, arrays.Meta[h.Slot].ident.z);
            Assert.AreEqual(flagsBefore, arrays.Meta[h.Slot].ident.x);
        }

        [Test]
        public void ACommandForASlotOutsideTheArrays_IsIgnored()
        {
            var arrays = new CpuBodyArrays(2);
            var command = new EditCommand { header = new uint4((uint)EditKind.Spawn, 5u, (uint)EditFields.All, 0u) };
            Assert.DoesNotThrow(() => arrays.Apply(command));
        }

        // ---- mass properties agree with the CPU reference solver ---------------------------------------

        [Test]
        public void BoxMassProperties_MatchTheReferenceSolversBody()
        {
            float3[] sizes = { new float3(1f), new float3(1f, 0.5f, 0.5f), new float3(2f, 3f, 0.25f), new float3(100f, 100f, 1f) };
            float[] densities = { 0f, 1f, 2.5f };

            foreach (float3 size in sizes)
            {
                foreach (float density in densities)
                {
                    var body = new AvbdBody(size, density, 0.5f, float3.zero, float3.zero);
                    float4 mine = BoxMassProperties.Compute(size, density);

                    Assert.AreEqual(body.Mass, mine.w, "mass of " + size + " density " + density);
                    Assert.AreEqual(body.Moment.x, mine.x, "moment x of " + size);
                    Assert.AreEqual(body.Moment.y, mine.y, "moment y of " + size);
                    Assert.AreEqual(body.Moment.z, mine.z, "moment z of " + size);
                }
            }
        }
    }
}
