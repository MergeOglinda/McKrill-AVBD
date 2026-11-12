using NUnit.Framework;

namespace McKrill.Avbd.Tests
{
    public class BodySlotAllocatorTests
    {
        [Test]
        public void DefaultHandle_IsInvalid()
        {
            BodyHandle handle = default;
            Assert.IsFalse(handle.IsValid);
            Assert.AreEqual(BodyHandle.Invalid, handle);
            Assert.IsFalse(new BodySlotAllocator(4).IsAlive(handle));
        }

        [Test]
        public void FreshSlots_AreHandedOutInOrder()
        {
            var a = new BodySlotAllocator(4);
            for (int i = 0; i < 3; i++)
            {
                Assert.IsTrue(a.TryAllocate(out BodyHandle h));
                Assert.AreEqual(i, h.Slot);
                Assert.IsTrue(a.IsAlive(h));
            }
            Assert.AreEqual(3, a.AliveCount);
            Assert.AreEqual(3, a.HighWaterMark);
        }

        [Test]
        public void FreedSlots_AreReusedMostRecentFirst()
        {
            var a = new BodySlotAllocator(4);
            a.TryAllocate(out BodyHandle h0);
            a.TryAllocate(out BodyHandle h1);
            a.TryAllocate(out BodyHandle h2);

            Assert.IsTrue(a.Free(h0));
            Assert.IsTrue(a.Free(h2));

            a.TryAllocate(out BodyHandle n1);
            a.TryAllocate(out BodyHandle n2);
            Assert.AreEqual(2, n1.Slot);
            Assert.AreEqual(0, n2.Slot);
            Assert.AreEqual(3, a.HighWaterMark, "reuse must not move the high-water mark");
            Assert.IsTrue(a.IsAlive(h1));
        }

        [Test]
        public void ReusedSlot_MakesTheOldHandleStale()
        {
            var a = new BodySlotAllocator(2);
            a.TryAllocate(out BodyHandle old);
            Assert.IsTrue(a.Free(old));
            Assert.IsFalse(a.IsAlive(old));

            a.TryAllocate(out BodyHandle fresh);
            Assert.AreEqual(old.Slot, fresh.Slot);
            Assert.AreNotEqual(old, fresh);
            Assert.IsFalse(a.IsAlive(old));
            Assert.IsTrue(a.IsAlive(fresh));
            Assert.IsFalse(a.Free(old), "a stale handle must not free the new body");
            Assert.IsTrue(a.IsAlive(fresh));
        }

        [Test]
        public void Free_RejectsDoubleFreeAndInvalidHandles()
        {
            var a = new BodySlotAllocator(2);
            a.TryAllocate(out BodyHandle h);
            Assert.IsTrue(a.Free(h));
            Assert.IsFalse(a.Free(h));
            Assert.IsFalse(a.Free(BodyHandle.Invalid));
            Assert.AreEqual(0, a.AliveCount);
        }

        [Test]
        public void TryAllocate_FailsWhenFull_AndRecoversAfterFree()
        {
            var a = new BodySlotAllocator(2);
            a.TryAllocate(out BodyHandle h0);
            a.TryAllocate(out _);

            Assert.IsFalse(a.TryAllocate(out BodyHandle none));
            Assert.IsFalse(none.IsValid);
            Assert.AreEqual(2, a.AliveCount);

            a.Free(h0);
            Assert.IsTrue(a.TryAllocate(out BodyHandle again));
            Assert.AreEqual(0, again.Slot);
        }

        [Test]
        public void IsSlotAlive_TracksTheSlot()
        {
            var a = new BodySlotAllocator(2);
            Assert.IsFalse(a.IsSlotAlive(0));
            Assert.IsFalse(a.IsSlotAlive(-1));
            Assert.IsFalse(a.IsSlotAlive(5));
            a.TryAllocate(out BodyHandle h);
            Assert.IsTrue(a.IsSlotAlive(0));
            a.Free(h);
            Assert.IsFalse(a.IsSlotAlive(0));
        }

        [Test]
        public void HandleEquality_UsesSlotAndGeneration()
        {
            var a = new BodySlotAllocator(2);
            a.TryAllocate(out BodyHandle h);
            BodyHandle copy = h;
            Assert.IsTrue(h == copy);
            Assert.AreEqual(h.GetHashCode(), copy.GetHashCode());
            Assert.IsTrue(h != BodyHandle.Invalid);
        }

        [Test]
        public void Constructor_RejectsBadCapacity()
        {
            Assert.Throws<System.ArgumentOutOfRangeException>(() => new BodySlotAllocator(0));
        }
    }
}
