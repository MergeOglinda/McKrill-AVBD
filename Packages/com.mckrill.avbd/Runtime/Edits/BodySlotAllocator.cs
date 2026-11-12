namespace McKrill.Avbd
{
    /// <summary>
    /// Hands out body slots (rows of the GPU arrays) on the CPU. Pure bookkeeping, no GPU access.
    /// A freed slot is reused before a never-used one (most recently freed first), so live bodies stay
    /// packed near the start of the arrays. Each free bumps the slot's generation, which makes every
    /// older handle to that slot stale.
    /// </summary>
    internal sealed class BodySlotAllocator
    {
        readonly uint[] generations;
        readonly bool[] alive;
        readonly int[] freeSlots;
        int freeCount;
        int nextFresh;
        int aliveCount;

        public BodySlotAllocator(int capacity)
        {
            BodyBufferCapacity.Validate(capacity);
            generations = new uint[capacity];
            alive = new bool[capacity];
            freeSlots = new int[capacity];
        }

        public int Capacity => generations.Length;

        public int AliveCount => aliveCount;

        /// <summary>One past the highest slot ever handed out. Kernels that scan all slots need not go beyond it.</summary>
        public int HighWaterMark => nextFresh;

        public bool TryAllocate(out BodyHandle handle)
        {
            int slot;
            if (freeCount > 0)
                slot = freeSlots[--freeCount];
            else if (nextFresh < generations.Length)
                slot = nextFresh++;
            else
            {
                handle = default;
                return false;
            }

            alive[slot] = true;
            aliveCount++;
            handle = new BodyHandle(slot, generations[slot]);
            return true;
        }

        /// <summary>Frees the slot of a live handle. Returns false for an invalid or stale handle.</summary>
        public bool Free(BodyHandle handle)
        {
            if (!IsAlive(handle))
                return false;

            int slot = handle.Slot;
            alive[slot] = false;
            unchecked { generations[slot]++; }
            freeSlots[freeCount++] = slot;
            aliveCount--;
            return true;
        }

        public bool IsAlive(BodyHandle handle)
        {
            if (!handle.IsValid)
                return false;
            int slot = handle.Slot;
            return slot < generations.Length && alive[slot] && generations[slot] == handle.Generation;
        }

        public bool IsSlotAlive(int slot) => (uint)slot < (uint)generations.Length && alive[slot];
    }
}
