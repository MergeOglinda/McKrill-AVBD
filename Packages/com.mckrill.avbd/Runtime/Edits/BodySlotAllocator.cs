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
        readonly uint[] freedAt;
        readonly bool[] alive;
        readonly int[] freeSlots;
        int freeCount;
        int nextFresh;
        int aliveCount;
        uint freeSerial;

        public BodySlotAllocator(int capacity)
        {
            BodyBufferCapacity.Validate(capacity);
            generations = new uint[capacity];
            freedAt = new uint[capacity];
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
            unchecked { generations[slot]++; freeSerial++; }
            freedAt[slot] = freeSerial;
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

        /// <summary>Number of frees so far. A free gets the next number, so "was this slot freed after I looked?" is a comparison with the number from then.</summary>
        public uint FreeSerial => freeSerial;

        /// <summary>
        /// The handle of the body in a slot, if it is alive and was not freed after the <paramref name="since"/> (a <see cref="FreeSerial"/> taken earlier).
        /// For an answer that names a slot and was computed when the serial was <paramref name="since"/> (a GPU ray cast): the body it meant is the one that
        /// held the slot then, so a slot freed since, and maybe given to another body, has no answer. Wraps after 2^32 frees, which the callers cannot tell apart from none.
        /// </summary>
        public bool TryGetHandleSince(int slot, uint since, out BodyHandle handle)
        {
            handle = default;
            if ((uint)slot >= (uint)generations.Length || !alive[slot] || freedAt[slot] > since)
                return false;
            handle = new BodyHandle(slot, generations[slot]);
            return true;
        }

        public bool IsSlotAlive(int slot) => (uint)slot < (uint)generations.Length && alive[slot];
    }
}
