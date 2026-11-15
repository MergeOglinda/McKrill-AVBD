using System.Collections.Generic;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>
    /// The manifold pool of stage N read back from the GPU: which pair holds which slot and how many contacts each slot has. For tests, tools
    /// and the CPU build of the constraint lists. See <see cref="ContactLayout"/>.
    /// </summary>
    internal sealed class ContactSnapshot
    {
        /// <summary>Contacts per slot; zero for a free slot.</summary>
        public readonly uint[] Counts;

        /// <summary>Per slot: x = body A (the higher slot), y = body B; both AVBD_INVALID for a free slot.</summary>
        public readonly uint4[] Bodies;

        /// <summary>Slots on the free stack.</summary>
        public readonly int FreeSlots;

        /// <summary>Candidate pairs dropped because no slot was free, since the buffers were made.</summary>
        public readonly int Overflow;

        /// <summary>The highest slot ever taken + 1 (<see cref="ContactLayout.PoolHighWaterWord"/>): every used slot is below it.</summary>
        public readonly int HighWater;

        readonly Dictionary<ulong, int> slotOfPair = new Dictionary<ulong, int>();

        public ContactSnapshot(uint[] counts, uint4[] bodies, int freeSlots, int overflow, int highWater)
        {
            Counts = counts;
            Bodies = bodies;
            FreeSlots = freeSlots;
            Overflow = overflow;
            HighWater = highWater;
            for (int s = 0; s < bodies.Length; s++)
                if (bodies[s].x != AvbdConst.Invalid)
                {
                    ulong key = ((ulong)bodies[s].x << 32) | bodies[s].y;
                    if (slotOfPair.ContainsKey(key))
                        DuplicatePairs++;
                    slotOfPair[key] = s;
                }
        }

        /// <summary>Slots that hold a pair.</summary>
        public int UsedSlots => slotOfPair.Count + DuplicatePairs;

        /// <summary>Pairs that appear in more than one slot. Must be zero.</summary>
        public int DuplicatePairs { get; private set; }

        /// <summary>Slots that hold a pair with contacts.</summary>
        public int SlotsWithContacts
        {
            get
            {
                int n = 0;
                for (int s = 0; s < Counts.Length; s++)
                    if (Counts[s] != 0u)
                        n++;
                return n;
            }
        }

        /// <summary>The slot of the pair of two body slots (in either order), or -1 if the pair has none.</summary>
        public int SlotOf(int bodyA, int bodyB)
        {
            int hi = bodyA > bodyB ? bodyA : bodyB;
            int lo = bodyA > bodyB ? bodyB : bodyA;
            return slotOfPair.TryGetValue(((ulong)(uint)hi << 32) | (uint)lo, out int slot) ? slot : -1;
        }

        /// <summary>Contacts of the pair of two body slots; zero if the pair has no slot.</summary>
        public uint CountOf(int bodyA, int bodyB)
        {
            int slot = SlotOf(bodyA, bodyB);
            return slot < 0 ? 0u : Counts[slot];
        }
    }
}
