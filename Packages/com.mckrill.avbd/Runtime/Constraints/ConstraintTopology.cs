using System;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>
    /// CPU construction of what stages G (per-body constraint lists) and C (graph coloring) will later build on the GPU:
    /// the constraint list of every solved body, a greedy coloring, and the bodies bucketed by color.
    ///
    /// A body is "solved" if the primal stage updates it (not free, not kinematic, mass above zero). Every other body acts as a
    /// fixed anchor: it gets no list and no color, and it creates no edge in the coloring graph.
    /// Both orders match the reference solver so that the colored sweep is the same sequence of updates:
    /// a body's list is newest constraint first, and coloring visits bodies from the highest slot down, giving each the lowest
    /// color not used by a colored neighbor.
    ///
    /// Contacts are constraints too. They are not in the record array on the CPU: the narrow phase writes them on the GPU, in the dense
    /// pair layout of <see cref="ContactLayout"/>, and only the count of each pair is known here. They come first in a body's list
    /// (the reference solver creates manifolds after the joints and springs, so they are the newest forces), pairs in ascending order and
    /// the contacts of a pair in order, which is also the order in which the reference solver's first step stamps them.
    /// </summary>
    internal sealed class ConstraintTopology
    {
        /// <summary>Per body, offset and count in <see cref="List"/> (rule: x = offset, y = count). Length = body count.</summary>
        public BodyRangeRecord[] Ranges = new BodyRangeRecord[0];

        /// <summary>Constraint indices, grouped by body, newest first inside a group. Valid up to <see cref="ListCount"/>.</summary>
        public uint[] List = new uint[0];

        /// <summary>For every entry of <see cref="List"/>, the other body of the constraint (AVBD_INVALID for the world).</summary>
        public uint[] ListOther = new uint[0];

        public int ListCount;

        /// <summary>Solved body slots, grouped by color, highest slot first inside a group. Valid up to <see cref="ColorBodyCount"/>.</summary>
        public uint[] ColorBodies = new uint[0];

        public int ColorBodyCount;

        /// <summary>Start of each color in <see cref="ColorBodies"/>; one extra entry at the end holds the total. Valid up to ColorCount + 1.</summary>
        public int[] ColorOffsets = new int[1];

        public int ColorCount;

        /// <summary>Color of each body, or -1 for a body that is not solved.</summary>
        public int[] ColorOfBody = new int[0];

        public int ColorSize(int color) => ColorOffsets[color + 1] - ColorOffsets[color];

        uint[] contactA = new uint[0];
        uint[] contactB = new uint[0];
        uint[] contactId = new uint[0];
        int contactCount;

        /// <param name="records">Constraint records, index = constraint id.</param>
        /// <param name="solved">One flag per body slot below <paramref name="bodyCount"/>.</param>
        /// <param name="buildColoring">False builds only the lists; the coloring is then left to stage C on the GPU and the color fields stay empty.</param>
        /// <param name="pairCounts">Contacts of each pair of the first <paramref name="contactBodies"/> slots, indexed as <see cref="ContactLayout.PairIndex"/>, or null for no contacts.</param>
        /// <param name="contactBodies">Number of slots the pair counts cover (at most <paramref name="bodyCount"/>).</param>
        /// <param name="contactBase">Constraint id of the first contact record.</param>
        public void Build(ConstraintRecord[] records, int recordCount, bool[] solved, int bodyCount, bool buildColoring = true,
            uint[] pairCounts = null, int contactBodies = 0, int contactBase = 0)
        {
            if (records == null) throw new ArgumentNullException(nameof(records));
            if (solved == null) throw new ArgumentNullException(nameof(solved));
            if (recordCount < 0 || recordCount > records.Length) throw new ArgumentOutOfRangeException(nameof(recordCount));
            if (bodyCount < 0 || bodyCount > solved.Length) throw new ArgumentOutOfRangeException(nameof(bodyCount));
            if (contactBodies < 0 || contactBodies > bodyCount) throw new ArgumentOutOfRangeException(nameof(contactBodies));
            if (pairCounts != null && pairCounts.Length < ContactLayout.PairCount(contactBodies))
                throw new ArgumentException("pairCounts has fewer entries than there are pairs of contactBodies bodies.", nameof(pairCounts));

            CollectContacts(pairCounts, contactBodies, contactBase, solved);

            BuildFromCollected(records, recordCount, solved, bodyCount, buildColoring);
        }

        /// <summary>
        /// Like <see cref="Build"/>, with the contacts of the manifold pool as the GPU lists them: slot s holds <paramref name="slotCounts"/>[s] contacts of
        /// the pair (<paramref name="slotBodies"/>[s].x = body A, .y = body B; AVBD_INVALID = free slot). Every manifold is one list entry,
        /// <see cref="AvbdConstraintLists.ContactFlag"/> | slot, however many contacts it has. The entries are in pair order (body A, then body B,
        /// ascending), whatever slots the pairs occupy, so the lists do not depend on the order in which the GPU handed out the slots.
        /// </summary>
        public void BuildWithManifolds(ConstraintRecord[] records, int recordCount, bool[] solved, int bodyCount, bool buildColoring,
            uint[] slotCounts, uint4[] slotBodies, int slotCount)
        {
            if (records == null) throw new ArgumentNullException(nameof(records));
            if (solved == null) throw new ArgumentNullException(nameof(solved));
            if (slotCounts == null) throw new ArgumentNullException(nameof(slotCounts));
            if (slotBodies == null) throw new ArgumentNullException(nameof(slotBodies));
            if (recordCount < 0 || recordCount > records.Length) throw new ArgumentOutOfRangeException(nameof(recordCount));
            if (bodyCount < 0 || bodyCount > solved.Length) throw new ArgumentOutOfRangeException(nameof(bodyCount));
            if (slotCount < 0 || slotCount > slotCounts.Length || slotCount > slotBodies.Length) throw new ArgumentOutOfRangeException(nameof(slotCount));

            CollectManifoldContacts(slotCounts, slotBodies, slotCount, solved, bodyCount);
            BuildFromCollected(records, recordCount, solved, bodyCount, buildColoring);
        }

        void BuildFromCollected(ConstraintRecord[] records, int recordCount, bool[] solved, int bodyCount, bool buildColoring)
        {
            // 1. List sizes. Constraints on bodies that are not solved are left out: nothing reads them.
            var counts = new int[bodyCount];
            var contacts = new int[bodyCount]; // entries that are contacts: the first ones of each list
            for (int i = 0; i < contactCount; i++)
            {
                if (solved[contactA[i]]) { counts[contactA[i]]++; contacts[contactA[i]]++; }
                if (solved[contactB[i]]) { counts[contactB[i]]++; contacts[contactB[i]]++; }
            }
            for (int i = 0; i < recordCount; i++)
            {
                if (IsBroken(records[i]))
                    continue;
                uint a = records[i].ident.x;
                uint b = records[i].ident.y;
                if (a != AvbdConst.Invalid)
                {
                    if (a >= (uint)bodyCount) throw new InvalidOperationException("Constraint " + i + " refers to body slot " + a + ", which is not a used slot.");
                    if (solved[a]) counts[a]++;
                }
                if (b >= (uint)bodyCount) throw new InvalidOperationException("Constraint " + i + " refers to body slot " + b + ", which is not a used slot.");
                if (solved[b]) counts[b]++;
            }

            if (Ranges.Length != bodyCount)
                Ranges = new BodyRangeRecord[bodyCount];
            int total = 0;
            var cursor = new int[bodyCount];
            for (int s = 0; s < bodyCount; s++)
            {
                Ranges[s] = new BodyRangeRecord { range = new uint4((uint)total, (uint)counts[s], (uint)contacts[s], 0u) };
                cursor[s] = total;
                total += counts[s];
            }

            // 2. Fill the lists, newest constraint first: the contacts, then the joints and springs from the last one added.
            List = new uint[total];
            ListOther = new uint[total];
            ListCount = total;
            for (int i = 0; i < contactCount; i++)
            {
                uint a = contactA[i];
                uint b = contactB[i];
                if (solved[a]) { ListOther[cursor[a]] = b; List[cursor[a]++] = contactId[i]; }
                if (solved[b]) { ListOther[cursor[b]] = a; List[cursor[b]++] = contactId[i]; }
            }
            for (int i = recordCount - 1; i >= 0; i--)
            {
                if (IsBroken(records[i]))
                    continue;
                uint a = records[i].ident.x;
                uint b = records[i].ident.y;
                if (a != AvbdConst.Invalid && solved[a]) { ListOther[cursor[a]] = b; List[cursor[a]++] = (uint)i; }
                if (solved[b]) { ListOther[cursor[b]] = a; List[cursor[b]++] = (uint)i; }
            }

            ColorOfBody = new int[bodyCount];
            for (int s = 0; s < bodyCount; s++)
                ColorOfBody[s] = -1;
            ColorCount = 0;
            ColorBodyCount = 0;
            ColorOffsets = new int[1];
            ColorBodies = new uint[0];
            if (!buildColoring)
                return;

            // 3. Greedy first-fit coloring, highest slot first.
            var stamp = new int[bodyCount + 2]; // stamp[c] == token: color c is used by a neighbor of the body being colored
            int token = 0;
            int colors = 0;
            for (int s = bodyCount - 1; s >= 0; s--)
            {
                if (!solved[s])
                    continue;

                token++;
                int offset = (int)Ranges[s].range.x;
                int count = (int)Ranges[s].range.y;
                for (int e = 0; e < count; e++)
                {
                    uint other = ListOther[offset + e];
                    if (other == AvbdConst.Invalid || !solved[other] || ColorOfBody[other] < 0)
                        continue;
                    stamp[ColorOfBody[other]] = token;
                }

                int color = 0;
                while (stamp[color] == token)
                    color++;
                ColorOfBody[s] = color;
                if (color + 1 > colors)
                    colors = color + 1;
            }

            // 4. Bucket the bodies by color, keeping the visit order inside a color.
            ColorCount = colors;
            ColorOffsets = new int[colors + 1];
            for (int s = 0; s < bodyCount; s++)
                if (ColorOfBody[s] >= 0)
                    ColorOffsets[ColorOfBody[s] + 1]++;
            for (int c = 0; c < colors; c++)
                ColorOffsets[c + 1] += ColorOffsets[c];

            ColorBodyCount = ColorOffsets[colors];
            ColorBodies = new uint[ColorBodyCount];
            var next = new int[colors];
            for (int c = 0; c < colors; c++)
                next[c] = ColorOffsets[c];
            for (int s = bodyCount - 1; s >= 0; s--)
                if (ColorOfBody[s] >= 0)
                    ColorBodies[next[ColorOfBody[s]]++] = (uint)s;
        }

        // Lists the used manifold slots, in pair order, as (body A, body B, entry), one entry per manifold. Slots of two bodies that are not solved are
        // skipped (the narrow phase writes none).
        void CollectManifoldContacts(uint[] slotCounts, uint4[] slotBodies, int slotCount, bool[] solved, int bodyCount)
        {
            contactCount = 0;

            int used = 0;
            for (int s = 0; s < slotCount; s++)
                if (slotCounts[s] != 0u && slotBodies[s].x != AvbdConst.Invalid)
                    used++;

            var keys = new ulong[used];
            var slots = new int[used];
            int n = 0;
            for (int s = 0; s < slotCount; s++)
            {
                if (slotCounts[s] == 0u || slotBodies[s].x == AvbdConst.Invalid)
                    continue;
                uint a = slotBodies[s].x;
                uint b = slotBodies[s].y;
                if (a >= (uint)bodyCount || b >= (uint)bodyCount)
                    throw new InvalidOperationException("Manifold slot " + s + " refers to body slots " + a + " and " + b + ", and only " + bodyCount + " are in use.");
                if (slotCounts[s] > ContactLayout.MaxContactsPerPair)
                    throw new InvalidOperationException("Manifold slot " + s + " reports " + slotCounts[s] + " contacts; the most a manifold can have is " + ContactLayout.MaxContactsPerPair + ".");
                keys[n] = ((ulong)a << 32) | b;
                slots[n] = s;
                n++;
            }
            Array.Sort(keys, slots);

            for (int i = 0; i < n; i++)
            {
                int slot = slots[i];
                uint a = slotBodies[slot].x;
                uint b = slotBodies[slot].y;
                if (!solved[a] && !solved[b])
                    continue;

                if (contactCount + 1 > contactA.Length)
                {
                    int size = Math.Max(64, contactA.Length * 2);
                    Array.Resize(ref contactA, size);
                    Array.Resize(ref contactB, size);
                    Array.Resize(ref contactId, size);
                }
                contactA[contactCount] = a;
                contactB[contactCount] = b;
                contactId[contactCount] = AvbdConstraintLists.ContactFlag | (uint)slot;
                contactCount++;
            }
        }
        /// <summary>A fractured joint does nothing any more: the reference solver removes it, and the lists and the coloring leave it out.</summary>
        internal static bool IsBroken(in ConstraintRecord r) => (r.ident.z & (uint)ConstraintFlags.Broken) != 0u;

        // Lists the contacts of the pairs that have any, in pair order, as (body A, body B, constraint id). Pairs of two bodies that are not solved are
        // skipped (the narrow phase writes none).
        void CollectContacts(uint[] pairCounts, int contactBodies, int contactBase, bool[] solved)
        {
            contactCount = 0;
            if (pairCounts == null)
                return;

            long pair = 0;
            for (int hi = 1; hi < contactBodies; hi++)
            {
                for (int lo = 0; lo < hi; lo++, pair++)
                {
                    uint n = pairCounts[pair];
                    if (n == 0u || (!solved[hi] && !solved[lo]))
                        continue;
                    if (n > ContactLayout.MaxContactsPerPair)
                        throw new InvalidOperationException("Pair " + pair + " reports " + n + " contacts; the most a pair can have is " + ContactLayout.MaxContactsPerPair + ".");

                    if (contactCount + (int)n > contactA.Length)
                    {
                        int size = Math.Max(64, Math.Max(contactA.Length * 2, contactCount + (int)n));
                        Array.Resize(ref contactA, size);
                        Array.Resize(ref contactB, size);
                        Array.Resize(ref contactId, size);
                    }
                    for (int k = 0; k < (int)n; k++)
                    {
                        contactA[contactCount] = (uint)hi;
                        contactB[contactCount] = (uint)lo;
                        contactId[contactCount] = (uint)(contactBase + pair * ContactLayout.MaxContactsPerPair + k);
                        contactCount++;
                    }
                }
            }
        }
    }
}
