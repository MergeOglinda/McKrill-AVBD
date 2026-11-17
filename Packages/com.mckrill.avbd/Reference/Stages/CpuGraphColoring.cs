using System;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle for stage C: the coloring that <c>AvbdColoring.compute</c> must produce. Bodies are visited in falling priority and each
    /// takes the lowest color none of its already colored neighbors has. Jones-Plassmann on the GPU yields exactly this coloring for any
    /// priority that is a strict total order. With <see cref="GraphColoringOrder.SlotOrder"/> it is <see cref="ConstraintTopology"/>'s
    /// (and so the reference solver's) coloring.
    /// </summary>
    internal static class CpuGraphColoring
    {
        /// <summary>Mirrors HashSlot in AvbdColoring.compute.</summary>
        public static uint HashSlot(uint x)
        {
            unchecked
            {
                x ^= x >> 16;
                x *= 0x7feb352du;
                x ^= x >> 15;
                x *= 0x846ca68bu;
                x ^= x >> 16;
                return x;
            }
        }

        /// <summary>True if body <paramref name="other"/> colors before body <paramref name="self"/>. Mirrors ColorsBefore in the shader.</summary>
        public static bool ColorsBefore(uint other, uint self, GraphColoringOrder order)
        {
            if (order == GraphColoringOrder.SlotOrder)
                return other > self;

            uint hashOther = HashSlot(other);
            uint hashSelf = HashSlot(self);
            return hashOther != hashSelf ? hashOther > hashSelf : other > self;
        }

        /// <summary>Color of every body slot below bodyCount, or -1 for a body that is not solved. The lists in <paramref name="topology"/> must be built.</summary>
        public static int[] Color(ConstraintRecord[] records, ConstraintTopology topology, bool[] solved, int bodyCount, GraphColoringOrder order)
        {
            var visit = new System.Collections.Generic.List<uint>();
            for (int s = 0; s < bodyCount; s++)
                if (solved[s])
                    visit.Add((uint)s);
            visit.Sort((a, b) => a == b ? 0 : ColorsBefore(a, b, order) ? -1 : 1);

            var colors = new int[bodyCount];
            for (int s = 0; s < bodyCount; s++)
                colors[s] = -1;

            foreach (uint slot in visit)
            {
                var used = new System.Collections.Generic.HashSet<int>();
                int offset = (int)topology.Ranges[slot].range.x;
                int count = (int)topology.Ranges[slot].range.y;
                for (int e = 0; e < count; e++)
                {
                    ConstraintRecord c = records[topology.List[offset + e]];
                    uint other = c.ident.x == slot ? c.ident.y : c.ident.x;
                    if (other == AvbdConst.Invalid || !solved[other] || colors[other] < 0)
                        continue;
                    used.Add(colors[other]);
                }

                int color = 0;
                while (used.Contains(color))
                    color++;
                colors[slot] = color;
            }
            return colors;
        }

        /// <summary>
        /// The persistent coloring (ColorKeep, ColorMark, ColorRelease, then Jones-Plassmann on what is released): a color below <paramref name="cap"/>
        /// is kept; of two neighbors with one color the one that colors later loses it; the bodies without a color then take, in falling priority, the
        /// lowest color none of their colored neighbors has, or the last color below the cap if there is none. The rounds the GPU runs, and ColorFinish,
        /// only change how fast it gets there, not the result. <paramref name="forced"/> counts the bodies that took the last color for lack of a free one.
        /// </summary>
        public static int[] Persistent(ConstraintRecord[] records, ConstraintTopology topology, bool[] solved, int bodyCount, GraphColoringOrder order,
            int[] previous, int cap, out int forced)
        {
            var colors = new int[bodyCount];
            for (int s = 0; s < bodyCount; s++)
                colors[s] = solved[s] && previous[s] >= 0 && previous[s] < cap ? previous[s] : -1;

            // Mark: reads only the colors that were kept, so the order of the bodies does not matter.
            var release = new bool[bodyCount];
            for (int s = 0; s < bodyCount; s++)
            {
                if (colors[s] < 0)
                    continue;
                foreach (int o in Neighbors(records, topology, solved, (uint)s))
                    if (colors[o] == colors[s] && ColorsBefore((uint)o, (uint)s, order))
                        release[s] = true;
            }
            var pending = new System.Collections.Generic.List<uint>();
            for (int s = 0; s < bodyCount; s++)
            {
                if (release[s])
                    colors[s] = -1;
                if (solved[s] && colors[s] < 0)
                    pending.Add((uint)s);
            }
            pending.Sort((a, b) => a == b ? 0 : ColorsBefore(a, b, order) ? -1 : 1);

            forced = 0;
            foreach (uint slot in pending)
            {
                var used = new System.Collections.Generic.HashSet<int>();
                foreach (int o in Neighbors(records, topology, solved, slot))
                    if (colors[o] >= 0)
                        used.Add(colors[o]);
                int color = 0;
                while (used.Contains(color))
                    color++;
                if (color >= cap)
                {
                    color = cap - 1;
                    forced++;
                }
                colors[slot] = color;
            }
            return colors;
        }

        /// <summary>The solved bodies joined to <paramref name="slot"/> by a constraint (a body may come more than once).</summary>
        public static System.Collections.Generic.IEnumerable<int> Neighbors(ConstraintRecord[] records, ConstraintTopology topology, bool[] solved, uint slot)
        {
            int offset = (int)topology.Ranges[slot].range.x;
            int count = (int)topology.Ranges[slot].range.y;
            for (int e = 0; e < count; e++)
            {
                ConstraintRecord c = records[topology.List[offset + e]];
                uint other = c.ident.x == slot ? c.ident.y : c.ident.x;
                if (other != AvbdConst.Invalid && other != slot && solved[other])
                    yield return (int)other;
            }
        }

        /// <summary>
        /// The persistent coloring of AvbdColoring.compute (ColorKeep, ColorMark, ColorRelease, then rounds and ColorFinish): the bodies whose previous
        /// color is below <paramref name="cap"/> keep it; of two neighbors that share a kept color the one that colors later is released; the released
        /// and the new bodies then take, in falling priority, the lowest color no neighbor has, or the last color below the cap if there is none.
        /// The result does not depend on the number of rounds. <paramref name="previous"/> is -1 for a body without a color. With no previous
        /// colors it is <see cref="Color"/> (as long as the cap is not reached).
        /// </summary>
        public static int[] Persistent(ConstraintRecord[] records, ConstraintTopology topology, bool[] solved, int bodyCount, GraphColoringOrder order,
            int[] previous, int cap, out int forced)
        {
            var colors = new int[bodyCount];
            for (int s = 0; s < bodyCount; s++)
                colors[s] = solved[s] && previous[s] >= 0 && previous[s] < cap ? previous[s] : -1;

            // Mark reads the colors as they were kept, and every body is released by what it reads, never by a release.
            var release = new bool[bodyCount];
            for (int s = 0; s < bodyCount; s++)
            {
                if (colors[s] < 0)
                    continue;
                foreach (int other in Neighbors(records, topology, solved, (uint)s))
                    if (colors[other] == colors[s] && ColorsBefore((uint)other, (uint)s, order))
                        release[s] = true;
            }
            for (int s = 0; s < bodyCount; s++)
                if (release[s])
                    colors[s] = -1;

            var visit = new System.Collections.Generic.List<uint>();
            for (int s = 0; s < bodyCount; s++)
                if (solved[s] && colors[s] < 0)
                    visit.Add((uint)s);
            visit.Sort((a, b) => a == b ? 0 : ColorsBefore(a, b, order) ? -1 : 1);

            forced = 0;
            foreach (uint slot in visit)
            {
                var used = new System.Collections.Generic.HashSet<int>();
                foreach (int other in Neighbors(records, topology, solved, slot))
                    if (colors[other] >= 0)
                        used.Add(colors[other]);

                int color = 0;
                while (used.Contains(color))
                    color++;
                if (color >= cap)
                {
                    color = cap - 1;
                    forced++;
                }
                colors[slot] = color;
            }
            return colors;
        }

        static System.Collections.Generic.IEnumerable<int> Neighbors(ConstraintRecord[] records, ConstraintTopology topology, bool[] solved, uint slot)
        {
            int offset = (int)topology.Ranges[slot].range.x;
            int count = (int)topology.Ranges[slot].range.y;
            for (int e = 0; e < count; e++)
            {
                ConstraintRecord c = records[topology.List[offset + e]];
                uint other = c.ident.x == slot ? c.ident.y : c.ident.x;
                if (other != AvbdConst.Invalid && other != slot && solved[other])
                    yield return (int)other;
            }
        }
    }
}
