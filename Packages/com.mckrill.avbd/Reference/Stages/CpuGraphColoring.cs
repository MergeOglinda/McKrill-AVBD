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
    }
}
