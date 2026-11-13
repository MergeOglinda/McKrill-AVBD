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
    /// </summary>
    internal sealed class ConstraintTopology
    {
        /// <summary>Per body, offset and count in <see cref="List"/> (rule: x = offset, y = count). Length = body count.</summary>
        public BodyRangeRecord[] Ranges = new BodyRangeRecord[0];

        /// <summary>Constraint indices, grouped by body, newest first inside a group. Valid up to <see cref="ListCount"/>.</summary>
        public uint[] List = new uint[0];

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

        /// <param name="records">Constraint records, index = constraint id.</param>
        /// <param name="solved">One flag per body slot below <paramref name="bodyCount"/>.</param>
        /// <param name="buildColoring">False builds only the lists; the coloring is then left to stage C on the GPU and the color fields stay empty.</param>
        public void Build(ConstraintRecord[] records, int recordCount, bool[] solved, int bodyCount, bool buildColoring = true)
        {
            if (records == null) throw new ArgumentNullException(nameof(records));
            if (solved == null) throw new ArgumentNullException(nameof(solved));
            if (recordCount < 0 || recordCount > records.Length) throw new ArgumentOutOfRangeException(nameof(recordCount));
            if (bodyCount < 0 || bodyCount > solved.Length) throw new ArgumentOutOfRangeException(nameof(bodyCount));

            // 1. List sizes. Constraints on bodies that are not solved are left out: nothing reads them.
            var counts = new int[bodyCount];
            for (int i = 0; i < recordCount; i++)
            {
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
                Ranges[s] = new BodyRangeRecord { range = new uint4((uint)total, (uint)counts[s], 0u, 0u) };
                cursor[s] = total;
                total += counts[s];
            }

            // 2. Fill the lists, newest constraint first.
            List = new uint[total];
            ListCount = total;
            for (int i = recordCount - 1; i >= 0; i--)
            {
                uint a = records[i].ident.x;
                uint b = records[i].ident.y;
                if (a != AvbdConst.Invalid && solved[a]) List[cursor[a]++] = (uint)i;
                if (solved[b]) List[cursor[b]++] = (uint)i;
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
                    ConstraintRecord c = records[List[offset + e]];
                    uint other = c.ident.x == (uint)s ? c.ident.y : c.ident.x;
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
    }
}
