using System;
using System.Collections.Generic;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>
    /// Z-curve (Morton) order of points, for placing bodies in the GPU arrays so that bodies that are close in space are close in memory.
    /// The solver gathers the pose of a neighbor by slot (partner bodies of contacts and joints, grid cells of the broadphase); a pile built layer by
    /// layer or ring by ring has neighbors thousands of slots apart, and a pile in Z-curve order has most of them within a few cache lines.
    /// This is the "initial Morton ordering of slots when a scene is spawned in one batch" of <c>Docs/design/three-avbd-comparison.md</c> section 7:
    /// it is applied once, when the bodies are spawned (<see cref="AvbdBodyBuffers.SpawnBatch"/>), and nothing is remapped while the simulation runs.
    /// Pure C#, no GPU.
    /// </summary>
    public static class SpatialOrder
    {
        /// <summary>Bits per axis of a key: 3 * 21 = 63 bits.</summary>
        public const int BitsPerAxis = 21;

        const uint MaxCoordinate = (1u << BitsPerAxis) - 1u;

        /// <summary>
        /// Morton keys of the points. The points are quantized on a cube grid of 2^21 cells per axis that covers the bounding box of the finite points
        /// (one scale for all three axes, so a flat scene does not spend bits on its thin side). A point that is not finite gets key 0.
        /// </summary>
        public static ulong[] Keys(IReadOnlyList<float3> positions)
        {
            if (positions == null) throw new ArgumentNullException(nameof(positions));

            int count = positions.Count;
            var keys = new ulong[count];
            if (count == 0)
                return keys;

            float3 min = new float3(float.PositiveInfinity);
            float3 max = new float3(float.NegativeInfinity);
            for (int i = 0; i < count; i++)
            {
                float3 p = positions[i];
                if (!math.all(math.isfinite(p)))
                    continue;
                min = math.min(min, p);
                max = math.max(max, p);
            }
            if (!math.all(math.isfinite(min)))
                return keys; // no finite point at all

            float extent = math.cmax(max - min);
            for (int i = 0; i < count; i++)
            {
                float3 p = positions[i];
                if (!math.all(math.isfinite(p)))
                    continue;
                // A division, so that the largest point is exactly 1 and gets the last cell.
                float3 q = extent > 0f ? math.clamp((p - min) / extent * (float)MaxCoordinate, 0f, (float)MaxCoordinate) : float3.zero;
                keys[i] = Interleave((uint)q.x, (uint)q.y, (uint)q.z);
            }
            return keys;
        }

        /// <summary>
        /// The order in which to place the points: <c>result[k]</c> is the index of the point that goes to rank k. Ascending Morton key; points with
        /// equal keys keep their input order, so the result does not depend on the sort algorithm.
        /// </summary>
        public static int[] Order(IReadOnlyList<float3> positions)
        {
            ulong[] keys = Keys(positions);
            var order = new int[keys.Length];
            for (int i = 0; i < order.Length; i++)
                order[i] = i;
            Array.Sort(order, (a, b) =>
            {
                int c = keys[a].CompareTo(keys[b]);
                return c != 0 ? c : a.CompareTo(b);
            });
            return order;
        }

        /// <summary>The bits of x, y and z interleaved: x in bits 0, 3, 6, ..., y in bits 1, 4, ..., z in bits 2, 5, .... Only the low 21 bits of each count.</summary>
        public static ulong Interleave(uint x, uint y, uint z)
        {
            return Spread(x) | (Spread(y) << 1) | (Spread(z) << 2);
        }

        // Puts a gap of two zero bits behind each of the low 21 bits.
        static ulong Spread(ulong v)
        {
            v &= 0x1FFFFFUL;
            v = (v | (v << 32)) & 0x001F00000000FFFFUL;
            v = (v | (v << 16)) & 0x001F0000FF0000FFUL;
            v = (v | (v << 8)) & 0x100F00F00F00F00FUL;
            v = (v | (v << 4)) & 0x10C30C30C30C30C3UL;
            v = (v | (v << 2)) & 0x1249249249249249UL;
            return v;
        }
    }
}
