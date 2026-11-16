using System;
using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle for stage S: the Morton key of every body and the stable order of the slots by key, which <c>AvbdMortonSort.compute</c> must
    /// produce. The key of a solved body at a finite position is the Morton code of the low 10 bits of its broadphase grid cell (cell size = twice the
    /// largest radius of the solved bodies, times the broadphase's margin, at least 1e-4); every other body has <see cref="InvalidKey"/>.
    /// The order sorts by key and breaks ties by slot. The GPU may round the division by the cell size differently, so a test compares keys only
    /// for bodies that are not on a cell boundary (<see cref="NearCellBoundary"/>).
    /// </summary>
    internal static class CpuMortonSort
    {
        public const uint InvalidKey = AvbdMortonSort.InvalidKey;
        public const int AxisBits = AvbdMortonSort.AxisBits;

        /// <summary>Spreads the low 10 bits of v to every third bit of a 30-bit word, the obvious way (a loop, not the shader's multiplications).</summary>
        public static uint ExpandBits(uint v)
        {
            uint result = 0u;
            for (int bit = 0; bit < AxisBits; bit++)
                result |= ((v >> bit) & 1u) << (3 * bit);
            return result;
        }

        public static uint Key(int3 cell)
        {
            uint mask = (1u << AxisBits) - 1u;
            uint x = unchecked((uint)cell.x) & mask;
            uint y = unchecked((uint)cell.y) & mask;
            uint z = unchecked((uint)cell.z) & mask;
            return (ExpandBits(x) << 2) | (ExpandBits(y) << 1) | ExpandBits(z);
        }

        public static float CellSize(CpuBodyArrays bodies, int bodyCount) =>
            math.max(2f * CpuBroadphase.MaxRadius(bodies, bodyCount) * AvbdBroadphase.CellMargin, 1e-4f);

        public static int3 CellOf(float3 position, float cellSize) => (int3)math.clamp(math.floor(position / cellSize), -1e9f, 1e9f);

        public static uint Key(CpuBodyArrays bodies, int slot, float cellSize)
        {
            if (CpuBroadphase.Classify(bodies, slot) != CpuBroadphase.BodyClass.Solved)
                return InvalidKey;
            float3 position = bodies.State[slot].position.xyz;
            if (!math.all(math.isfinite(position)))
                return InvalidKey;
            return Key(CellOf(position, cellSize));
        }

        /// <summary>The key of every slot below <paramref name="bodyCount"/>.</summary>
        public static uint[] Keys(CpuBodyArrays bodies, int bodyCount)
        {
            float cellSize = CellSize(bodies, bodyCount);
            var keys = new uint[bodyCount];
            for (int s = 0; s < bodyCount; s++)
                keys[s] = Key(bodies, s, cellSize);
            return keys;
        }

        /// <summary>The slots sorted by key, equal keys in slot order (a stable sort).</summary>
        public static uint[] Order(uint[] keys)
        {
            var order = new uint[keys.Length];
            for (int i = 0; i < order.Length; i++)
                order[i] = (uint)i;
            Array.Sort(order, (a, b) => keys[a] != keys[b] ? keys[a].CompareTo(keys[b]) : a.CompareTo(b));
            return order;
        }

        /// <summary>True if the position is within <paramref name="tolerance"/> (in cells) of a cell boundary on some axis, where a rounding difference can change the cell.</summary>
        public static bool NearCellBoundary(CpuBodyArrays bodies, int slot, float cellSize, float tolerance)
        {
            float3 q = bodies.State[slot].position.xyz / cellSize;
            float3 distance = math.abs(q - math.round(q));
            return math.any(distance < tolerance);
        }
    }
}
