using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// 3x3 helpers for the solver port. The upstream C++ stores matrices row-major and builds them from
    /// rows; Unity.Mathematics stores float3x3 column-major and its float3x3(float3, float3, float3)
    /// constructor takes COLUMNS. Everything in the reference layer treats float3x3 as the mathematical
    /// matrix (math.mul(m, v) is m * v), and these helpers do the row/column bookkeeping in one place.
    /// </summary>
    public static class AvbdMat3
    {
        /// <summary>Matrix whose rows are r0, r1, r2.</summary>
        public static float3x3 FromRows(float3 r0, float3 r1, float3 r2)
        {
            return math.transpose(new float3x3(r0, r1, r2));
        }

        /// <summary>Row i of m as a vector.</summary>
        public static float3 Row(float3x3 m, int i)
        {
            return new float3(m.c0[i], m.c1[i], m.c2[i]);
        }

        public static float3x3 Diagonal(float a, float b, float c)
        {
            return new float3x3(new float3(a, 0f, 0f), new float3(0f, b, 0f), new float3(0f, 0f, c));
        }

        public static float3x3 Diagonal(float3 d)
        {
            return Diagonal(d.x, d.y, d.z);
        }

        /// <summary>Diagonal matrix of the column lengths (upstream's diagonalize()).</summary>
        public static float3x3 Diagonalize(float3x3 m)
        {
            return Diagonal(math.length(m.c0), math.length(m.c1), math.length(m.c2));
        }

        /// <summary>Component-wise clamp to [lo, hi] (upstream's clamp(float3, float, float)).</summary>
        public static float3 Clamp(float3 v, float lo, float hi)
        {
            return math.min(math.max(v, new float3(lo)), new float3(hi));
        }
    }
}
