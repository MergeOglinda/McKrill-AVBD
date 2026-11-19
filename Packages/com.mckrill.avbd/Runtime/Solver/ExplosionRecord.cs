using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>
    /// One explosion as the predict shader reads it (32 B). Mirrors <c>ExplosionRecord</c> in Shaders/Include/AvbdExplosions.hlsl.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct ExplosionRecord
    {
        /// <summary>xyz = center, w = radius.</summary>
        public float4 centerRadius;

        /// <summary>The impulse in N s that a body at the center would receive.</summary>
        public float impulse;

        /// <summary>Makes the hashed kicks of different explosions different.</summary>
        public uint seed;

        public float2 padding;
    }
}
