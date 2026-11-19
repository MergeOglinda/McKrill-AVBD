using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>
    /// One explosion as the predict shader reads it (48 B). Mirrors <c>ExplosionRecord</c> in Shaders/Include/AvbdExplosions.hlsl.
    /// Only float4 and uint4 members, like every record (layout rule P1).
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct ExplosionRecord
    {
        /// <summary>xyz = center, w = radius.</summary>
        public float4 centerRadius;

        /// <summary>x = the impulse in N s that a body at the center would receive.</summary>
        public float4 impulse;

        /// <summary>x = seed: makes the hashed kicks of different explosions different.</summary>
        public uint4 ident;
    }
}
