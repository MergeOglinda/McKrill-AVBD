using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>
    /// One explosion as the predict shader reads it (64 B). Mirrors <c>ExplosionRecord</c> in Shaders/Include/AvbdExplosions.hlsl.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct ExplosionRecord
    {
        /// <summary>xyz = center, w = radius of a sphere region (not used by a box).</summary>
        public float4 centerRadius;

        /// <summary>Box region: orientation (unit quaternion).</summary>
        public float4 boxRotation;

        /// <summary>xyz = box region: half the size along each of its axes, w = the impulse in N s that a body at the center would receive.</summary>
        public float4 halfExtentsImpulse;

        /// <summary>x = seed (makes the hashed kicks of different explosions different), y = <see cref="ExplosionRegion"/> as a number,
        /// z = include mask (a body is hit only if its group mask has one of these bits...), w = exclude mask (...and none of these).</summary>
        public uint4 ident;
    }
}
