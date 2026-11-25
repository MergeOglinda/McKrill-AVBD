using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>
    /// One explosion as the predict shader reads it (48 B). Mirrors <c>ExplosionRecord</c> in Shaders/Include/AvbdExplosions.hlsl.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct ExplosionRecord
    {
        /// <summary>xyz = center, w = radius. For an explosion anchored to a body (<see cref="anchorSlot"/>) the center is not known to the CPU: the
        /// <c>ResolveExplosions</c> kernel writes it, from the pose of the body at the start of the step.</summary>
        public float4 centerRadius;

        /// <summary>The impulse in N s that a body at the center would receive.</summary>
        public float impulse;

        /// <summary>Makes the hashed kicks of different explosions different.</summary>
        public uint seed;

        /// <summary>Slot + 1 of the body that the center is relative to; 0 for a center in world space.</summary>
        public uint anchorSlot;

        public float padding;

        /// <summary>xyz = the center in the frame of the anchor body (origin at its center of mass, its axes), w unused. Zero when there is no anchor.</summary>
        public float4 anchorLocal;
    }
}
