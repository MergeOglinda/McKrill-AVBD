using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>
    /// One force field as the predict shader reads it (64 B). Mirrors <c>ForceFieldRecord</c> in Shaders/Include/AvbdForceFields.hlsl.
    /// Only float4 and uint4 members, like every record (layout rule P1).
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct ForceFieldRecord
    {
        /// <summary>Constant: the force in N. Drag and wing: the velocity of the air in m/s.</summary>
        public float4 value;

        /// <summary>Wing: xyz = the normal of the wing in the body's frame (unit).</summary>
        public float4 axis;

        /// <summary>Gravity scale: x = scale. Drag: x = linear, y = quadratic. Wing: x = lift slope, y = drag coefficient, z = half the air density times the area.</summary>
        public float4 coefficients;

        /// <summary>x = <see cref="ForceKind"/> (0 for an empty slot), y = group mask of the bodies the field acts on, z = flags.</summary>
        public uint4 ident;
    }
}
