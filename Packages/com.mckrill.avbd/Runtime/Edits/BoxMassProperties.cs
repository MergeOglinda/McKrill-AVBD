using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>Mass properties of a solid box, same formulas and operation order as upstream avbd-demo3d (rigid.cpp).</summary>
    internal static class BoxMassProperties
    {
        /// <summary>Returns xyz = principal moments of inertia about the box axes, w = mass.</summary>
        public static float4 Compute(float3 size, float density)
        {
            float mass = size.x * size.y * size.z * density;
            return new float4(
                (size.y * size.y + size.z * size.z) / 12f * mass,
                (size.x * size.x + size.z * size.z) / 12f * mass,
                (size.x * size.x + size.y * size.y) / 12f * mass,
                mass);
        }
    }
}
