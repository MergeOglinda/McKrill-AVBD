using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>Mass properties of a solid sphere.</summary>
    internal static class SphereMassProperties
    {
        /// <summary>Returns xyz = the moment of inertia about every axis (2/5 m r^2), w = mass (4/3 pi r^3 times the density).</summary>
        public static float4 Compute(float radius, float density)
        {
            float mass = 4f / 3f * math.PI * radius * radius * radius * density;
            float inertia = 0.4f * mass * radius * radius;
            return new float4(inertia, inertia, inertia, mass);
        }
    }
}
