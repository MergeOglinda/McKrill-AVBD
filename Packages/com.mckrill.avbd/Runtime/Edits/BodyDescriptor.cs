using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>Everything needed to spawn a box or a sphere body. Use <see cref="Box"/> or <see cref="Sphere"/> to get sensible defaults.</summary>
    public struct BodyDescriptor
    {
        /// <summary>Position of the center of mass.</summary>
        public float3 position;

        /// <summary>Orientation. Normalized when the body is spawned.</summary>
        public quaternion rotation;

        public float3 linearVelocity;
        public float3 angularVelocity;

        /// <summary>Full box widths in each dimension. Every component must be positive. For a sphere only x is used: the diameter (see <see cref="Sphere"/>).</summary>
        public float3 size;

        /// <summary>True for a sphere (a ball of diameter <c>size.x</c>), false for a box. The shape of a body cannot change after it is spawned.</summary>
        public bool sphere;

        /// <summary>Mass per volume. 0 makes the body static (never solved, never moves).</summary>
        public float density;

        public float friction;

        /// <summary>Collision group bits. Meaning is defined by the broadphase; the default has every bit set.</summary>
        public uint groupMask;

        /// <summary>Index into the user's appearance table. 0xFFFFFFFF is reserved.</summary>
        public uint appearanceId;

        /// <summary>A kinematic body is moved by the user and not solved.</summary>
        public bool kinematic;

        /// <summary>
        /// A solid sphere at rest with identity rotation, friction 0.5 and every collision group bit set. <paramref name="radius"/> must be positive; the mass
        /// is 4/3 pi r^3 times <paramref name="density"/>. A sphere collides with boxes and with other spheres by one contact point per pair.
        /// </summary>
        public static BodyDescriptor Sphere(float radius, float density, float3 position)
        {
            return new BodyDescriptor
            {
                position = position,
                rotation = quaternion.identity,
                size = new float3(radius * 2f),
                sphere = true,
                density = density,
                friction = 0.5f,
                groupMask = 0xFFFFFFFFu,
            };
        }

        /// <summary>A box at rest with identity rotation, friction 0.5 and every collision group bit set.</summary>
        public static BodyDescriptor Box(float3 size, float density, float3 position)
        {
            return new BodyDescriptor
            {
                position = position,
                rotation = quaternion.identity,
                size = size,
                density = density,
                friction = 0.5f,
                groupMask = 0xFFFFFFFFu,
            };
        }
    }
}
