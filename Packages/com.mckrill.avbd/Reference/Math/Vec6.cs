using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// A 6-dimensional rigid-body vector: linear part (indices 0-2) followed by angular part (indices 3-5).
    /// Used for generalized positions errors, gradients and Newton updates of one rigid body.
    /// </summary>
    public struct Vec6
    {
        public float3 lin;
        public float3 ang;

        public Vec6(float3 lin, float3 ang)
        {
            this.lin = lin;
            this.ang = ang;
        }

        public float this[int i]
        {
            get => i < 3 ? lin[i] : ang[i - 3];
            set
            {
                if (i < 3) lin[i] = value;
                else ang[i - 3] = value;
            }
        }

        public static Vec6 Zero => default;

        public static Vec6 operator +(Vec6 a, Vec6 b) => new Vec6(a.lin + b.lin, a.ang + b.ang);
        public static Vec6 operator -(Vec6 a, Vec6 b) => new Vec6(a.lin - b.lin, a.ang - b.ang);
        public static Vec6 operator -(Vec6 a) => new Vec6(-a.lin, -a.ang);
        public static Vec6 operator *(Vec6 a, float s) => new Vec6(a.lin * s, a.ang * s);
        public static Vec6 operator *(float s, Vec6 a) => new Vec6(a.lin * s, a.ang * s);

        public static float Dot(Vec6 a, Vec6 b) => math.dot(a.lin, b.lin) + math.dot(a.ang, b.ang);

        public static float LengthSq(Vec6 a) => Dot(a, a);

        /// <summary>
        /// The 6D "difference" of two rigid-body poses: position difference followed by the
        /// rotation-error vector from <see cref="AvbdQuat.Difference"/>. This is the quantity the
        /// paper calls subtraction of two rigid-body states.
        /// </summary>
        public static Vec6 PoseDifference(float3 posA, quaternion rotA, float3 posB, quaternion rotB)
        {
            return new Vec6(posA - posB, AvbdQuat.Difference(rotA, rotB));
        }
    }
}
