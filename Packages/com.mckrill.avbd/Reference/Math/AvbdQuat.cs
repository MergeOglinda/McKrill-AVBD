using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// Quaternion helpers for the AVBD rigid-body solver (CPU reference).
    /// All quaternions are unit quaternions in Unity.Mathematics layout: value.xyz = vector, value.w = scalar.
    /// </summary>
    public static class AvbdQuat
    {
        /// <summary>
        /// Rotation-error vector between two orientations: the vector part of 2 * a * conj(b),
        /// with the sign canonicalized so the shortest rotation is used (q and -q describe the same
        /// orientation). For small rotations the result approximates the rotation vector
        /// (axis * angle) that takes b to a, expressed in the world frame.
        /// </summary>
        /// <remarks>
        /// ASSUMPTION TO VERIFY against upstream avbd-demo3d: the sign canonicalization (w &lt; 0 flip)
        /// is not taken from the paper text; confirm against the upstream operator.
        /// </remarks>
        public static float3 Difference(quaternion a, quaternion b)
        {
            float4 d = math.mul(a, math.conjugate(b)).value;
            float s = d.w < 0f ? -2f : 2f;
            return s * d.xyz;
        }

        /// <summary>
        /// Applies a world-frame angular update dw to q, following Eq. 9 of the AVBD real-time-live
        /// abstract: q' = normalize(q + 0.5 * (0, dw) * q).
        /// </summary>
        public static quaternion IntegrateAngular(quaternion q, float3 dw)
        {
            quaternion dq = new quaternion(new float4(dw, 0f)); // raw constructor: no normalization
            float4 r = q.value + 0.5f * math.mul(dq, q).value;
            return math.normalize(new quaternion(r));
        }

        /// <summary>
        /// World-space 3x3 matrix R * diag(d) * R^T for a body-space diagonal d. Use with the
        /// principal inertia (to get the rotated moment of inertia) or inverse inertia.
        /// </summary>
        public static float3x3 RotatedDiagonal(quaternion q, float3 d)
        {
            float3x3 r = new float3x3(q);
            return math.mul(math.mul(r, float3x3.Scale(d)), math.transpose(r));
        }

        /// <summary>Outer product a * b^T as a 3x3 matrix (entry [i][j] = a_i * b_j).</summary>
        public static float3x3 Outer(float3 a, float3 b)
        {
            return new float3x3(a * b.x, a * b.y, a * b.z);
        }

        /// <summary>Cross-product (skew-symmetric) matrix: Skew(a) * v == cross(a, v).</summary>
        public static float3x3 Skew(float3 a)
        {
            return new float3x3(
                new float3(0f, a.z, -a.y),   // column 0
                new float3(-a.z, 0f, a.x),   // column 1
                new float3(a.y, -a.x, 0f));  // column 2
        }
    }
}
