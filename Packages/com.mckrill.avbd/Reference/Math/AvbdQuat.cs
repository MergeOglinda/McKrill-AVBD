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

        // ---- Upstream-exact operations (avbd-demo3d maths.h) ----------------------------------------
        // The solver port uses these instead of Unity.Mathematics' quaternion operators so that every
        // formula is written out the same way as the C++ reference. Quaternion layout is the same:
        // value.xyz = vector, value.w = scalar.

        /// <summary>Hamilton product a * b, written out term by term like upstream's quat operator*.</summary>
        public static quaternion Mul(quaternion a, quaternion b)
        {
            float4 p = a.value;
            float4 q = b.value;
            return new quaternion(new float4(
                p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y,
                p.w * q.y - p.x * q.z + p.y * q.w + p.z * q.x,
                p.w * q.z + p.x * q.y - p.y * q.x + p.z * q.w,
                p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z));
        }

        /// <summary>Conjugate (-x, -y, -z, w). No normalization.</summary>
        public static quaternion Conjugate(quaternion q)
        {
            float4 v = q.value;
            return new quaternion(new float4(-v.x, -v.y, -v.z, v.w));
        }

        /// <summary>Inverse as conjugate / squared length, like upstream. No normalization.</summary>
        public static quaternion Inverse(quaternion q)
        {
            float4 v = q.value;
            float lengthSq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            return new quaternion(new float4(-v.x, -v.y, -v.z, v.w) / lengthSq);
        }

        /// <summary>Rotates v by q using upstream's formula v + 2w(u x v) + u x (2 (u x v)).</summary>
        public static float3 Rotate(quaternion q, float3 v)
        {
            float3 u = q.value.xyz;
            float3 t = math.cross(u, v) * 2f;
            return v + t * q.value.w + math.cross(u, t);
        }

        /// <summary>Rotates v by qAng, then translates by qLin (upstream's transform()).</summary>
        public static float3 Transform(float3 qLin, quaternion qAng, float3 v)
        {
            return Rotate(qAng, v) + qLin;
        }

        /// <summary>
        /// Upstream's quat - quat: the vector part of 2 * a * inverse(b), with NO sign canonicalization.
        /// This is what the C++ reference uses everywhere (joint angular error, velocity update, body
        /// primal right-hand side). <see cref="Difference"/> adds a w &lt; 0 flip that upstream does not
        /// have; the two differ only when the relative rotation exceeds 180 degrees.
        /// </summary>
        public static float3 DifferenceRaw(quaternion a, quaternion b)
        {
            return Mul(a, Inverse(b)).value.xyz * 2f;
        }
    }
}
