using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>Stride of the plastic joint array in bytes. Tests assert that the record struct has exactly this size.</summary>
    internal static class PlasticRecordStrides
    {
        /// <summary>
        /// Four 16 B words: a multiple of 32 B, because the primal and dual stages gather it by the joint's slot (rule P3). The two words that are read every
        /// iteration (rest, limits) come first, so they are one 32 B sector; the two that are read when a joint yields or breaks (restStart, reserved) are the second.
        /// </summary>
        public const int Plastic = 64;
    }

    /// <summary>
    /// The extra state of a plastic joint, in a buffer of its own that only joints with <see cref="ConstraintFlags.Plastic"/> or <see cref="ConstraintFlags.PlasticLinear"/> read: a joint without
    /// it keeps the 128 B <see cref="GpuConstraintRecord"/> and its memory traffic. The joint finds its record at <c>ident.w</c>. The buffer exists only if
    /// the constraint buffers were made with a plastic capacity (see <see cref="AvbdConstraintBuffers"/>), and the kernels that read it are
    /// separate compiled variants (AVBD_PLASTIC_JOINTS in AvbdDual.compute and AvbdPrimal.compute). Field-by-field mirror of PlasticRecord in
    /// AvbdPlasticRecords.hlsl.
    ///
    /// The angular row of a joint compares the rotation of body B with <c>rotA * rest</c>. A new joint has the identity as rest, which is the
    /// reference solver's joint. When the angular multiplier of a plastic joint is above the yield threshold, the joint yields: the multiplier is
    /// cut back to the threshold and rest becomes the current relative rotation, so the bend stays. Owner of rest: stage D.
    ///
    /// The linear row of a plastic joint holds the anchor on body A (<see cref="ConstraintRecord.anchorA"/>) at the anchor on body B. A new joint has the
    /// anchors it was made with, which is the reference solver's joint. When the linear multiplier is above the linear yield threshold, the multiplier is cut
    /// back to the threshold and the anchor on body A moves to the current position of the anchor on body B, so the stretch stays. The anchor on body A is
    /// the state of the linear row (like rest for the angular row): it lives in the constraint record, which the primal stage reads anyway, so the primal
    /// stage needs no plastic code for it. <see cref="reserved"/> keeps the anchor it started as.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct PlasticRecord
    {
        // The first 32 B (one sector) are what the primal and dual stages read of every plastic joint in every iteration: the rest rotation and the limits.
        // The second 32 B are read only in the step in which a joint breaks or yields. A record is 64 B at a multiple of 64 B, so the two halves are two
        // sectors and a read of the hot half touches one.

        /// <summary>Rest rotation, a unit quaternion (x, y, z, w): body B is at rest when its rotation is <c>rotA * rest</c>. Identity at creation. Owner: stage D.</summary>
        public float4 rest;

        /// <summary>
        /// x = angular yield threshold (norm of the angular multiplier; read only with <see cref="ConstraintFlags.Plastic"/>), y = bend limit in radians (read only
        /// with <see cref="ConstraintFlags.BendBreak"/>), z = linear yield threshold (norm of the linear multiplier; read only with
        /// <see cref="ConstraintFlags.PlasticLinear"/>), w = stretch limit in meters (read only with <see cref="ConstraintFlags.StretchBreak"/>).
        /// </summary>
        public float4 limits;

        /// <summary>The rest rotation at creation, the origin the bend is measured from. Written once, at creation. Read only when a joint yields (to test the bend limit).</summary>
        public float4 restStart;

        /// <summary>xyz = the anchor on body A at creation (a world point if A is the world), the origin the stretch is measured from. Written once, at creation (with <see cref="ConstraintFlags.PlasticLinear"/>). w unused.</summary>
        public float4 reserved;
    }
}
