using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>Stride of the plastic joint array in bytes. Tests assert that the record struct has exactly this size.</summary>
    internal static class PlasticRecordStrides
    {
        /// <summary>Four 16 B words: a multiple of 32 B, because the primal and dual stages gather it by the joint's slot (rule P3).</summary>
        public const int Plastic = 64;
    }

    /// <summary>
    /// The extra state of a plastic joint, in a buffer of its own that only joints with <see cref="ConstraintFlags.Plastic"/> read: a joint without
    /// it keeps the 160 B <see cref="ConstraintRecord"/> and its memory traffic. The joint finds its record at <c>ident.w</c>. The buffer exists only if
    /// the constraint buffers were made with a plastic capacity (see <see cref="AvbdConstraintBuffers"/>), and the kernels that read it are
    /// separate compiled variants (AVBD_PLASTIC_JOINTS in AvbdDual.compute and AvbdPrimal.compute). Field-by-field mirror of PlasticRecord in
    /// AvbdPlasticRecords.hlsl.
    ///
    /// The angular row of a joint compares the rotation of body B with <c>rotA * rest</c>. A new joint has the identity as rest, which is the
    /// reference solver's joint. When the angular multiplier of a plastic joint is above the yield threshold, the joint yields: the multiplier is
    /// cut back to the threshold and rest becomes the current relative rotation, so the bend stays. Owner of rest: stage D.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct PlasticRecord
    {
        /// <summary>Rest rotation, a unit quaternion (x, y, z, w): body B is at rest when its rotation is <c>rotA * rest</c>. Identity at creation. Owner: stage D.</summary>
        public float4 rest;

        /// <summary>The rest rotation at creation, the origin the bend is measured from. Written once, at creation.</summary>
        public float4 restStart;

        /// <summary>x = yield threshold (norm of the angular multiplier), y = bend limit in radians (read only with <see cref="ConstraintFlags.BendBreak"/>), zw unused.</summary>
        public float4 limits;

        /// <summary>Reserved, zero. Pads the record to 64 B.</summary>
        public float4 reserved;
    }
}
