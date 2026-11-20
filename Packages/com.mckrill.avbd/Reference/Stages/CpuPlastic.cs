using System;
using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// Helpers of the CPU oracles for plastic joints (see <see cref="PlasticRecord"/>). The oracles take the plastic array as an optional
    /// argument: null means no plastic joints, and a joint with the Plastic flag then throws.
    /// </summary>
    internal static class CpuPlastic
    {
        /// <summary>The plastic record of a joint with <see cref="ConstraintFlags.Plastic"/> or <see cref="ConstraintFlags.PlasticLinear"/> (slot ident.w).</summary>
        public static ref PlasticRecord Of(PlasticRecord[] plastic, in ConstraintRecord joint)
        {
            if (plastic == null)
                throw new InvalidOperationException("A plastic joint needs the plastic array.");
            return ref plastic[joint.ident.w];
        }

        /// <summary>The rest rotation of a joint: the identity unless it is plastic.</summary>
        public static quaternion RestOf(PlasticRecord[] plastic, in ConstraintRecord joint, ConstraintFlags flags)
        {
            return (flags & ConstraintFlags.Plastic) != 0 ? new quaternion(Of(plastic, joint).rest) : quaternion.identity;
        }
    }
}
