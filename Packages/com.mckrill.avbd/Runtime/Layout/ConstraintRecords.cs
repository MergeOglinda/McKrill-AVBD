using System;
using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>Constraint type, stored in the high 8 bits of the flags word (rule P7). Mirrors AVBD_CONSTRAINT_KIND_* in AvbdConstraintRecords.hlsl.</summary>
    internal enum ConstraintKind : uint
    {
        /// <summary>Ball-socket joint with an optional angle lock and fracture. Body A may be the world.</summary>
        Joint = 0,

        /// <summary>Linear spring between a point on body A and a point on body B. A plain force: no multiplier, no penalty.</summary>
        Spring = 1,
    }

    /// <summary>
    /// Constraint flag bits (rule P7), in the low 24 bits of the flags word. Each bit has one owning writer.
    /// Mirrors AVBD_CONSTRAINT_FLAG_* in AvbdConstraintRecords.hlsl.
    /// </summary>
    [Flags]
    internal enum ConstraintFlags : uint
    {
        None = 0,

        /// <summary>
        /// The linear part is a hard constraint (stabilized, with a multiplier). The reference solver encodes this as an infinite
        /// stiffness; the record carries a flag instead, because GPU fast-math is not guaranteed to preserve infinity.
        /// The stored stiffness is then 0 and ignored. Owner: constraint creation.
        /// </summary>
        HardLinear = 1u << 0,

        /// <summary>The angular part is a hard constraint. The stored stiffness is then 0 and ignored. Owner: constraint creation.</summary>
        HardAngular = 1u << 1,

        /// <summary>The joint breaks when its angular multiplier exceeds the stored fracture value. Owner: constraint creation.</summary>
        Fracturable = 1u << 2,

        /// <summary>The joint has fractured and exerts no force. Owner: stage D (not written by stage V).</summary>
        Broken = 1u << 3,
    }

    /// <summary>Packing of the constraint flags word: low 24 bits are flags, high 8 bits are the <see cref="ConstraintKind"/>.</summary>
    internal static class ConstraintFlagLayout
    {
        public const int KindShift = 24;
        public const uint FlagsMask = 0x00FFFFFFu;

        public static uint Pack(ConstraintFlags flags, ConstraintKind kind)
        {
            return ((uint)kind << KindShift) | ((uint)flags & FlagsMask);
        }

        public static ConstraintFlags FlagsOf(uint packed) => (ConstraintFlags)(packed & FlagsMask);

        public static ConstraintKind KindOf(uint packed) => (ConstraintKind)(packed >> KindShift);
    }

    /// <summary>Strides of the constraint arrays in bytes. Tests assert that the record struct has exactly this size.</summary>
    internal static class ConstraintRecordStrides
    {
        /// <summary>Ten 16 B words: a multiple of 32 B, because the primal and dual stages gather constraints by id (rule P3).</summary>
        public const int Constraint = 160;
    }

    /// <summary>
    /// One joint or spring. Every stage that touches a constraint (V and D, and later the per-step warm start) reads nearly all of
    /// it, so there is no hot/cold split (rule P5). The unused words of a spring are zero; a separate smaller spring
    /// record is a possible later optimization. Field-by-field mirror of ConstraintRecord in AvbdConstraintRecords.hlsl.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct ConstraintRecord
    {
        /// <summary>x = body A slot (AVBD_INVALID for a joint pinned to the world point rA), y = body B slot, z = flags | kind &lt;&lt; 24, w unused.</summary>
        public uint4 ident;

        /// <summary>xyz = anchor on body A in body space (a world-space point if A is the world). w = linear stiffness of a joint, stiffness of a spring.</summary>
        public float4 anchorA;

        /// <summary>xyz = anchor on body B in body space. w = angular stiffness of a joint, rest length of a spring.</summary>
        public float4 anchorB;

        /// <summary>xyz = linear constraint value at the start of the step (C0). w = torque arm (joint).</summary>
        public float4 c0Lin;

        /// <summary>xyz = angular constraint value at the start of the step. w = fracture threshold (joint).</summary>
        public float4 c0Ang;

        /// <summary>Per-axis penalty of the linear rows. w unused.</summary>
        public float4 penaltyLin;

        /// <summary>Per-axis penalty of the angular rows. w unused.</summary>
        public float4 penaltyAng;

        /// <summary>Multiplier of the linear rows. w unused.</summary>
        public float4 lambdaLin;

        /// <summary>Multiplier of the angular rows. w unused.</summary>
        public float4 lambdaAng;

        /// <summary>Reserved, zero. Pads the record to 160 B.</summary>
        public float4 reserved;
    }
}
