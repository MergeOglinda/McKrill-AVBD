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

        /// <summary>
        /// One contact point of a box-box manifold, in the form the CPU oracle (<c>CpuContacts</c>) stores it. The GPU keeps the manifolds of the
        /// pool as <see cref="ContactPairRecord"/>s and <see cref="ContactPointRecord"/>s, and <see cref="ContactCodec.ToRecord"/> converts a point
        /// to this form. Never added by the user. See <see cref="ContactLayout"/>.
        /// </summary>
        Contact = 2,
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

        /// <summary>
        /// A contact record of the CPU oracle held static friction in the last dual update. Owner: stage D (stage N only carries it over, to decide which
        /// anchors to keep). The GPU keeps these bits in the pair record (<see cref="ContactPairRecord.ident"/>.w, bit k for point k).
        /// </summary>
        Stick = 1u << 4,

        /// <summary>
        /// The joint is plastic: its <see cref="PlasticRecord"/> is slot ident.w of the plastic array. Needs <see cref="HardAngular"/>. Without this flag
        /// ident.w means nothing (a contact record of the CPU oracle keeps its feature key there). Owner: constraint creation.
        /// </summary>
        Plastic = 1u << 5,

        /// <summary>A plastic joint breaks when it has bent further than <see cref="PlasticRecord.limits"/>.y radians off its original rest rotation. Owner: constraint creation.</summary>
        BendBreak = 1u << 6,
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
    ///
    /// A contact uses the same record: ident = (body A, body B, flags | kind, feature key), anchorA = (rA, friction),
    /// anchorB = (rB, unused), c0Lin = (C0 in the contact basis, unused), c0Ang = (contact normal pointing from B to A, unused),
    /// penaltyLin = per-row penalty (normal, tangent 1, tangent 2), lambdaLin = per-row multiplier. The tangents are derived from the normal.
    /// The remaining words are zero.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct ConstraintRecord
    {
        /// <summary>x = body A slot (AVBD_INVALID for a joint pinned to the world point rA), y = body B slot, z = flags | kind &lt;&lt; 24, w = feature key (contact), slot in the plastic array (joint with <see cref="ConstraintFlags.Plastic"/>), else unused.</summary>
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

namespace McKrill.Avbd
{
    /// <summary>
    /// Layout of the contacts, and the constants of box-box collision (the reference solver's COLLISION_MARGIN and STICK_THRESH). Mirrors AvbdContacts.hlsl.
    ///
    /// Manifolds live in a pool of slots. Slot s has a <see cref="ContactPairRecord"/> (the pair of bodies it belongs to, body A the higher slot, the
    /// contact count, the stick bits, the normal and the friction) and room for <see cref="MaxContactsPerPair"/> <see cref="ContactPointRecord"/>s
    /// starting at <c>s * MaxContactsPerPair</c> in the point array. The CPU oracle stores a contact as a <see cref="ConstraintRecord"/> at
    /// <c>contactBase + pair * MaxContactsPerPair + k</c> of its own array instead.
    /// A pair keeps its slot while it has contacts, which is how the warm start finds the previous contacts; the narrow phase takes a free
    /// slot for a new pair (from the pairs of the broadphase) and gives it back when the pair has no contacts.
    ///
    /// The dense pair index below is how the CPU oracle stores its manifolds (every pair of bodies has a fixed place) and the sort key of
    /// the contact lists: the lists order the contacts by pair, like the reference solver's first step.
    /// </summary>
    internal static class ContactLayout
    {
        /// <summary>Contacts per manifold (the reference solver's MAX_CONTACTS).</summary>
        public const int MaxContactsPerPair = 8;

        /// <summary>Distance added to the normal row of every contact's initial constraint value.</summary>
        public const float CollisionMargin = 0.01f;

        /// <summary>A contact sticks if its tangential constraint value is below this after the dual update.</summary>
        public const float StickThreshold = 0.00001f;

        /// <summary>
        /// Mirrors AVBD_SPHERE_FEATURE: the feature key of every contact that involves a sphere (one point per pair). The key sits in the top byte; the
        /// box keys use 0 to 2 there (the axis type). A contact with this key keeps no anchors when it sticks, and the primal and dual stages take its
        /// lever arms at the rotation of the start of the step, because a rolling sphere's contact point moves over its surface.
        /// </summary>
        public const int SphereFeature = 3 << 24;

        /// <summary>Mirrors AVBD_NP_FREE_TOP: the free-slot count in the pool's control buffer.</summary>
        public const int PoolFreeTopWord = 0;

        /// <summary>Mirrors AVBD_NP_OVERFLOW: candidate pairs dropped because no slot was free.</summary>
        public const int PoolOverflowWord = 1;

        /// <summary>
        /// Mirrors AVBD_NP_HIGH_WATER: the highest slot ever taken + 1. Every used slot is below it, and it only grows, so it is the peak number
        /// of manifolds that were ever in use at once (a freed slot is the next to be taken, so the slots in use stay packed at the bottom).
        /// The kernels of stage N and the dual update of the contacts run over this many slots, not over the capacity.
        /// </summary>
        public const int PoolHighWaterWord = 2;

        /// <summary>Mirrors AVBD_NP_CONTROL_WORDS.</summary>
        public const int PoolControlWords = 3;

        // Entries of the indirect dispatch arguments of stage N, AvbdIndirect.ArgsStride bytes each. Mirror AVBD_NP_ARGS_*.
        public const int PoolArgsRetire = 0;
        public const int PoolArgsAllocateDynamic = 1;
        public const int PoolArgsAllocateStatic = 2;
        public const int PoolArgsNarrowphase = 3;
        public const int PoolArgsDual = 4;
        public const int PoolArgsCount = 5;

        /// <summary>Number of pairs of <paramref name="bodyCount"/> bodies.</summary>
        public static long PairCount(int bodyCount) => (long)bodyCount * (bodyCount - 1) / 2;

        /// <summary>Index of the pair of two different slots, in either order.</summary>
        public static long PairIndex(int slotA, int slotB)
        {
            long hi = slotA > slotB ? slotA : slotB;
            long lo = slotA > slotB ? slotB : slotA;
            return hi * (hi - 1) / 2 + lo;
        }

        /// <summary>Inverse of <see cref="PairIndex"/>: the higher slot is a, the lower is b.</summary>
        public static void BodiesOfPair(long pair, out int a, out int b)
        {
            long hi = (long)((1.0 + System.Math.Sqrt(1.0 + 8.0 * pair)) * 0.5);
            while (hi * (hi - 1) / 2 > pair)
                hi--;
            while ((hi + 1) * hi / 2 <= pair)
                hi++;
            a = (int)hi;
            b = (int)(pair - hi * (hi - 1) / 2);
        }
    }
}
