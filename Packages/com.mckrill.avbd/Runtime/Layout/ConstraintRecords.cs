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

        /// <summary>The joint breaks when its angular multiplier exceeds the stored fracture value (c0Ang.w). Owner: constraint creation.</summary>
        Fracturable = 1u << 2,

        /// <summary>The joint has fractured and exerts no force. Owner: stage D (not written by stage V).</summary>
        Broken = 1u << 3,

        /// <summary>
        /// A contact record of the CPU oracle held static friction in the last dual update. Owner: stage D (stage N only carries it over, to decide which
        /// anchors to keep). The GPU keeps these bits in the pair record (<see cref="ContactPairRecord.ident"/>.w, bit k for point k).
        /// </summary>
        Stick = 1u << 4,

        /// <summary>
        /// The joint is plastic in its angular row: its <see cref="PlasticRecord"/> is slot ident.w of the plastic array. Needs <see cref="HardAngular"/>.
        /// Without this flag and without <see cref="PlasticLinear"/>, ident.w means nothing (a contact record of the CPU oracle keeps its feature key there).
        /// Owner: constraint creation.
        /// </summary>
        Plastic = 1u << 5,

        /// <summary>A plastic joint breaks when it has bent further than <see cref="PlasticRecord.limits"/>.y radians off its original rest rotation. Owner: constraint creation.</summary>
        BendBreak = 1u << 6,

        /// <summary>The joint breaks when its linear multiplier exceeds the stored linear fracture value (<see cref="ConstraintRecord.reserved"/>.x). Needs <see cref="HardLinear"/>. Owner: constraint creation.</summary>
        FracturableLinear = 1u << 7,

        /// <summary>
        /// The joint is plastic in its linear row: when the linear multiplier is above the yield value (<see cref="PlasticRecord.limits"/>.z) the anchor on body A
        /// moves to where the anchor on body B is, so the stretch stays. Needs <see cref="HardLinear"/>. The joint's <see cref="PlasticRecord"/> is slot ident.w
        /// of the plastic array, as with <see cref="Plastic"/> (a joint can have both). Owner: constraint creation.
        /// </summary>
        PlasticLinear = 1u << 8,

        /// <summary>A linearly plastic joint breaks when its anchor on body A has moved further than <see cref="PlasticRecord.limits"/>.w meters from where it started. Owner: constraint creation.</summary>
        StretchBreak = 1u << 9,
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

    /// <summary>Strides of the constraint arrays in bytes. Tests assert that the record structs have exactly these sizes.</summary>
    internal static class ConstraintRecordStrides
    {
        /// <summary>
        /// The joint or spring record on the GPU, <see cref="GpuConstraintRecord"/>: eight 16 B words with no padding. A multiple of 32 B, because the primal
        /// and dual stages gather constraints by id (rule P3), and 128 B is four whole sectors and one cache line, which is what the primal sweep reads of every joint twice
        /// per iteration.
        /// </summary>
        public const int Constraint = 128;

        /// <summary>The CPU form of a record, <see cref="ConstraintRecord"/> (ten 16 B words): what the oracles compute with. Packed into <see cref="Constraint"/> bytes for the GPU.</summary>
        public const int Logical = 160;

        /// <summary>The fracture thresholds of four joints, <see cref="FractureRecord"/>.</summary>
        public const int Fracture = 32;
    }

    /// <summary>
    /// A joint or spring as the GPU stores it. Field-by-field mirror of ConstraintRecord in AvbdConstraintRecords.hlsl; <see cref="ConstraintRecordCodec"/> converts
    /// from and to the CPU form <see cref="ConstraintRecord"/>. The three state words hold the penalties and multipliers (12 floats) with no padding between them.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct GpuConstraintRecord
    {
        /// <summary>As <see cref="ConstraintRecord.ident"/>.</summary>
        public uint4 ident;

        /// <summary>As <see cref="ConstraintRecord.anchorA"/>.</summary>
        public float4 anchorA;

        /// <summary>As <see cref="ConstraintRecord.anchorB"/>.</summary>
        public float4 anchorB;

        /// <summary>As <see cref="ConstraintRecord.c0Lin"/>.</summary>
        public float4 c0Lin;

        /// <summary>xyz = angular constraint value at the start of the step. w = the mark of the fused dual update: the id of a sweep as the bits of a float, 0 for none (the CPU form keeps it in reserved.w).</summary>
        public float4 c0Ang;

        /// <summary>penaltyLin.xyz, penaltyAng.x.</summary>
        public float4 state0;

        /// <summary>penaltyAng.yz, lambdaLin.xy.</summary>
        public float4 state1;

        /// <summary>lambdaLin.z, lambdaAng.xyz.</summary>
        public float4 state2;
    }

    /// <summary>
    /// The fracture thresholds of four joints, in an array of its own that only the dual update reads, and only for a joint with a fracture flag: joint j is
    /// lane j &amp; 3 of record j &gt;&gt; 2. <c>a</c> holds the joints 0 and 1 of the record as (linear, angular, linear, angular), <c>b</c> the joints 2 and 3.
    /// Written once, at creation. Mirror of FractureRecord in AvbdConstraintRecords.hlsl.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct FractureRecord
    {
        public float4 a;
        public float4 b;
    }

    /// <summary>Converts a constraint record between the CPU form (<see cref="ConstraintRecord"/>) and the GPU form (<see cref="GpuConstraintRecord"/> and the <see cref="FractureRecord"/> of its joint).</summary>
    internal static class ConstraintRecordCodec
    {
        /// <summary>Records of the fracture array that hold <paramref name="records"/> joints.</summary>
        public static int FractureRecordCount(int records) => (records + 3) / 4;

        /// <summary>The GPU form of a record. The fracture thresholds (c0Ang.w and reserved.x of the CPU form) are not in it: see <see cref="SetFracture"/>.</summary>
        public static GpuConstraintRecord Pack(in ConstraintRecord r)
        {
            return new GpuConstraintRecord
            {
                ident = r.ident,
                anchorA = r.anchorA,
                anchorB = r.anchorB,
                c0Lin = r.c0Lin,
                c0Ang = new float4(r.c0Ang.x, r.c0Ang.y, r.c0Ang.z, r.reserved.w),
                state0 = new float4(r.penaltyLin.x, r.penaltyLin.y, r.penaltyLin.z, r.penaltyAng.x),
                state1 = new float4(r.penaltyAng.y, r.penaltyAng.z, r.lambdaLin.x, r.lambdaLin.y),
                state2 = new float4(r.lambdaLin.z, r.lambdaAng.x, r.lambdaAng.y, r.lambdaAng.z),
            };
        }

        /// <summary>The CPU form of a GPU record, with the fracture thresholds of its joint (<see cref="GetFracture"/>). The unused words are zero.</summary>
        public static ConstraintRecord Unpack(in GpuConstraintRecord g, float fractureLinear, float fractureAngular)
        {
            return new ConstraintRecord
            {
                ident = g.ident,
                anchorA = g.anchorA,
                anchorB = g.anchorB,
                c0Lin = g.c0Lin,
                c0Ang = new float4(g.c0Ang.x, g.c0Ang.y, g.c0Ang.z, fractureAngular),
                penaltyLin = new float4(g.state0.x, g.state0.y, g.state0.z, 0f),
                penaltyAng = new float4(g.state0.w, g.state1.x, g.state1.y, 0f),
                lambdaLin = new float4(g.state1.z, g.state1.w, g.state2.x, 0f),
                lambdaAng = new float4(g.state2.y, g.state2.z, g.state2.w, 0f),
                reserved = new float4(fractureLinear, 0f, 0f, g.c0Ang.w),
            };
        }

        /// <summary>Stores the thresholds of joint <paramref name="index"/>.</summary>
        public static void SetFracture(FractureRecord[] records, int index, float linear, float angular)
        {
            ref FractureRecord r = ref records[index >> 2];
            float4 pair = (index & 2) == 0 ? r.a : r.b;
            if ((index & 1) == 0)
            {
                pair.x = linear;
                pair.y = angular;
            }
            else
            {
                pair.z = linear;
                pair.w = angular;
            }
            if ((index & 2) == 0)
                r.a = pair;
            else
                r.b = pair;
        }

        /// <summary>The thresholds of joint <paramref name="index"/>.</summary>
        public static void GetFracture(FractureRecord[] records, int index, out float linear, out float angular)
        {
            FractureRecord r = records[index >> 2];
            float4 pair = (index & 2) == 0 ? r.a : r.b;
            linear = (index & 1) == 0 ? pair.x : pair.z;
            angular = (index & 1) == 0 ? pair.y : pair.w;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct ConstraintRecord
    {
        /// <summary>x = body A slot (AVBD_INVALID for a joint pinned to the world point rA), y = body B slot, z = flags | kind &lt;&lt; 24, w = feature key (contact), slot in the plastic array (joint with <see cref="ConstraintFlags.Plastic"/>), else unused.</summary>
        public uint4 ident;

        /// <summary>
        /// xyz = anchor on body A in body space (a world-space point if A is the world). w = linear stiffness of a joint, stiffness of a spring.
        /// Owner of xyz: constraint creation; stage D too, for a joint with <see cref="ConstraintFlags.PlasticLinear"/>, whose anchor moves when it yields.
        /// </summary>
        public float4 anchorA;

        /// <summary>xyz = anchor on body B in body space. w = angular stiffness of a joint, rest length of a spring.</summary>
        public float4 anchorB;

        /// <summary>xyz = linear constraint value at the start of the step (C0). w = torque arm (joint).</summary>
        public float4 c0Lin;

        /// <summary>xyz = angular constraint value at the start of the step. w = angular fracture threshold (joint): in the <see cref="FractureRecord"/> on the GPU.</summary>
        public float4 c0Ang;

        /// <summary>Per-axis penalty of the linear rows. w unused.</summary>
        public float4 penaltyLin;

        /// <summary>Per-axis penalty of the angular rows. w unused.</summary>
        public float4 penaltyAng;

        /// <summary>Multiplier of the linear rows. w unused.</summary>
        public float4 lambdaLin;

        /// <summary>Multiplier of the angular rows. w unused.</summary>
        public float4 lambdaAng;

        /// <summary>x = linear fracture threshold of a joint (norm of the linear multiplier; read only with <see cref="ConstraintFlags.FracturableLinear"/>), else zero: in the <see cref="FractureRecord"/> on the GPU. yz zero. w = the mark of the fused dual update (GPU only; 0 on the CPU): in c0Ang.w on the GPU.</summary>
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
