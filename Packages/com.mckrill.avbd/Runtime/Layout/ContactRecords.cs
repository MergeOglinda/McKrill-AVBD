using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>Strides of the contact arrays in bytes. Tests assert that the record structs have exactly these sizes.</summary>
    internal static class ContactRecordStrides
    {
        /// <summary>Two 16 B words: what the two bodies of a manifold share.</summary>
        public const int Pair = 32;

        /// <summary>Four 16 B words: one contact point, a multiple of 32 B (rule P3). A manifold's points are consecutive, so a stage reads them in order.</summary>
        public const int Point = 64;
    }

    /// <summary>
    /// One manifold of the pool: the pair of bodies and what all its contact points share. Mirror of ContactPairRecord in AvbdContactRecords.hlsl.
    /// A pair that has no manifold (a free slot) holds AVBD_INVALID in both bodies and no points.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct ContactPairRecord
    {
        /// <summary>x = body A (the higher slot), y = body B, z = number of points, w = stick bits (bit k: point k held static friction in the last dual update).</summary>
        public uint4 ident;

        /// <summary>xyz = contact normal pointing from body B to body A, w = friction coefficient.</summary>
        public float4 normalFriction;
    }

    /// <summary>
    /// One contact point: 16 floats, of which 15 are values and one is the feature key (the bits of an int). Mirror of ContactPointRecord in
    /// AvbdContactRecords.hlsl; use <see cref="ContactCodec"/> to read and build one.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct ContactPointRecord
    {
        /// <summary>xyz = anchor on body A in body space, w = feature key (the bits of an int).</summary>
        public float4 anchorA;

        /// <summary>xyz = anchor on body B in body space, w = C0.x.</summary>
        public float4 anchorB;

        /// <summary>x = C0.y, y = C0.z, z = penalty.x, w = penalty.y.</summary>
        public float4 c0Penalty;

        /// <summary>x = penalty.z, y = lambda.x, z = lambda.y, w = lambda.z.</summary>
        public float4 penaltyLambda;
    }

    /// <summary>
    /// Reads and builds <see cref="ContactPointRecord"/>s, and converts a manifold to and from the per-contact <see cref="ConstraintRecord"/> the CPU oracle
    /// (<c>CpuContacts</c>) stores, which is what the tests compare the GPU with.
    /// </summary>
    internal static class ContactCodec
    {
        public static float3 RA(in ContactPointRecord p) => p.anchorA.xyz;
        public static float3 RB(in ContactPointRecord p) => p.anchorB.xyz;
        public static int Key(in ContactPointRecord p) => math.asint(p.anchorA.w);
        public static float3 C0(in ContactPointRecord p) => new float3(p.anchorB.w, p.c0Penalty.x, p.c0Penalty.y);
        public static float3 Penalty(in ContactPointRecord p) => new float3(p.c0Penalty.z, p.c0Penalty.w, p.penaltyLambda.x);
        public static float3 Lambda(in ContactPointRecord p) => p.penaltyLambda.yzw;

        public static ContactPointRecord Make(float3 rA, float3 rB, int key, float3 c0, float3 penalty, float3 lambda)
        {
            return new ContactPointRecord
            {
                anchorA = new float4(rA, math.asfloat(key)),
                anchorB = new float4(rB, c0.x),
                c0Penalty = new float4(c0.y, c0.z, penalty.x, penalty.y),
                penaltyLambda = new float4(penalty.z, lambda),
            };
        }

        public static bool Stick(in ContactPairRecord pair, int k) => (pair.ident.w & (1u << k)) != 0u;

        /// <summary>The stick bits of a pair: <c>ident.w</c> without the mark of the fused dual update.</summary>
        public static uint StickBits(in ContactPairRecord pair) => pair.ident.w & StickMask;

        /// <summary>The id of the sweep that ran the dual update of this pair (<c>AVBD_FUSED_DUAL</c>), 0 if no sweep did. It is 0 outside a step.</summary>
        public static uint Mark(in ContactPairRecord pair) => pair.ident.w >> ContactLayout.StickBits;

        /// <summary><c>ident.w</c> of a pair with these stick bits and this mark.</summary>
        public static uint PackIdentW(uint stickBits, uint mark) => (mark << ContactLayout.StickBits) | (stickBits & StickMask);

        public const uint StickMask = (1u << ContactLayout.StickBits) - 1u;
        public const uint MarkMask = 0xFFFFFFu;

        /// <summary>The contact in the form the CPU oracle stores: one <see cref="ConstraintRecord"/> per point.</summary>
        public static ConstraintRecord ToRecord(in ContactPairRecord pair, in ContactPointRecord point, int k)
        {
            return new ConstraintRecord
            {
                ident = new uint4(pair.ident.x, pair.ident.y,
                    ConstraintFlagLayout.Pack(Stick(pair, k) ? ConstraintFlags.Stick : ConstraintFlags.None, ConstraintKind.Contact), (uint)Key(point)),
                anchorA = new float4(RA(point), pair.normalFriction.w),
                anchorB = new float4(RB(point), 0f),
                c0Lin = new float4(C0(point), 0f),
                c0Ang = new float4(pair.normalFriction.xyz, 0f),
                penaltyLin = new float4(Penalty(point), 0f),
                lambdaLin = new float4(Lambda(point), 0f),
            };
        }

        /// <summary>The point of a CPU-oracle contact record (the normal and the friction belong to the pair, see <see cref="ToRecord"/>).</summary>
        public static ContactPointRecord FromRecord(in ConstraintRecord r)
        {
            return Make(r.anchorA.xyz, r.anchorB.xyz, (int)r.ident.w, r.c0Lin.xyz, r.penaltyLin.xyz, r.lambdaLin.xyz);
        }
    }
}
