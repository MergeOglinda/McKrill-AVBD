#ifndef MCKRILL_AVBD_CONSTRAINT_RECORDS_INCLUDED
#define MCKRILL_AVBD_CONSTRAINT_RECORDS_INCLUDED

#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdCommon.hlsl"

// Constraint (joint and spring) records. Field-by-field mirror of Runtime/Layout/ConstraintRecords.cs.
// Built from float4 and uint4 words only (packing rules P1 and P2).

// Flag bits (rule P7), low 24 bits of ident.z.
#define AVBD_CONSTRAINT_FLAG_HARD_LIN    (1u << 0) // linear part is hard (stabilized, has a multiplier). Stored stiffness is ignored
#define AVBD_CONSTRAINT_FLAG_HARD_ANG    (1u << 1) // angular part is hard. Stored stiffness is ignored
#define AVBD_CONSTRAINT_FLAG_FRACTURABLE (1u << 2) // breaks above the stored fracture threshold of the angular multiplier (the FractureRecord of the joint)
#define AVBD_CONSTRAINT_FLAG_BROKEN      (1u << 3) // fractured. Owner: D
#define AVBD_CONSTRAINT_FLAG_PLASTIC     (1u << 5) // plastic angular row: its PlasticRecord is slot ident.w (AvbdPlasticRecords.hlsl). Owner: creation
#define AVBD_CONSTRAINT_FLAG_BEND_BREAK  (1u << 6) // a plastic joint breaks when it has bent past the limit. Owner: creation
#define AVBD_CONSTRAINT_FLAG_FRACTURABLE_LIN (1u << 7) // breaks above the stored fracture threshold of the linear multiplier (the FractureRecord of the joint). Needs HARD_LIN. Owner: creation
#define AVBD_CONSTRAINT_FLAG_PLASTIC_LIN (1u << 8) // plastic linear row: the PlasticRecord is slot ident.w too. Needs HARD_LIN. Owner: creation
#define AVBD_CONSTRAINT_FLAG_STRETCH_BREAK (1u << 9) // a linearly plastic joint breaks when its anchor on A has moved past the limit. Owner: creation
// Bit 4 is the CPU oracle's stick flag of a contact record. On the GPU the stick bits are in the pair record of the manifold (AvbdContactRecords.hlsl).

#define AVBD_CONSTRAINT_FLAGS_MASK 0x00FFFFFFu
#define AVBD_CONSTRAINT_KIND_SHIFT 24

#define AVBD_CONSTRAINT_KIND_JOINT  0u
#define AVBD_CONSTRAINT_KIND_SPRING 1u
// Kind 2 is the CPU oracle's contact record. The GPU stores contacts as manifolds (pair record + 64 B points, AvbdContactRecords.hlsl), not as
// constraint records.

// 128 B: a joint or a spring. Eight 16 B words with no padding: the primal sweep reads this record twice per iteration for every joint (once for each
// of its bodies), so its size is most of what the sweep streams from memory, and 128 B is four 32 B sectors and one cache line, aligned. The CPU oracle's
// ConstraintRecord (Runtime/Layout/ConstraintRecords.cs, 160 B) is the same information in the form that is easy to compute with; ConstraintRecordCodec
// packs it into this one: the fracture thresholds go to the FractureRecord below (only the dual update reads them), the mark of the fused dual update takes
// the unused w of c0Ang, and the penalties and multipliers (12 floats) take the last three words, with no padding between them. Read them with the
// accessors below, not as fields.
struct ConstraintRecord
{
    uint4  ident;   // x = body A (AVBD_INVALID = world), y = body B, z = flags | kind << 24, w = plastic slot (only with the PLASTIC or PLASTIC_LIN flag)
    float4 anchorA; // xyz = anchor on A (stage D moves it when a joint with PLASTIC_LIN yields), w = linear stiffness (joint) / stiffness (spring)
    float4 anchorB; // xyz = anchor on B, w = angular stiffness (joint) / rest length (spring)
    float4 c0Lin;   // xyz = linear constraint at the step start, w = torque arm
    float4 c0Ang;   // xyz = angular constraint at the step start, w = mark of the fused dual update (the id of a sweep as the bits of a float, 0 = none)
    float4 state0;  // penaltyLin.xyz, penaltyAng.x
    float4 state1;  // penaltyAng.yz, lambdaLin.xy
    float4 state2;  // lambdaLin.z, lambdaAng.xyz
};

float3 AvbdPenaltyLin(ConstraintRecord c) { return c.state0.xyz; }
float3 AvbdPenaltyAng(ConstraintRecord c) { return float3(c.state0.w, c.state1.x, c.state1.y); }
float3 AvbdLambdaLin(ConstraintRecord c) { return float3(c.state1.z, c.state1.w, c.state2.x); }
float3 AvbdLambdaAng(ConstraintRecord c) { return c.state2.yzw; }

// Sets the four state vectors (the penalties and multipliers of the linear and angular rows) together: they share the last three words.
void AvbdSetConstraintState(inout ConstraintRecord c, float3 penaltyLin, float3 penaltyAng, float3 lambdaLin, float3 lambdaAng)
{
    c.state0 = float4(penaltyLin, penaltyAng.x);
    c.state1 = float4(penaltyAng.yz, lambdaLin.xy);
    c.state2 = float4(lambdaLin.z, lambdaAng);
}

// 32 B: the fracture thresholds of four joints, in an array of its own (_Fracture) that only the dual update reads, and only for a joint with the
// FRACTURABLE or FRACTURABLE_LIN flag. Joint j is lane j & 3 of record j >> 2: a holds the joints 0 and 1 of the record as (linear, angular, linear, angular),
// b the joints 2 and 3. Written once, at creation.
struct FractureRecord
{
    float4 a;
    float4 b;
};

#endif // MCKRILL_AVBD_CONSTRAINT_RECORDS_INCLUDED
