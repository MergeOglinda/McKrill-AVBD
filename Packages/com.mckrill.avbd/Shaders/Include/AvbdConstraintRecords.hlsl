#ifndef MCKRILL_AVBD_CONSTRAINT_RECORDS_INCLUDED
#define MCKRILL_AVBD_CONSTRAINT_RECORDS_INCLUDED

#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdCommon.hlsl"

// Constraint (joint and spring) records. Field-by-field mirror of Runtime/Layout/ConstraintRecords.cs.
// Built from float4 and uint4 words only (packing rules P1 and P2).

// Flag bits (rule P7), low 24 bits of ident.z.
#define AVBD_CONSTRAINT_FLAG_HARD_LIN    (1u << 0) // linear part is hard (stabilized, has a multiplier). Stored stiffness is ignored
#define AVBD_CONSTRAINT_FLAG_HARD_ANG    (1u << 1) // angular part is hard. Stored stiffness is ignored
#define AVBD_CONSTRAINT_FLAG_FRACTURABLE (1u << 2) // breaks above the stored fracture threshold
#define AVBD_CONSTRAINT_FLAG_BROKEN      (1u << 3) // fractured. Owner: D

#define AVBD_CONSTRAINT_FLAGS_MASK 0x00FFFFFFu
#define AVBD_CONSTRAINT_KIND_SHIFT 24

#define AVBD_CONSTRAINT_KIND_JOINT  0u
#define AVBD_CONSTRAINT_KIND_SPRING 1u

// 160 B.
struct ConstraintRecord
{
    uint4  ident;      // x = body A (AVBD_INVALID = world), y = body B, z = flags | kind << 24, w unused
    float4 anchorA;    // xyz = anchor on A, w = linear stiffness (joint) / stiffness (spring)
    float4 anchorB;    // xyz = anchor on B, w = angular stiffness (joint) / rest length (spring)
    float4 c0Lin;      // xyz = linear constraint at the step start, w = torque arm
    float4 c0Ang;      // xyz = angular constraint at the step start, w = fracture threshold
    float4 penaltyLin; // xyz per-axis penalty
    float4 penaltyAng;
    float4 lambdaLin;  // xyz multiplier
    float4 lambdaAng;
    float4 reserved;   // zero
};

#endif // MCKRILL_AVBD_CONSTRAINT_RECORDS_INCLUDED
