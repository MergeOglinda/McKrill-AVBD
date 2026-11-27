#ifndef MCKRILL_AVBD_CONTACT_RECORDS_INCLUDED
#define MCKRILL_AVBD_CONTACT_RECORDS_INCLUDED

#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdCommon.hlsl"

// Contact storage of the manifold pool (stage N). Field-by-field mirror of Runtime/Layout/ContactRecords.cs; built from float4 and uint4 words only
// (packing rules P1 and P2). A manifold slot s has one 32 B pair record (_ContactPairs[s]) and room for AVBD_MAX_CONTACTS 64 B points at
// _ContactPoints[s * AVBD_MAX_CONTACTS + k], k < the pair's count. What the two bodies, the normal and the friction share is stored once per pair, and the
// primal and dual stages read the pair, then loop over its points.

// The stick flag of every point of a pair: bit k of ident.w (Owner: stage D; stage N only carries it over to decide which anchors to keep).
#define AVBD_CONTACT_STICK_MASK 0xFFu

// With AVBD_FUSED_CONTACT_DUAL (AvbdPrimal.compute) the sweeps run the dual update of a manifold themselves and leave a mark in the bits above the stick
// bits of ident.w: the id of the sweep that updated it, folded into 1 .. 0xFFFFFF (AvbdContactSweepMark). The mark is read by the next sweep, so that the
// second body of the manifold does not update it again, and cleared by the dual dispatch after the last sweep (DualContact with _ClearMarks) and by the
// narrow phase, which writes the whole word. A record that leaves a step holds no mark.
#define AVBD_CONTACT_MARK_SHIFT 8u

// The mark of the sweep with this id (above 0). Consecutive ids give different marks, and never 0, which is the mark of a manifold nothing has updated.
uint AvbdContactSweepMark(uint sweepId)
{
    return (sweepId - 1u) % 0xFFFFFFu + 1u;
}

// 32 B.
struct ContactPairRecord
{
    uint4  ident;          // x = body A (the higher slot), y = body B (AVBD_INVALID in both: a free slot), z = number of points, w = stick bits
    float4 normalFriction; // xyz = contact normal pointing from B to A, w = friction coefficient
};

// 64 B: 16 floats, 15 of them the values of a point and one the feature key. The tangents are derived from the normal.
struct ContactPointRecord
{
    float4 anchorA;      // xyz = anchor on A in body space, w = feature key (the bits of an int)
    float4 anchorB;      // xyz = anchor on B in body space, w = C0.x
    float4 c0Penalty;    // x = C0.y, y = C0.z, z = penalty.x, w = penalty.y
    float4 penaltyLambda; // x = penalty.z, y = lambda.x, z = lambda.y, w = lambda.z
};

// C0, the constraint value at the start of the step in the contact basis (normal row first).
float3 ContactPointC0(ContactPointRecord p)
{
    return float3(p.anchorB.w, p.c0Penalty.x, p.c0Penalty.y);
}

// The penalty of the three rows (normal, tangent 1, tangent 2).
float3 ContactPointPenalty(ContactPointRecord p)
{
    return float3(p.c0Penalty.z, p.c0Penalty.w, p.penaltyLambda.x);
}

// The multiplier of the three rows.
float3 ContactPointLambda(ContactPointRecord p)
{
    return p.penaltyLambda.yzw;
}

int ContactPointKey(ContactPointRecord p)
{
    return asint(p.anchorA.w);
}

// A complete point: the anchors, the feature key, C0, the penalty and the multiplier.
ContactPointRecord MakeContactPoint(float3 rA, float3 rB, int key, float3 c0, float3 penalty, float3 lambda)
{
    ContactPointRecord p;
    p.anchorA = float4(rA, asfloat(key));
    p.anchorB = float4(rB, c0.x);
    p.c0Penalty = float4(c0.y, c0.z, penalty.x, penalty.y);
    p.penaltyLambda = float4(penalty.z, lambda);
    return p;
}

// A pair record that holds no manifold.
ContactPairRecord FreeContactPair()
{
    ContactPairRecord pair;
    pair.ident = uint4(AVBD_INVALID, AVBD_INVALID, 0u, 0u);
    pair.normalFriction = float4(0.0, 0.0, 0.0, 0.0);
    return pair;
}

#endif // MCKRILL_AVBD_CONTACT_RECORDS_INCLUDED
