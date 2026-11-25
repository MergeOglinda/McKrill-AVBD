#ifndef MCKRILL_AVBD_PLASTIC_RECORDS_INCLUDED
#define MCKRILL_AVBD_PLASTIC_RECORDS_INCLUDED

#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdCommon.hlsl"
#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdQuaternion.hlsl"

// Plastic joint records. Field-by-field mirror of Runtime/Layout/PlasticRecords.cs. Built from float4 words only (packing rules P1 and P2).
// The kernels that read them are compiled with AVBD_PLASTIC_JOINTS defined (a second variant of Dual, ConstraintInit and Primal).

// 64 B: the extra state of a joint with AVBD_CONSTRAINT_FLAG_PLASTIC or AVBD_CONSTRAINT_FLAG_PLASTIC_LIN, found at its ident.w.
// The first 32 B (one sector) are what the primal and dual stages read of a plastic joint every iteration: rest and limits. The second 32 B (restStart,
// reserved) are read only in the step in which a joint yields or breaks.
struct PlasticRecord
{
    float4 rest;      // rest rotation (unit quaternion): body B is at rest when its rotation is rotA * rest. Identity at creation. Owner: D
    float4 limits;    // x = angular yield threshold (norm of the angular multiplier, with PLASTIC), y = bend limit in radians (with BEND_BREAK),
                      // z = linear yield threshold (norm of the linear multiplier, with PLASTIC_LIN), w = stretch limit in meters (with STRETCH_BREAK)
    float4 restStart; // the rest rotation at creation: the origin of the bend
    float4 reserved;  // xyz = the anchor on body A at creation: the origin the stretch is measured from (with PLASTIC_LIN)
};

// The angular constraint value of a joint, before the stabilization: the difference between the rotation of body B and the rest pose
// rotA * rest. With rest = identity it is exactly AvbdQuatDifference(rotA, rotB). rotA is the identity for a joint to the world.
float3 AvbdJointAngularError(float4 rotA, float4 rotB, float4 rest)
{
    return AvbdQuatDifference(AvbdQuatMul(rotA, rest), rotB);
}

// The rest rotation that makes the current relative rotation the rest pose: conj(rotA) * rotB, so that rotA * rest = rotB.
float4 AvbdJointRestFromPose(float4 rotA, float4 rotB)
{
    return normalize(AvbdQuatMul(AvbdQuatConjugate(rotA), rotB));
}

// How far the rest rotation has moved from where it started: the angle of conj(restStart) * rest, in [0, pi].
float AvbdJointBend(float4 restStart, float4 rest)
{
    float4 d = AvbdQuatMul(AvbdQuatConjugate(restStart), rest);
    return 2.0 * atan2(length(d.xyz), abs(d.w));
}

#endif // MCKRILL_AVBD_PLASTIC_RECORDS_INCLUDED
