#ifndef MCKRILL_AVBD_BODY_RECORDS_INCLUDED
#define MCKRILL_AVBD_BODY_RECORDS_INCLUDED

#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdCommon.hlsl"

// Per-body array records. Field-by-field mirror of Runtime/Layout/BodyRecords.cs.
// Every record is built from float4 and uint4 words only (packing rules P1 and P2).

// Flag bits (rule P7). Each bit has one owning writer stage.
#define AVBD_BODY_FLAG_FREE      (1u << 0) // slot holds no body. Owner: E
#define AVBD_BODY_FLAG_KINEMATIC (1u << 1) // moved by the user, not solved. Owner: E
#define AVBD_BODY_FLAG_ASLEEP    (1u << 2) // resting. Owner: Z

#define AVBD_BODY_FLAGS_MASK 0x00FFFFFFu
#define AVBD_BODY_KIND_SHIFT 24

#define AVBD_KIND_BOX 0u

// 64 B, hot. Pose and step-start pose.
struct BodyStateRecord
{
    float4 position;      // xyz = center of mass, w unused
    float4 rotation;      // unit quaternion (x, y, z, w)
    float4 startPosition; // pose at the start of the step
    float4 startRotation;
};

// 32 B, hot. A copy of the current pose (position and rotation) of BodyStateRecord, for the readers that only need that half of it: the primal and dual
// stages read the pose of the neighbors of a joint or spring (and the primal sweep that of the body it solves), the broadphase, Predict, the ray cast and the
// group kernels read the pose of a body, and a 32 B record puts twice as many bodies in a cache line as the 64 B state record. Every
// kernel that writes BodyStateRecord.position or .rotation writes this record too (stage E, Predict, the group mover and the primal sweep); nothing
// else writes it, so the two never differ. Free slots hold the same values as the state record.
struct BodyPoseRecord
{
    float4 position; // xyz = center of mass, w unused
    float4 rotation; // unit quaternion (x, y, z, w)
};

// 32 B, hot. Inertial target. Written by stage P only.
struct BodyInertialRecord
{
    float4 position;
    float4 rotation;
};

// 16 B, hot. xyz = principal moments, w = mass. Mass 0 is a static body.
struct BodyMassRecord
{
    float4 momentMass;
};

// 48 B, cold.
struct BodyMotionRecord
{
    float4 linearVel;
    float4 angularVel;
    float4 prevLinearVel; // last step's linear velocity, for the adaptive warm start
};

// 16 B, cold. xyz = full box widths, w = friction.
struct BodyShapeRecord
{
    float4 sizeFriction;
};

// 32 B, cold. The flags word and the graph color are not in this record: they are the arrays _BodyFlags (flags | kind << 24, see the AVBD_BODY_FLAG_*
// bits above) and _BodyColor (the graph color, AVBD_INVALID for a body that is not solved), one uint per body, because the hot kernels ask every body
// whether it is live and solved, and the coloring reads the color of random neighbors: 4 B in a sector of their own and not in a 32 B record.
// Declare them as StructuredBuffer<uint> (RWStructuredBuffer<uint> where the kernel writes them). ident.x and state.x of this record are always 0.
struct BodyMetaRecord
{
    uint4 ident; // x = unused (the flags word is _BodyFlags), y = group mask, z = appearance id, w unused
    uint4 state; // x = unused (the graph color is _BodyColor), y = sleep counter, zw unused
};

// 16 B, hot. Range in the constraint list.
struct BodyRangeRecord
{
    uint4 range; // x = offset, y = count, z = contact entries (the first z of the count, written by stage G), w unused
};

// 32 B, hot. The primal sweep's view of a body, derived once per step (AvbdSolver.DispatchSolveHeader, after ConstraintInit) from _BodyFlags, _BodyMass and
// _BodyRange, so that the sweep reads one sector per body for what it used to read from three arrays (and three sectors). Never written by anything else.
struct BodySolveRecord
{
    float4 momentMass; // xyz = principal moments of inertia, w = mass (as in BodyMassRecord)
    uint4 info;        // x = flags word (as in _BodyFlags), y = offset and z = count of the body's entries in the constraint list (as in BodyRangeRecord), w unused
};

// Flags word of a free slot of the given kind.
uint AvbdFreeFlagsWord(uint kindBits)
{
    return (kindBits & ~AVBD_BODY_FLAGS_MASK) | AVBD_BODY_FLAG_FREE;
}

#endif // MCKRILL_AVBD_BODY_RECORDS_INCLUDED
