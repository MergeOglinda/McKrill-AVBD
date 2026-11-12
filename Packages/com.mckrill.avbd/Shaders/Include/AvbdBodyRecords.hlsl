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
    float4 linear;
    float4 angular;
    float4 prevLinear; // last step's linear velocity, for the adaptive warm start
};

// 16 B, cold. xyz = full box widths, w = friction.
struct BodyShapeRecord
{
    float4 sizeFriction;
};

// 32 B, cold.
struct BodyMetaRecord
{
    uint4 ident; // x = flags | kind << 24, y = group mask, z = appearance id, w unused
    uint4 state; // x = graph color, y = sleep counter, zw unused
};

// 16 B, hot. Range in the constraint list.
struct BodyRangeRecord
{
    uint4 range; // x = offset, y = count, zw unused
};

// Flags word of a free slot of the given kind.
uint AvbdFreeFlagsWord(uint kindBits)
{
    return (kindBits & ~AVBD_BODY_FLAGS_MASK) | AVBD_BODY_FLAG_FREE;
}

#endif // MCKRILL_AVBD_BODY_RECORDS_INCLUDED
