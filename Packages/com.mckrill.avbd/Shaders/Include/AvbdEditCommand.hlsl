#ifndef MCKRILL_AVBD_EDIT_COMMAND_INCLUDED
#define MCKRILL_AVBD_EDIT_COMMAND_INCLUDED

#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdCommon.hlsl"

// Edit command, 128 B. Mirror of Runtime/Edits/EditCommand.cs.
// A batch holds at most one command per slot, so commands can be applied in parallel.

#define AVBD_EDIT_SPAWN  1u // writes every persistent field
#define AVBD_EDIT_REMOVE 2u // marks the slot free
#define AVBD_EDIT_SET    3u // overwrites the fields named in the field mask

#define AVBD_EDIT_FIELD_POSE       (1u << 0)
#define AVBD_EDIT_FIELD_VELOCITY   (1u << 1)
#define AVBD_EDIT_FIELD_SHAPE      (1u << 2)
#define AVBD_EDIT_FIELD_FRICTION   (1u << 3)
#define AVBD_EDIT_FIELD_GROUPMASK  (1u << 4)
#define AVBD_EDIT_FIELD_APPEARANCE (1u << 5)
#define AVBD_EDIT_FIELD_KINEMATIC  (1u << 6)

struct EditCommand
{
    uint4  header;       // x = kind, y = slot, z = field mask, w unused
    float4 position;     // xyz = position
    float4 rotation;
    float4 linearVel;       // xyz = linear velocity
    float4 angularVel;
    float4 momentMass;   // xyz = principal moments, w = mass
    float4 sizeFriction; // xyz = box size, w = friction
    uint4  meta;         // x = flags | kind << 24, y = group mask, z = appearance id, w unused
};

#endif // MCKRILL_AVBD_EDIT_COMMAND_INCLUDED
