// Force fields: forces that act on the bodies of a group, every step, in stage P (the predict kernel variants with AVBD_FORCES).
// A body belongs to a group through its group mask (BodyMetaRecord.ident.y, set with BodyDescriptor.groupMask or AvbdBodyBuffers.SetGroupMask): no
// per-body data is added. A field targets every body whose group mask has a bit in common with the field's mask.
// Mirrored by Reference/Stages/CpuForceFields.cs, which the tests compare against. Must match Runtime/Solver/ForceFieldRecord.cs.

#define AVBD_MAX_FORCE_FIELDS 16

// Kinds (ForceKind on the CPU). 0 is an empty slot.
#define AVBD_FORCE_NONE 0
#define AVBD_FORCE_CONSTANT 1
#define AVBD_FORCE_GRAVITY_SCALE 2
#define AVBD_FORCE_DRAG 3
#define AVBD_FORCE_WING 4
#define AVBD_FORCE_SPRING 5

// A spring field needs the centroid of its group. ForceFieldAccumulate sums, for each spring field, the offsets (position - target) of the bodies of its
// group as integers in 1/AVBD_FORCE_FIXED_POINT m (an integer sum does not depend on the order the threads add in), and counts the bodies.
#define AVBD_FORCE_FIXED_POINT 1024

// Flags in ident.z.
#define AVBD_FORCE_FLAG_BODY_SPACE 1 // the force of a constant field is in the body's frame, not the world's

// 64 B. One force field. Only float4 and uint4 members, like every record (layout rule P1).
struct ForceFieldRecord
{
    float4 value;        // constant: the force in N. drag and wing: the velocity of the air (wind) in m/s. spring: the target point
    float4 axis;         // wing: xyz = the normal of the wing in the body's frame (unit)
    float4 coefficients; // gravity scale: x = scale. drag: x = linear, y = quadratic. wing: x = lift slope, y = drag coefficient, z = half the air density times the area.
                         // spring: x = stiffness in N/m, y = damping ratio
    uint4  ident;        // x = kind, y = group mask of the bodies it acts on, z = flags
};

