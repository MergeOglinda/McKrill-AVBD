using System;
using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>Constants shared by the C# mirrors and the HLSL includes (design doc, packing rules P6 and P9).</summary>
    internal static class AvbdConst
    {
        /// <summary>No body, no contact, empty slot or no color (rule P6). Mirrors AVBD_INVALID in AvbdCommon.hlsl.</summary>
        public const uint Invalid = 0xFFFFFFFFu;

        /// <summary>Thread group size of 1D kernels. Mirrors AVBD_GROUP_SIZE in AvbdCommon.hlsl.</summary>
        public const int GroupSize = 64;
    }

    /// <summary>
    /// Body flag bits (rule P7). Each bit has exactly one owning writer stage. Mirrors AVBD_BODY_FLAG_* in AvbdBodyRecords.hlsl.
    /// </summary>
    [Flags]
    internal enum BodyFlags : uint
    {
        None = 0,

        /// <summary>The slot holds no body. Owner: stage E (spawn clears it, remove sets it).</summary>
        Free = 1u << 0,

        /// <summary>Moved by the user, not solved. Owner: stage E.</summary>
        Kinematic = 1u << 1,

        /// <summary>Resting, skipped by the solver. Owner: stage Z (never written by stage E).</summary>
        Asleep = 1u << 2,
    }

    /// <summary>Shape type, stored in the high 8 bits of the flags word (rule P7).</summary>
    internal enum ShapeKind : uint
    {
        Box = 0,

        /// <summary>
        /// A ball. The shape record holds its diameter in x, y and z (so code that only needs a size still sees a sensible bounding box), the
        /// inertia is that of a solid sphere, and the rotation does not change what it collides with, only where its contact anchors are.
        /// </summary>
        Sphere = 1,
    }

    /// <summary>Packing of the flags word: low 24 bits are flags, high 8 bits are the shape kind.</summary>
    internal static class BodyFlagLayout
    {
        public const int KindShift = 24;
        public const uint FlagsMask = 0x00FFFFFFu;

        /// <summary>Bits stage E may change after spawn: only Kinematic. Free changes only through spawn and remove.</summary>
        public const uint UserSettable = (uint)BodyFlags.Kinematic;

        public static uint Pack(BodyFlags flags, ShapeKind kind)
        {
            return ((uint)kind << KindShift) | ((uint)flags & FlagsMask);
        }

        public static BodyFlags FlagsOf(uint packed) => (BodyFlags)(packed & FlagsMask);

        public static ShapeKind KindOf(uint packed) => (ShapeKind)(packed >> KindShift);
    }

    /// <summary>Strides of the per-body arrays in bytes. Tests assert that each record struct has exactly this size.</summary>
    internal static class BodyRecordStrides
    {
        public const int State = 64;
        public const int Pose = 32;
        public const int Inertial = 32;
        public const int Mass = 16;
        public const int Motion = 48;
        public const int Shape = 16;
        public const int Meta = 32;
        public const int Range = 16;

        /// <summary>The flags word of a body (<see cref="BodyFlagLayout"/>), one uint per body: an array of its own, so that a kernel that only asks whether a body is live and solved reads 4 B and not a 32 B meta record.</summary>
        public const int Flags = 4;

        /// <summary>The graph color of a body, one uint per body, for the same reason. Written by stage C (and cleared by stage E).</summary>
        public const int Color = 4;

        /// <summary>The primal sweep's view of a body (<see cref="BodySolveRecord"/>): moments and mass, flags, and the range in the constraint list in one 32 B sector. Derived once per step.</summary>
        public const int Solve = 32;

        public const int Total = State + Pose + Inertial + Mass + Motion + Shape + Meta + Range + Flags + Color + Solve;
        public const int Largest = 64;
    }

    // Every record is built from float4 and uint4 words only (rules P1 and P2). Unused lanes are zero.
    // Field-by-field mirrors of Shaders/Include/AvbdBodyRecords.hlsl.

    /// <summary>Hot. Pose and step-start pose (read by V and D for the body and for every partner of a contact).</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct BodyStateRecord
    {
        /// <summary>xyz = position of the center of mass, w unused.</summary>
        public float4 position;

        /// <summary>Unit quaternion (x, y, z, w).</summary>
        public float4 rotation;

        /// <summary>Pose at the start of the step (x-). xyz = position, w unused.</summary>
        public float4 startPosition;

        public float4 startRotation;
    }

    /// <summary>
    /// Hot. A copy of the current pose of <see cref="BodyStateRecord"/> (position and rotation) for the readers that need nothing else of a body: the primal
    /// and dual stages read the pose of the neighbors of a joint or spring, and 32 B records put twice as many bodies in a cache line as the 64 B state
    /// records. Every kernel that writes the pose of the state record writes this too, so the two never differ.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct BodyPoseRecord
    {
        /// <summary>xyz = position of the center of mass, w unused.</summary>
        public float4 position;

        /// <summary>Unit quaternion (x, y, z, w).</summary>
        public float4 rotation;
    }

    /// <summary>Hot. Inertial target of the step. Written by stage P only; stage E never touches it.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct BodyInertialRecord
    {
        public float4 position;
        public float4 rotation;
    }

    /// <summary>Hot. xyz = principal moments of inertia, w = mass. Mass 0 means a static body.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct BodyMassRecord
    {
        public float4 momentMass;
    }

    /// <summary>Cold. Velocities, and last step's linear velocity for the adaptive warm start (design doc finding 9).</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct BodyMotionRecord
    {
        public float4 linearVel;
        public float4 angularVel;
        public float4 prevLinearVel;
    }

    /// <summary>Cold. xyz = full box widths, w = friction.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct BodyShapeRecord
    {
        public float4 sizeFriction;
    }

    /// <summary>
    /// Cold. Identity and state words. The flags word and the graph color are not here: they are the arrays <c>_BodyFlags</c> and <c>_BodyColor</c>
    /// (<see cref="BodyRecordStrides.Flags"/>, <see cref="BodyRecordStrides.Color"/>), one uint per body, which the hot kernels read. The words
    /// <c>ident.x</c> and <c>state.x</c> are kept so that the record keeps its size and the names of the others; they are always 0.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct BodyMetaRecord
    {
        /// <summary>x = unused (always 0: the flags word is in <c>_BodyFlags</c>), y = group mask, z = appearance id, w unused.</summary>
        public uint4 ident;

        /// <summary>x = unused (always 0: the graph color is in <c>_BodyColor</c>), y = sleep counter, zw unused. Written by stage Z.</summary>
        public uint4 state;
    }

    /// <summary>
    /// Hot. What the primal sweep needs of a body besides its pose and inertial target, in one sector: a copy of <see cref="BodyMassRecord"/>, the flags
    /// word and the offset and count of <see cref="BodyRangeRecord"/>. Derived once per step, after stage D's init and so after the edits, the lists and the
    /// colors of that step; nothing writes it between the sweeps. A mirror, like <see cref="BodyPoseRecord"/>: the three arrays stay the source.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct BodySolveRecord
    {
        /// <summary>xyz = principal moments of inertia, w = mass.</summary>
        public float4 momentMass;

        /// <summary>x = flags word, y = list offset, z = list count, w unused.</summary>
        public uint4 info;
    }

    /// <summary>Hot. Range of this body's entries in the constraint list. Written by stage G.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct BodyRangeRecord
    {
        /// <summary>x = offset, y = count, zw unused.</summary>
        public uint4 range;
    }
}
