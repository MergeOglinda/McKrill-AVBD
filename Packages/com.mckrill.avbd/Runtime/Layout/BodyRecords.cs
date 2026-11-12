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
        public const int Inertial = 32;
        public const int Mass = 16;
        public const int Motion = 48;
        public const int Shape = 16;
        public const int Meta = 32;
        public const int Range = 16;

        public const int Total = State + Inertial + Mass + Motion + Shape + Meta + Range;
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

    /// <summary>Cold. Identity and state words.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct BodyMetaRecord
    {
        /// <summary>x = flags (low 24 bits) and shape kind (high 8 bits), y = group mask, z = appearance id, w unused.</summary>
        public uint4 ident;

        /// <summary>x = graph color, y = sleep counter, zw unused. Written by stages C and Z.</summary>
        public uint4 state;
    }

    /// <summary>Hot. Range of this body's entries in the constraint list. Written by stage G.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct BodyRangeRecord
    {
        /// <summary>x = offset, y = count, zw unused.</summary>
        public uint4 range;
    }
}
