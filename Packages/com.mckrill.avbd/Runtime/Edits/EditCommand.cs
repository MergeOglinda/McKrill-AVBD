using System;
using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>Kind of an edit command. Mirrors AVBD_EDIT_* in AvbdEditCommand.hlsl.</summary>
    internal enum EditKind : uint
    {
        None = 0,

        /// <summary>Writes every persistent field of a free or live slot.</summary>
        Spawn = 1,

        /// <summary>Marks the slot free.</summary>
        Remove = 2,

        /// <summary>Overwrites the fields named in the field mask of a live slot.</summary>
        Set = 3,
    }

    /// <summary>Fields a Set command changes. Mirrors AVBD_EDIT_FIELD_* in AvbdEditCommand.hlsl.</summary>
    [Flags]
    internal enum EditFields : uint
    {
        None = 0,

        /// <summary>Position and rotation. Also resets the step-start pose, so a teleport leaves no motion streak.</summary>
        Pose = 1u << 0,

        /// <summary>Linear and angular velocity. Also resets the previous linear velocity, so the adaptive warm start sees no acceleration.</summary>
        Velocity = 1u << 1,

        /// <summary>Box size and the mass properties derived from size and density.</summary>
        Shape = 1u << 2,

        Friction = 1u << 3,
        GroupMask = 1u << 4,
        Appearance = 1u << 5,

        /// <summary>The Kinematic flag bit, taken from the flags word of the command.</summary>
        Kinematic = 1u << 6,

        All = 0x7Fu,
    }

    /// <summary>
    /// One edit command, 128 B: a header word and seven payload words (rule P1). A batch holds at most one command
    /// per slot, so the kernel can apply commands in parallel without ordering. Mirrors EditCommand in AvbdEditCommand.hlsl.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct EditCommand
    {
        /// <summary>x = EditKind, y = slot, z = EditFields, w unused.</summary>
        public uint4 header;

        /// <summary>xyz = position, w unused.</summary>
        public float4 position;

        public float4 rotation;

        /// <summary>xyz = linear velocity, w unused.</summary>
        public float4 linearVel;

        public float4 angularVel;

        /// <summary>xyz = principal moments, w = mass.</summary>
        public float4 momentMass;

        /// <summary>xyz = box size, w = friction.</summary>
        public float4 sizeFriction;

        /// <summary>x = flags (low 24 bits) and shape kind (high 8 bits), y = group mask, z = appearance id, w unused.</summary>
        public uint4 meta;

        public EditKind Kind => (EditKind)header.x;

        public int Slot => (int)header.y;

        public EditFields Fields => (EditFields)header.z;
    }

    internal static class EditCommandStride
    {
        public const int Size = 128;
    }
}
