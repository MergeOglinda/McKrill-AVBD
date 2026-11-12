using System.Collections.Generic;
using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle for stage E: the per-body arrays held in managed memory, with edit commands applied exactly as
    /// Shaders/Compute/AvbdBodyEdits.compute applies them. Tests replay the commands the GPU received on this
    /// class and require the GPU arrays to match it bit for bit. Any change to the kernel must be made here too.
    /// </summary>
    internal sealed class CpuBodyArrays
    {
        public readonly BodyStateRecord[] State;
        public readonly BodyInertialRecord[] Inertial;
        public readonly BodyMassRecord[] Mass;
        public readonly BodyMotionRecord[] Motion;
        public readonly BodyShapeRecord[] Shape;
        public readonly BodyMetaRecord[] Meta;
        public readonly BodyRangeRecord[] Range;

        public CpuBodyArrays(int capacity)
        {
            State = new BodyStateRecord[capacity];
            Inertial = new BodyInertialRecord[capacity];
            Mass = new BodyMassRecord[capacity];
            Motion = new BodyMotionRecord[capacity];
            Shape = new BodyShapeRecord[capacity];
            Meta = new BodyMetaRecord[capacity];
            Range = new BodyRangeRecord[capacity];

            for (int slot = 0; slot < capacity; slot++)
                WriteFreeSlot(slot);
        }

        public int Capacity => State.Length;

        public bool IsFree(int slot) => (Meta[slot].ident.x & (uint)BodyFlags.Free) != 0u;

        /// <summary>The ClearBodies kernel for one slot.</summary>
        public void WriteFreeSlot(int slot)
        {
            float4 identity = new float4(0f, 0f, 0f, 1f);

            State[slot] = new BodyStateRecord { position = float4.zero, rotation = identity, startPosition = float4.zero, startRotation = identity };
            Inertial[slot] = new BodyInertialRecord { position = float4.zero, rotation = identity };
            Mass[slot] = new BodyMassRecord { momentMass = float4.zero };
            Motion[slot] = new BodyMotionRecord { linearVel = float4.zero, angularVel = float4.zero, prevLinearVel = float4.zero };
            Shape[slot] = new BodyShapeRecord { sizeFriction = float4.zero };
            Meta[slot] = new BodyMetaRecord
            {
                ident = new uint4(FreeFlagsWord((uint)ShapeKind.Box << BodyFlagLayout.KindShift), 0u, AvbdConst.Invalid, 0u),
                state = new uint4(AvbdConst.Invalid, 0u, 0u, 0u),
            };
            Range[slot] = new BodyRangeRecord { range = uint4.zero };
        }

        public void Apply(IEnumerable<EditCommand> commands)
        {
            foreach (EditCommand c in commands)
                Apply(c);
        }

        /// <summary>The ApplyEdits kernel for one command.</summary>
        public void Apply(in EditCommand c)
        {
            int slot = c.Slot;
            if ((uint)slot >= (uint)Capacity)
                return;

            switch (c.Kind)
            {
                case EditKind.Spawn: ApplySpawn(slot, c); break;
                case EditKind.Remove: ApplyRemove(slot); break;
                case EditKind.Set: ApplySet(slot, c); break;
            }
        }

        void ApplySpawn(int slot, in EditCommand c)
        {
            State[slot] = new BodyStateRecord { position = c.position, rotation = c.rotation, startPosition = c.position, startRotation = c.rotation };
            Mass[slot] = new BodyMassRecord { momentMass = c.momentMass };
            Motion[slot] = new BodyMotionRecord { linearVel = c.linearVel, angularVel = c.angularVel, prevLinearVel = c.linearVel };
            Shape[slot] = new BodyShapeRecord { sizeFriction = c.sizeFriction };
            Meta[slot] = new BodyMetaRecord
            {
                ident = new uint4(c.meta.x & ~(uint)BodyFlags.Free, c.meta.y, c.meta.z, 0u),
                state = new uint4(AvbdConst.Invalid, 0u, 0u, 0u),
            };
            Range[slot] = new BodyRangeRecord { range = uint4.zero };
        }

        void ApplyRemove(int slot)
        {
            Mass[slot] = new BodyMassRecord { momentMass = float4.zero };
            Meta[slot] = new BodyMetaRecord
            {
                ident = new uint4(FreeFlagsWord(Meta[slot].ident.x), 0u, AvbdConst.Invalid, 0u),
                state = new uint4(AvbdConst.Invalid, 0u, 0u, 0u),
            };
            Range[slot] = new BodyRangeRecord { range = uint4.zero };
        }

        void ApplySet(int slot, in EditCommand c)
        {
            EditFields fields = c.Fields;

            if ((fields & EditFields.Pose) != 0)
            {
                BodyStateRecord s = State[slot];
                s.position = c.position;
                s.rotation = c.rotation;
                s.startPosition = c.position;
                s.startRotation = c.rotation;
                State[slot] = s;
            }

            if ((fields & EditFields.Velocity) != 0)
                Motion[slot] = new BodyMotionRecord { linearVel = c.linearVel, angularVel = c.angularVel, prevLinearVel = c.linearVel };

            if ((fields & (EditFields.Shape | EditFields.Friction)) != 0)
            {
                BodyShapeRecord sh = Shape[slot];
                if ((fields & EditFields.Shape) != 0)
                    sh.sizeFriction = new float4(c.sizeFriction.xyz, sh.sizeFriction.w);
                if ((fields & EditFields.Friction) != 0)
                    sh.sizeFriction = new float4(sh.sizeFriction.xyz, c.sizeFriction.w);
                Shape[slot] = sh;
            }

            if ((fields & EditFields.Shape) != 0)
                Mass[slot] = new BodyMassRecord { momentMass = c.momentMass };

            if ((fields & (EditFields.GroupMask | EditFields.Appearance | EditFields.Kinematic)) != 0)
            {
                BodyMetaRecord meta = Meta[slot];
                if ((fields & EditFields.GroupMask) != 0)
                    meta.ident.y = c.meta.y;
                if ((fields & EditFields.Appearance) != 0)
                    meta.ident.z = c.meta.z;
                if ((fields & EditFields.Kinematic) != 0)
                {
                    uint bit = (uint)BodyFlags.Kinematic;
                    meta.ident.x = (meta.ident.x & ~bit) | (c.meta.x & bit);
                }
                Meta[slot] = meta;
            }
        }

        // AvbdFreeFlagsWord in AvbdBodyRecords.hlsl.
        static uint FreeFlagsWord(uint kindBits)
        {
            return (kindBits & ~BodyFlagLayout.FlagsMask) | (uint)BodyFlags.Free;
        }
    }
}
