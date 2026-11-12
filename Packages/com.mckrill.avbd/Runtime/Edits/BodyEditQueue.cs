using System;
using System.Collections.Generic;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>
    /// Collects body edits on the CPU and turns them into at most one GPU command per slot (stage E input).
    /// Pure bookkeeping, no GPU access.
    ///
    /// Slots are allocated immediately, so Spawn returns a usable handle before the GPU sees anything.
    /// Edits to the same slot inside one batch are merged by their final effect, compared with what the GPU holds:
    /// a body spawned and removed in one batch costs nothing, a spawn followed by property changes is one
    /// Spawn command, and a remove followed by a reuse of the slot is one Spawn command that overwrites everything.
    /// </summary>
    internal sealed class BodyEditQueue
    {
        sealed class Pending
        {
            public int slot;
            public bool startAlive;   // did the GPU slot hold a live body when this batch began
            public bool finalAlive;   // does it after this batch
            public bool respawned;    // was a Spawn issued for it in this batch (the payload then holds a full spawn)
            public EditCommand command;
        }

        readonly BodySlotAllocator allocator;
        readonly bool[] gpuAlive;
        readonly Dictionary<int, Pending> bySlot = new Dictionary<int, Pending>();
        readonly List<Pending> order = new List<Pending>();

        public BodyEditQueue(int capacity)
        {
            allocator = new BodySlotAllocator(capacity);
            gpuAlive = new bool[capacity];
        }

        public int Capacity => allocator.Capacity;

        public int AliveCount => allocator.AliveCount;

        public int HighWaterMark => allocator.HighWaterMark;

        /// <summary>Spawns that failed because every slot was in use.</summary>
        public int OverflowCount { get; private set; }

        /// <summary>Slots with an edit waiting for the next Drain.</summary>
        public int PendingSlotCount => order.Count;

        public bool IsAlive(BodyHandle handle) => allocator.IsAlive(handle);

        // ---- edits ------------------------------------------------------------------------------------

        /// <summary>
        /// Allocates a slot and queues a Spawn. Returns the invalid handle (and counts an overflow) if no slot is free.
        /// Throws ArgumentException for a descriptor with non-finite or out-of-range values; no slot is used in that case.
        /// </summary>
        public BodyHandle Spawn(in BodyDescriptor d)
        {
            ValidateDescriptor(d);

            if (!allocator.TryAllocate(out BodyHandle handle))
            {
                OverflowCount++;
                return BodyHandle.Invalid;
            }

            Pending p = Touch(handle.Slot);
            p.finalAlive = true;
            p.respawned = true;
            p.command = EncodeSpawn(handle.Slot, d);
            return handle;
        }

        /// <summary>Frees the body. Returns false for an invalid or stale handle.</summary>
        public bool Remove(BodyHandle handle)
        {
            if (!allocator.IsAlive(handle))
                return false;

            int slot = handle.Slot;
            Pending p = Touch(slot);
            allocator.Free(handle);
            p.finalAlive = false;
            p.respawned = false;
            return true;
        }

        /// <summary>Teleports the body. The rotation is normalized.</summary>
        public bool SetPose(BodyHandle handle, float3 position, quaternion rotation)
        {
            RequireFinite(position, nameof(position));
            float4 q = NormalizeRotation(rotation);
            if (!allocator.IsAlive(handle))
                return false;

            Pending p = Touch(handle.Slot);
            p.command.position = new float4(position, 0f);
            p.command.rotation = q;
            Mark(p, EditFields.Pose);
            return true;
        }

        public bool SetVelocity(BodyHandle handle, float3 linear, float3 angular)
        {
            RequireFinite(linear, nameof(linear));
            RequireFinite(angular, nameof(angular));
            if (!allocator.IsAlive(handle))
                return false;

            Pending p = Touch(handle.Slot);
            p.command.linear = new float4(linear, 0f);
            p.command.angular = new float4(angular, 0f);
            Mark(p, EditFields.Velocity);
            return true;
        }

        /// <summary>Changes the box size and recomputes mass and inertia from the density. Density 0 makes the body static.</summary>
        public bool SetShape(BodyHandle handle, float3 size, float density)
        {
            RequireSize(size);
            RequireNonNegative(density, nameof(density));
            if (!allocator.IsAlive(handle))
                return false;

            Pending p = Touch(handle.Slot);
            p.command.sizeFriction = new float4(size, p.command.sizeFriction.w);
            p.command.momentMass = BoxMassProperties.Compute(size, density);
            Mark(p, EditFields.Shape);
            return true;
        }

        public bool SetFriction(BodyHandle handle, float friction)
        {
            RequireNonNegative(friction, nameof(friction));
            if (!allocator.IsAlive(handle))
                return false;

            Pending p = Touch(handle.Slot);
            p.command.sizeFriction = new float4(p.command.sizeFriction.xyz, friction);
            Mark(p, EditFields.Friction);
            return true;
        }

        public bool SetGroupMask(BodyHandle handle, uint groupMask)
        {
            if (!allocator.IsAlive(handle))
                return false;

            Pending p = Touch(handle.Slot);
            p.command.meta.y = groupMask;
            Mark(p, EditFields.GroupMask);
            return true;
        }

        public bool SetAppearanceId(BodyHandle handle, uint appearanceId)
        {
            RequireAppearance(appearanceId);
            if (!allocator.IsAlive(handle))
                return false;

            Pending p = Touch(handle.Slot);
            p.command.meta.z = appearanceId;
            Mark(p, EditFields.Appearance);
            return true;
        }

        public bool SetKinematic(BodyHandle handle, bool kinematic)
        {
            if (!allocator.IsAlive(handle))
                return false;

            Pending p = Touch(handle.Slot);
            uint bit = (uint)BodyFlags.Kinematic;
            p.command.meta.x = kinematic ? (p.command.meta.x | bit) : (p.command.meta.x & ~bit);
            Mark(p, EditFields.Kinematic);
            return true;
        }

        // ---- draining ---------------------------------------------------------------------------------

        /// <summary>
        /// Appends the merged commands to <paramref name="output"/>, in the order slots were first touched, and
        /// treats them as applied: from now on the GPU is assumed to hold the result. Returns the number appended.
        /// </summary>
        public int Drain(List<EditCommand> output)
        {
            int before = output.Count;
            for (int i = 0; i < order.Count; i++)
            {
                Pending p = order[i];
                gpuAlive[p.slot] = p.finalAlive;

                if (!p.startAlive && !p.finalAlive)
                    continue; // spawned and removed inside this batch: the GPU never needs to know

                if (!p.finalAlive)
                    output.Add(EncodeRemove(p.slot));
                else if (p.respawned)
                    output.Add(p.command);
                else if (p.command.header.z != 0u)
                    output.Add(p.command);
            }
            bySlot.Clear();
            order.Clear();
            return output.Count - before;
        }

        // ---- helpers ----------------------------------------------------------------------------------

        Pending Touch(int slot)
        {
            if (!bySlot.TryGetValue(slot, out Pending p))
            {
                p = new Pending { slot = slot, startAlive = gpuAlive[slot], finalAlive = gpuAlive[slot] };
                p.command.header = new uint4((uint)EditKind.Set, (uint)slot, 0u, 0u);
                bySlot.Add(slot, p);
                order.Add(p);
            }
            return p;
        }

        // A spawn already writes every field, so only a Set needs to remember which fields to touch.
        static void Mark(Pending p, EditFields field)
        {
            if (!p.respawned)
                p.command.header.z |= (uint)field;
        }

        static EditCommand EncodeSpawn(int slot, in BodyDescriptor d)
        {
            float4 q = NormalizeRotation(d.rotation);
            BodyFlags flags = d.kinematic ? BodyFlags.Kinematic : BodyFlags.None;
            return new EditCommand
            {
                header = new uint4((uint)EditKind.Spawn, (uint)slot, (uint)EditFields.All, 0u),
                position = new float4(d.position, 0f),
                rotation = q,
                linear = new float4(d.linearVelocity, 0f),
                angular = new float4(d.angularVelocity, 0f),
                momentMass = BoxMassProperties.Compute(d.size, d.density),
                sizeFriction = new float4(d.size, d.friction),
                meta = new uint4(BodyFlagLayout.Pack(flags, ShapeKind.Box), d.groupMask, d.appearanceId, 0u),
            };
        }

        static EditCommand EncodeRemove(int slot)
        {
            return new EditCommand { header = new uint4((uint)EditKind.Remove, (uint)slot, 0u, 0u) };
        }

        // ---- validation -------------------------------------------------------------------------------

        static void ValidateDescriptor(in BodyDescriptor d)
        {
            RequireFinite(d.position, nameof(d.position));
            RequireFinite(d.linearVelocity, nameof(d.linearVelocity));
            RequireFinite(d.angularVelocity, nameof(d.angularVelocity));
            NormalizeRotation(d.rotation);
            RequireSize(d.size);
            RequireNonNegative(d.density, nameof(d.density));
            RequireNonNegative(d.friction, nameof(d.friction));
            RequireAppearance(d.appearanceId);
        }

        static float4 NormalizeRotation(quaternion rotation)
        {
            float4 v = rotation.value;
            if (!math.all(math.isfinite(v)) || math.lengthsq(v) <= 0f)
                throw new ArgumentException("rotation must be finite and non-zero.", nameof(rotation));
            return math.normalize(rotation).value;
        }

        static void RequireFinite(float3 v, string name)
        {
            if (!math.all(math.isfinite(v)))
                throw new ArgumentException(name + " must be finite.", name);
        }

        static void RequireSize(float3 size)
        {
            if (!math.all(math.isfinite(size)) || math.any(size <= 0f))
                throw new ArgumentException("size must be finite and positive in every dimension.", nameof(size));
        }

        static void RequireNonNegative(float value, string name)
        {
            if (!math.isfinite(value) || value < 0f)
                throw new ArgumentException(name + " must be finite and not negative.", name);
        }

        static void RequireAppearance(uint appearanceId)
        {
            if (appearanceId == AvbdConst.Invalid)
                throw new ArgumentException("appearanceId 0xFFFFFFFF is reserved for free slots.", nameof(appearanceId));
        }
    }
}
