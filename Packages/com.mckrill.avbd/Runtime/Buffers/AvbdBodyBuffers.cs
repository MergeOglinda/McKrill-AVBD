using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd
{
    /// <summary>
    /// Owns the per-body GPU arrays and applies edits to them (stage E).
    ///
    /// Spawn, Remove and the Set methods only record edits on the CPU and return at once. ApplyEdits
    /// merges the recorded edits into at most one command per slot, uploads them and runs the edit kernel.
    /// A body is a row in the arrays, never a GameObject. The arrays have a fixed capacity.
    ///
    /// Methods must be called from the main thread. Disposing the object releases every buffer.
    /// </summary>
    public sealed class AvbdBodyBuffers : IDisposable
    {
        /// <summary>Default number of commands uploaded and dispatched at a time.</summary>
        public const int DefaultCommandBatchSize = 1024;

        /// <summary>Largest command batch: keeps one dispatch within the 65535 thread group limit.</summary>
        public const int MaxCommandBatchSize = 65535 * AvbdConst.GroupSize;

        const int MaxGroupsPerDispatch = 65535;

        static readonly int CapacityId = Shader.PropertyToID("_Capacity");
        static readonly int BaseId = Shader.PropertyToID("_Base");
        static readonly int CommandCountId = Shader.PropertyToID("_CommandCount");
        static readonly int CommandsId = Shader.PropertyToID("_Commands");
        static readonly int BodyStateId = Shader.PropertyToID("_BodyState");
        static readonly int BodyInertialId = Shader.PropertyToID("_BodyInertial");
        static readonly int BodyMassId = Shader.PropertyToID("_BodyMass");
        static readonly int BodyMotionId = Shader.PropertyToID("_BodyMotion");
        static readonly int BodyShapeId = Shader.PropertyToID("_BodyShape");
        static readonly int BodyMetaId = Shader.PropertyToID("_BodyMeta");
        static readonly int BodyRangeId = Shader.PropertyToID("_BodyRange");

        readonly BodyEditQueue queue;
        readonly ComputeShader shader;
        readonly int clearKernel;
        readonly int applyKernel;
        readonly int commandBatchSize;
        readonly List<EditCommand> drained = new List<EditCommand>();

        GraphicsBuffer state;
        GraphicsBuffer inertial;
        GraphicsBuffer mass;
        GraphicsBuffer motion;
        GraphicsBuffer shape;
        GraphicsBuffer meta;
        GraphicsBuffer range;
        GraphicsBuffer commands;
        bool disposed;

        /// <param name="capacity">Number of body slots. Fixed for the lifetime of the object.</param>
        /// <param name="commandBatchSize">Commands uploaded per dispatch. Larger edit batches are applied in several dispatches.</param>
        /// <param name="editShader">The edit compute shader. Defaults to the one in the package's shader set.</param>
        public AvbdBodyBuffers(int capacity, int commandBatchSize = DefaultCommandBatchSize, ComputeShader editShader = null)
        {
            BodyBufferCapacity.Validate(capacity);
            if (commandBatchSize < 1 || commandBatchSize > MaxCommandBatchSize)
                throw new ArgumentOutOfRangeException(nameof(commandBatchSize), "Command batch size must be between 1 and " + MaxCommandBatchSize + ".");
            if (!SystemInfo.supportsComputeShaders)
                throw new InvalidOperationException("Compute shaders are not supported on this device.");

            shader = editShader != null ? editShader : AvbdShaderSet.Load()?.BodyEdits;
            if (shader == null)
                throw new InvalidOperationException("The body edit compute shader was not found. Check the AvbdShaderSet asset.");

            this.commandBatchSize = commandBatchSize;
            clearKernel = shader.FindKernel("ClearBodies");
            applyKernel = shader.FindKernel("ApplyEdits");
            queue = new BodyEditQueue(capacity);

            try
            {
                state = Create(capacity, BodyRecordStrides.State);
                inertial = Create(capacity, BodyRecordStrides.Inertial);
                mass = Create(capacity, BodyRecordStrides.Mass);
                motion = Create(capacity, BodyRecordStrides.Motion);
                shape = Create(capacity, BodyRecordStrides.Shape);
                meta = Create(capacity, BodyRecordStrides.Meta);
                range = Create(capacity, BodyRecordStrides.Range);
                commands = Create(commandBatchSize, EditCommandStride.Size);
            }
            catch
            {
                Dispose();
                throw;
            }

            ClearAllSlots();
        }

        static GraphicsBuffer Create(int count, int stride)
        {
            return new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, stride);
        }

        // ---- state ------------------------------------------------------------------------------------

        /// <summary>Number of body slots.</summary>
        public int Capacity => queue.Capacity;

        /// <summary>Bodies alive, counting edits that have not been applied yet.</summary>
        public int AliveCount => queue.AliveCount;

        /// <summary>One past the highest slot ever used. Stage kernels that scan all slots need not go beyond it.</summary>
        public int HighWaterMark => queue.HighWaterMark;

        /// <summary>Spawn calls that failed because every slot was in use.</summary>
        public int OverflowCount => queue.OverflowCount;

        /// <summary>True if edits are waiting for <see cref="ApplyEdits()"/>.</summary>
        public bool HasPendingEdits => queue.PendingSlotCount > 0;

        public bool IsAlive(BodyHandle handle) => queue.IsAlive(handle);

        // ---- buffers for the other stages ---------------------------------------------------------------
        // Read-only by convention: only stage E writes the fields it owns (design doc, section 1).

        public GraphicsBuffer State => state;
        public GraphicsBuffer Inertial => inertial;
        public GraphicsBuffer Mass => mass;
        public GraphicsBuffer Motion => motion;
        public GraphicsBuffer Shape => shape;
        public GraphicsBuffer Meta => meta;
        public GraphicsBuffer Range => range;

        // ---- edits (recorded now, applied by ApplyEdits) ------------------------------------------------

        /// <summary>
        /// Reserves a slot for a new body. Returns <see cref="BodyHandle.Invalid"/> if the buffers are full
        /// (and counts it in <see cref="OverflowCount"/>). Throws ArgumentException for non-finite or out-of-range values.
        /// </summary>
        public BodyHandle Spawn(in BodyDescriptor descriptor)
        {
            ThrowIfDisposed();
            return queue.Spawn(descriptor);
        }

        /// <summary>Frees the body. Returns false for an invalid or stale handle.</summary>
        public bool Remove(BodyHandle handle)
        {
            ThrowIfDisposed();
            return queue.Remove(handle);
        }

        /// <summary>Teleports the body; also resets its step-start pose. Returns false for an invalid or stale handle.</summary>
        public bool SetPose(BodyHandle handle, float3 position, quaternion rotation)
        {
            ThrowIfDisposed();
            return queue.SetPose(handle, position, rotation);
        }

        /// <summary>Sets the velocities; also resets the previous velocity used by the adaptive warm start.</summary>
        public bool SetVelocity(BodyHandle handle, float3 linear, float3 angular)
        {
            ThrowIfDisposed();
            return queue.SetVelocity(handle, linear, angular);
        }

        /// <summary>Sets the box size and recomputes mass and inertia from the density. Density 0 makes the body static.</summary>
        public bool SetShape(BodyHandle handle, float3 size, float density)
        {
            ThrowIfDisposed();
            return queue.SetShape(handle, size, density);
        }

        public bool SetFriction(BodyHandle handle, float friction)
        {
            ThrowIfDisposed();
            return queue.SetFriction(handle, friction);
        }

        public bool SetGroupMask(BodyHandle handle, uint groupMask)
        {
            ThrowIfDisposed();
            return queue.SetGroupMask(handle, groupMask);
        }

        public bool SetAppearanceId(BodyHandle handle, uint appearanceId)
        {
            ThrowIfDisposed();
            return queue.SetAppearanceId(handle, appearanceId);
        }

        public bool SetKinematic(BodyHandle handle, bool kinematic)
        {
            ThrowIfDisposed();
            return queue.SetKinematic(handle, kinematic);
        }

        // ---- stage E ----------------------------------------------------------------------------------

        /// <summary>
        /// Counts applied edits that change the set of solved bodies: spawns, removals, and changes of shape (mass) or of the kinematic flag.
        /// Moving a body or changing its velocity does not count. <see cref="AvbdConstraintBuffers"/> rebuilds its schedule when this changes.
        /// </summary>
        internal int StructureVersion { get; private set; }

        /// <summary>Uploads the recorded edits and runs the edit kernel. Returns the number of commands applied.</summary>
        public int ApplyEdits() => ApplyEdits(null);

        // Tests pass a list to receive the commands, to replay them on the CPU oracle.
        internal int ApplyEdits(List<EditCommand> applied)
        {
            ThrowIfDisposed();

            drained.Clear();
            int total = queue.Drain(drained);
            if (applied != null)
                applied.AddRange(drained);
            if (total == 0)
                return 0;

            // Anything that can change which bodies the solver updates or how they are joined invalidates the constraint schedule.
            const uint structural = (uint)EditFields.Shape | (uint)EditFields.Kinematic;
            for (int i = 0; i < drained.Count; i++)
            {
                uint kind = drained[i].header.x;
                if (kind == (uint)EditKind.Spawn || kind == (uint)EditKind.Remove || (drained[i].header.z & structural) != 0u)
                {
                    StructureVersion++;
                    break;
                }
            }

            BindBodyBuffers(applyKernel);
            shader.SetBuffer(applyKernel, CommandsId, commands);
            shader.SetInt(CapacityId, Capacity);

            for (int offset = 0; offset < total; offset += commandBatchSize)
            {
                int count = Math.Min(commandBatchSize, total - offset);
                commands.SetData(drained, offset, 0, count);
                shader.SetInt(CommandCountId, count);
                shader.Dispatch(applyKernel, (count + AvbdConst.GroupSize - 1) / AvbdConst.GroupSize, 1, 1);
            }
            return total;
        }

        // Puts every slot into the free state (rule P9: new buffers are never assumed to be zeroed).
        void ClearAllSlots()
        {
            BindBodyBuffers(clearKernel);
            shader.SetInt(CapacityId, Capacity);

            int groups = (Capacity + AvbdConst.GroupSize - 1) / AvbdConst.GroupSize;
            for (int firstGroup = 0; firstGroup < groups; firstGroup += MaxGroupsPerDispatch)
            {
                int count = Math.Min(MaxGroupsPerDispatch, groups - firstGroup);
                shader.SetInt(BaseId, firstGroup * AvbdConst.GroupSize);
                shader.Dispatch(clearKernel, count, 1, 1);
            }
        }

        void BindBodyBuffers(int kernel) => BindTo(shader, kernel);

        /// <summary>
        /// Binds all seven body arrays to a kernel under the standard names (_BodyState, _BodyInertial, _BodyMass,
        /// _BodyMotion, _BodyShape, _BodyMeta, _BodyRange). A kernel that does not declare one of them ignores it.
        /// </summary>
        internal void BindTo(ComputeShader target, int kernel)
        {
            target.SetBuffer(kernel, BodyStateId, state);
            target.SetBuffer(kernel, BodyInertialId, inertial);
            target.SetBuffer(kernel, BodyMassId, mass);
            target.SetBuffer(kernel, BodyMotionId, motion);
            target.SetBuffer(kernel, BodyShapeId, shape);
            target.SetBuffer(kernel, BodyMetaId, meta);
            target.SetBuffer(kernel, BodyRangeId, range);
        }

        void ThrowIfDisposed()
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(AvbdBodyBuffers));
        }

        public void Dispose()
        {
            if (disposed)
                return;
            disposed = true;

            Release(ref state);
            Release(ref inertial);
            Release(ref mass);
            Release(ref motion);
            Release(ref shape);
            Release(ref meta);
            Release(ref range);
            Release(ref commands);
        }

        static void Release(ref GraphicsBuffer buffer)
        {
            buffer?.Release();
            buffer = null;
        }
    }
}
