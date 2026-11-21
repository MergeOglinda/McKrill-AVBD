using System;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd
{
    /// <summary>Which values <see cref="AvbdGroupMover.Move"/> writes. The others are left as they are.</summary>
    [Flags]
    public enum GroupMoveFields : uint
    {
        None = 0,

        /// <summary>Add the translation to the position.</summary>
        Position = 1u << 0,

        /// <summary>Set the linear velocity.</summary>
        LinearVelocity = 1u << 1,

        /// <summary>Set the angular velocity.</summary>
        AngularVelocity = 1u << 2,
    }

    /// <summary>
    /// Moves a group of bodies directly, on the GPU: adds a translation to their position and/or sets their velocity, for every body whose group mask
    /// (<see cref="BodyDescriptor.groupMask"/>) shares a bit with a mask. The group can be picked on the GPU too (<see cref="AvbdGroupLabeler"/>), so the CPU
    /// never reads the bodies.
    ///
    /// This is neither a force nor a kinematic body: no force or impulse acts, and nothing but the pose and the velocity of a body changes. Its mass, moments
    /// of inertia, shape, friction and flags stay the same, so it keeps responding to contacts and joints as the same body. The step after the move starts from the
    /// new pose and velocity. Moving by a distance and setting the velocity that goes with it (the distance over the time of the step) makes the bodies keep
    /// moving the way they were moved. To keep every body of a group at its own place relative to a point that moves (a direct grab), whatever the steps do to it (gravity,
    /// contacts, joints), use <see cref="BeginHold"/> and <see cref="Hold"/>: they set absolute positions, so they do not let the group sink. For a single body, <see cref="AvbdBodyBuffers.SetPose"/> and <see cref="AvbdBodyBuffers.SetVelocity"/> set absolute values from the CPU.
    /// </summary>
    public sealed class AvbdGroupMover : IDisposable
    {
        readonly AvbdBodyBuffers bodies;
        readonly ComputeShader shader;
        readonly int moveKernel;
        readonly int holdCaptureKernel;
        readonly int holdApplyKernel;
        GraphicsBuffer holdOffsets; // one float4 per slot of the body buffers: the offset of the body from the held point, w = 1 if it is held
        bool disposed;

        static readonly int MoveTranslationId = Shader.PropertyToID("_MoveTranslation");
        static readonly int MoveLinearId = Shader.PropertyToID("_MoveLinear");
        static readonly int MoveAngularId = Shader.PropertyToID("_MoveAngular");
        static readonly int MoveControlId = Shader.PropertyToID("_MoveControl");
        static readonly int HoldOffsetsId = Shader.PropertyToID("_HoldOffsets");
        static readonly int HoldPointId = Shader.PropertyToID("_HoldPoint");

        public AvbdGroupMover(AvbdBodyBuffers bodies, AvbdShaderSet shaderSet = null)
        {
            this.bodies = bodies ?? throw new ArgumentNullException(nameof(bodies));
            AvbdShaderSet shaders = shaderSet != null ? shaderSet : AvbdShaderSet.Load();
            if (shaders == null)
                throw new InvalidOperationException("The AvbdShaderSet asset was not found.");
            shader = shaders.GroupMove != null ? shaders.GroupMove : throw new InvalidOperationException("The AvbdShaderSet has no group move shader.");
            moveKernel = shader.FindKernel("Move");
            holdCaptureKernel = shader.FindKernel("HoldCapture");
            holdApplyKernel = shader.FindKernel("HoldApply");
        }

        /// <summary>
        /// For every body whose group mask shares a bit with <paramref name="groupMask"/>: adds <paramref name="translation"/> to its position (with
        /// <see cref="GroupMoveFields.Position"/>), sets its linear velocity to <paramref name="linearVelocity"/> (with <see cref="GroupMoveFields.LinearVelocity"/>)
        /// and its angular velocity to <paramref name="angularVelocity"/> (with <see cref="GroupMoveFields.AngularVelocity"/>). Values of fields that are not
        /// selected are ignored. Static bodies (mass 0) and free slots are never moved; kinematic bodies are.
        /// A position change also resets the step-start pose, as <see cref="AvbdBodyBuffers.SetPose"/> does, and a linear velocity change also sets the previous linear velocity
        /// (see <see cref="AvbdBodyBuffers.SetVelocity"/>). The rotations are not changed. It runs on the GPU in order with the steps and edits submitted before it.
        /// </summary>
        /// <param name="translation">In m, added to the world position of each body.</param>
        /// <param name="linearVelocity">In m/s.</param>
        /// <param name="angularVelocity">In rad/s.</param>
        public void Move(uint groupMask, GroupMoveFields fields, float3 translation, float3 linearVelocity, float3 angularVelocity)
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(AvbdGroupMover));
            if ((fields & GroupMoveFields.Position) != 0 && !math.all(math.isfinite(translation)))
                throw new ArgumentException("The translation must be finite.", nameof(translation));
            if ((fields & GroupMoveFields.LinearVelocity) != 0 && !math.all(math.isfinite(linearVelocity)))
                throw new ArgumentException("The linear velocity must be finite.", nameof(linearVelocity));
            if ((fields & GroupMoveFields.AngularVelocity) != 0 && !math.all(math.isfinite(angularVelocity)))
                throw new ArgumentException("The angular velocity must be finite.", nameof(angularVelocity));
            if (fields == GroupMoveFields.None || groupMask == 0u)
                return;

            bodies.ApplyEdits();
            shader.SetVector(MoveTranslationId, new Vector4(translation.x, translation.y, translation.z, 0f));
            shader.SetVector(MoveLinearId, new Vector4(linearVelocity.x, linearVelocity.y, linearVelocity.z, 0f));
            shader.SetVector(MoveAngularId, new Vector4(angularVelocity.x, angularVelocity.y, angularVelocity.z, 0f));
            shader.SetInts(MoveControlId, (int)groupMask, (int)fields, 0, 0);
            bodies.BindTo(shader, moveKernel);
            AvbdDispatch.Linear(shader, moveKernel, bodies.HighWaterMark);
        }

        /// <summary>Adds <paramref name="translation"/> to the position of the bodies of the group, and sets their linear velocity to <paramref name="linearVelocity"/>.</summary>
        public void MoveWithVelocity(uint groupMask, float3 translation, float3 linearVelocity)
        {
            Move(groupMask, GroupMoveFields.Position | GroupMoveFields.LinearVelocity, translation, linearVelocity, float3.zero);
        }

        /// <summary>Adds <paramref name="translation"/> to the position of the bodies of the group; their velocities stay.</summary>
        public void Translate(uint groupMask, float3 translation)
        {
            Move(groupMask, GroupMoveFields.Position, translation, float3.zero, float3.zero);
        }

        /// <summary>Sets the velocity of the bodies of the group; their positions stay.</summary>
        public void SetVelocity(uint groupMask, float3 linearVelocity, float3 angularVelocity)
        {
            Move(groupMask, GroupMoveFields.LinearVelocity | GroupMoveFields.AngularVelocity, float3.zero, linearVelocity, angularVelocity);
        }

        /// <summary>
        /// Remembers where every body of the group is relative to <paramref name="point"/>, on the GPU (nothing is read back), for <see cref="Hold"/>: one offset per body,
        /// in a buffer of one float4 per slot of the body buffers. The group is the bodies whose group mask shares a bit with <paramref name="groupMask"/>, that are not
        /// static (mass 0) or free; kinematic bodies count. Nothing moves. Call it again to remember a new arrangement; a body that joins the group later is not held.
        /// </summary>
        /// <param name="groupMask">The bodies that are held.</param>
        /// <param name="point">The held point, in m: where the point that the group will follow is now.</param>
        public void BeginHold(uint groupMask, float3 point)
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(AvbdGroupMover));
            if (!math.all(math.isfinite(point)))
                throw new ArgumentException("The point must be finite.", nameof(point));
            if (groupMask == 0u)
                return;

            bodies.ApplyEdits();
            if (holdOffsets == null)
            {
                holdOffsets = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bodies.Capacity, 4 * sizeof(float));
                holdOffsets.SetData(new Vector4[bodies.Capacity]); // nobody is held yet
            }
            shader.SetVector(HoldPointId, new Vector4(point.x, point.y, point.z, 0f));
            shader.SetInts(MoveControlId, (int)groupMask, 0, 0, 0);
            shader.SetBuffer(holdCaptureKernel, HoldOffsetsId, holdOffsets);
            bodies.BindTo(shader, holdCaptureKernel);
            AvbdDispatch.Linear(shader, holdCaptureKernel, bodies.Capacity);
        }

        /// <summary>
        /// Puts every body that <see cref="BeginHold"/> remembered (and that is still in the group) at <paramref name="point"/> plus its own offset, and sets its linear
        /// velocity (also the previous one) to <paramref name="linearVelocity"/>. Each body is placed on its own, so the group keeps the arrangement it had when it was
        /// remembered, whatever the steps did to the bodies since. The position is absolute: what a step did (gravity, a push) is undone by the next call, so a group that is
        /// held every frame does not sink or lag. The velocity is the one of the point, so that the bodies that are let go keep moving the way they were moved (and a throw
        /// is possible); the angular velocity and the rotation are not changed. Nothing else changes either: mass, moments of inertia, shape, friction, flags and the group
        /// mask stay, so the bodies keep responding to contacts and joints as the same body. It is a teleport like <see cref="Move"/> with a position (the step-start pose
        /// follows). Does nothing before <see cref="BeginHold"/>.
        /// </summary>
        /// <param name="groupMask">The bodies that are held (the mask of <see cref="BeginHold"/>).</param>
        /// <param name="point">The held point now, in m.</param>
        /// <param name="linearVelocity">The velocity of the held point, in m/s.</param>
        public void Hold(uint groupMask, float3 point, float3 linearVelocity)
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(AvbdGroupMover));
            if (!math.all(math.isfinite(point)))
                throw new ArgumentException("The point must be finite.", nameof(point));
            if (!math.all(math.isfinite(linearVelocity)))
                throw new ArgumentException("The linear velocity must be finite.", nameof(linearVelocity));
            if (groupMask == 0u || holdOffsets == null)
                return;

            bodies.ApplyEdits();
            shader.SetVector(HoldPointId, new Vector4(point.x, point.y, point.z, 0f));
            shader.SetVector(MoveLinearId, new Vector4(linearVelocity.x, linearVelocity.y, linearVelocity.z, 0f));
            shader.SetInts(MoveControlId, (int)groupMask, 0, 0, 0);
            shader.SetBuffer(holdApplyKernel, HoldOffsetsId, holdOffsets);
            bodies.BindTo(shader, holdApplyKernel);
            AvbdDispatch.Linear(shader, holdApplyKernel, bodies.HighWaterMark);
        }

        public void Dispose()
        {
            disposed = true;
            if (holdOffsets != null) holdOffsets.Release();
            holdOffsets = null;
        }
    }
}
