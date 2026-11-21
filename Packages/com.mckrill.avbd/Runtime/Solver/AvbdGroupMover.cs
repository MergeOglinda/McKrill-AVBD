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
    /// moving the way they were moved. To keep a group where the user holds it, whatever the steps do to it (gravity, contacts, joints), use
    /// <see cref="BeginHold"/> and <see cref="Hold"/>: they put the centroid of the group at a point and set the velocities to zero, so that calling Hold before every step
    /// does not let the group sink. For a single body, <see cref="AvbdBodyBuffers.SetPose"/> and <see cref="AvbdBodyBuffers.SetVelocity"/> set absolute values from the CPU.
    /// </summary>
    public sealed class AvbdGroupMover : IDisposable
    {
        readonly AvbdBodyBuffers bodies;
        readonly ComputeShader shader;
        readonly int moveKernel;
        readonly int holdClearKernel;
        readonly int holdAccumulateKernel;
        readonly int holdCaptureKernel;
        readonly int holdApplyKernel;
        GraphicsBuffer holdAccum;  // 4 words: the sums of the offsets of the group from the reference (3, fixed point) and its number of bodies
        GraphicsBuffer holdAnchor; // 1 float4: the centroid captured by BeginHold
        bool disposed;

        static readonly int MoveTranslationId = Shader.PropertyToID("_MoveTranslation");
        static readonly int MoveLinearId = Shader.PropertyToID("_MoveLinear");
        static readonly int MoveAngularId = Shader.PropertyToID("_MoveAngular");
        static readonly int MoveControlId = Shader.PropertyToID("_MoveControl");
        static readonly int HoldAccumId = Shader.PropertyToID("_HoldAccum");
        static readonly int HoldAnchorId = Shader.PropertyToID("_HoldAnchor");
        static readonly int HoldReferenceId = Shader.PropertyToID("_HoldReference");
        static readonly int HoldOffsetId = Shader.PropertyToID("_HoldOffset");

        public AvbdGroupMover(AvbdBodyBuffers bodies, AvbdShaderSet shaderSet = null)
        {
            this.bodies = bodies ?? throw new ArgumentNullException(nameof(bodies));
            AvbdShaderSet shaders = shaderSet != null ? shaderSet : AvbdShaderSet.Load();
            if (shaders == null)
                throw new InvalidOperationException("The AvbdShaderSet asset was not found.");
            shader = shaders.GroupMove != null ? shaders.GroupMove : throw new InvalidOperationException("The AvbdShaderSet has no group move shader.");
            moveKernel = shader.FindKernel("Move");
            holdClearKernel = shader.FindKernel("HoldClear");
            holdAccumulateKernel = shader.FindKernel("HoldAccumulate");
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
        /// Takes the centroid of the bodies of the group as the anchor of a hold (<see cref="Hold"/>), on the GPU: nothing is read back. The group is the bodies whose group
        /// mask shares a bit with <paramref name="groupMask"/>, that are not static (mass 0) or free; kinematic bodies count. Nothing moves. Call it again to take a new anchor.
        /// </summary>
        /// <param name="groupMask">The bodies that are held.</param>
        /// <param name="reference">A point near the group (the offsets of the bodies from it are summed as integers in 1/1024 m, so a reference far from the group, or a group
        /// of more than about 2 million m of summed offsets, overflows). The grab point is a good one.</param>
        public void BeginHold(uint groupMask, float3 reference)
        {
            CheckHold(groupMask, reference);
            if (groupMask == 0u)
                return;
            SumOffsets(groupMask, reference);
            shader.SetBuffer(holdCaptureKernel, HoldAccumId, holdAccum);
            shader.SetBuffer(holdCaptureKernel, HoldAnchorId, holdAnchor);
            shader.SetVector(HoldReferenceId, new Vector4(reference.x, reference.y, reference.z, 0f));
            shader.Dispatch(holdCaptureKernel, 1, 1, 1);
        }

        /// <summary>
        /// Holds the group: moves every body of it by the same translation so that the centroid of the group is at the anchor of <see cref="BeginHold"/> plus
        /// <paramref name="offset"/>, and sets their linear (also the previous one) and angular velocity to zero. Nothing else changes: not the rotations, mass, moments, shape,
        /// friction, flags or group masks, so the bodies keep responding to contacts and joints as the same body, and the group keeps its arrangement. It is a teleport like
        /// <see cref="Move"/> with a position (the step-start pose follows). Because the target is absolute, a body that the step before pulled or pushed away is put back, so
        /// calling it before every step holds the group, and with its velocities at zero the step starts from rest. Does nothing if <see cref="BeginHold"/> found no body.
        /// </summary>
        /// <param name="groupMask">The bodies that are held (the mask of <see cref="BeginHold"/>).</param>
        /// <param name="reference">The reference point of <see cref="BeginHold"/>.</param>
        /// <param name="offset">In m: where the centroid is to be, relative to the anchor.</param>
        public void Hold(uint groupMask, float3 reference, float3 offset)
        {
            CheckHold(groupMask, reference);
            if (!math.all(math.isfinite(offset)))
                throw new ArgumentException("The offset must be finite.", nameof(offset));
            if (groupMask == 0u || holdAnchor == null)
                return;
            SumOffsets(groupMask, reference);
            shader.SetBuffer(holdApplyKernel, HoldAccumId, holdAccum);
            shader.SetBuffer(holdApplyKernel, HoldAnchorId, holdAnchor);
            shader.SetVector(HoldReferenceId, new Vector4(reference.x, reference.y, reference.z, 0f));
            shader.SetVector(HoldOffsetId, new Vector4(offset.x, offset.y, offset.z, 0f));
            bodies.BindTo(shader, holdApplyKernel);
            AvbdDispatch.Linear(shader, holdApplyKernel, bodies.HighWaterMark);
        }

        void CheckHold(uint groupMask, float3 reference)
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(AvbdGroupMover));
            if (!math.all(math.isfinite(reference)))
                throw new ArgumentException("The reference must be finite.", nameof(reference));
        }

        // Clears the sums and adds the offsets of the bodies of the group from the reference.
        void SumOffsets(uint groupMask, float3 reference)
        {
            bodies.ApplyEdits();
            if (holdAccum == null)
            {
                holdAccum = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 4, sizeof(uint));
                holdAnchor = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, 4 * sizeof(float));
                holdAnchor.SetData(new Vector4[1]); // no anchor yet
            }
            shader.SetBuffer(holdClearKernel, HoldAccumId, holdAccum);
            shader.Dispatch(holdClearKernel, 1, 1, 1);

            shader.SetVector(HoldReferenceId, new Vector4(reference.x, reference.y, reference.z, 0f));
            shader.SetInts(MoveControlId, (int)groupMask, 0, 0, 0);
            shader.SetBuffer(holdAccumulateKernel, HoldAccumId, holdAccum);
            bodies.BindTo(shader, holdAccumulateKernel);
            AvbdDispatch.Linear(shader, holdAccumulateKernel, bodies.HighWaterMark);
        }

        public void Dispose()
        {
            disposed = true;
            if (holdAccum != null) holdAccum.Release();
            holdAccum = null;
            if (holdAnchor != null) holdAnchor.Release();
            holdAnchor = null;
        }
    }
}
