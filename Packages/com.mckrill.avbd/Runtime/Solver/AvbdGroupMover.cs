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
    /// moving the way they were moved. For a single body, <see cref="AvbdBodyBuffers.SetPose"/> and <see cref="AvbdBodyBuffers.SetVelocity"/> set absolute values from the CPU.
    /// </summary>
    public sealed class AvbdGroupMover : IDisposable
    {
        readonly AvbdBodyBuffers bodies;
        readonly ComputeShader shader;
        readonly int moveKernel;
        bool disposed;

        static readonly int MoveTranslationId = Shader.PropertyToID("_MoveTranslation");
        static readonly int MoveLinearId = Shader.PropertyToID("_MoveLinear");
        static readonly int MoveAngularId = Shader.PropertyToID("_MoveAngular");
        static readonly int MoveControlId = Shader.PropertyToID("_MoveControl");

        public AvbdGroupMover(AvbdBodyBuffers bodies, AvbdShaderSet shaderSet = null)
        {
            this.bodies = bodies ?? throw new ArgumentNullException(nameof(bodies));
            AvbdShaderSet shaders = shaderSet != null ? shaderSet : AvbdShaderSet.Load();
            if (shaders == null)
                throw new InvalidOperationException("The AvbdShaderSet asset was not found.");
            shader = shaders.GroupMove != null ? shaders.GroupMove : throw new InvalidOperationException("The AvbdShaderSet has no group move shader.");
            moveKernel = shader.FindKernel("Move");
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

        public void Dispose()
        {
            disposed = true;
        }
    }
}
