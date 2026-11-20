using System;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd
{
    /// <summary>
    /// Changes the group masks (<see cref="BodyDescriptor.groupMask"/>) of the bodies near a point, on the GPU. The group mask is the label that
    /// <see cref="ForceFieldDescriptor"/> fields use, so this picks a group of bodies for a field without the CPU reading the bodies: a ray cast
    /// (<see cref="AvbdRaycaster"/>) finds a point, <see cref="Label"/> puts a bit on every body within a radius of it, and a field with that bit acts on them.
    /// </summary>
    public sealed class AvbdGroupLabeler : IDisposable
    {
        readonly AvbdBodyBuffers bodies;
        readonly ComputeShader shader;
        readonly int labelKernel;
        bool disposed;

        static readonly int LabelCenterId = Shader.PropertyToID("_LabelCenter");
        static readonly int LabelBitsId = Shader.PropertyToID("_LabelBits");
        static readonly int BodyMetaId = Shader.PropertyToID("_BodyMeta");

        public AvbdGroupLabeler(AvbdBodyBuffers bodies, AvbdShaderSet shaderSet = null)
        {
            this.bodies = bodies ?? throw new ArgumentNullException(nameof(bodies));
            AvbdShaderSet shaders = shaderSet != null ? shaderSet : AvbdShaderSet.Load();
            if (shaders == null)
                throw new InvalidOperationException("The AvbdShaderSet asset was not found.");
            shader = shaders.GroupLabel != null ? shaders.GroupLabel : throw new InvalidOperationException("The AvbdShaderSet has no group label shader.");
            labelKernel = shader.FindKernel("Label");
        }

        /// <summary>
        /// Gives every body whose box is within <paramref name="radius"/> of <paramref name="center"/> (distance 0 inside the box) the group mask
        /// <c>(mask | setBits) &amp; ~clearBits</c>. Static and kinematic bodies are labelled too; only free slots are skipped. It runs on the GPU in order
        /// with the steps and edits submitted before it, and acts on the poses the last step left.
        /// </summary>
        /// <param name="radius">In m, not negative. 0 selects the bodies that contain the point.</param>
        public void Label(float3 center, float radius, uint setBits, uint clearBits)
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(AvbdGroupLabeler));
            if (!math.all(math.isfinite(center)) || !float.IsFinite(radius) || radius < 0f)
                throw new ArgumentException("The center must be finite and the radius finite and not negative.");

            bodies.ApplyEdits();
            shader.SetVector(LabelCenterId, new Vector4(center.x, center.y, center.z, radius));
            shader.SetInts(LabelBitsId, (int)setBits, (int)clearBits, 0, 0);
            bodies.BindTo(shader, labelKernel);
            AvbdDispatch.Linear(shader, labelKernel, bodies.HighWaterMark);
        }

        /// <summary>Clears <paramref name="bits"/> from the group mask of every body.</summary>
        public void ClearAll(uint bits)
        {
            Label(float3.zero, 1e18f, 0u, bits);
        }

        public void Dispose()
        {
            disposed = true;
        }
    }
}
