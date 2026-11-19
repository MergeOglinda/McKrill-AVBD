using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace McKrill.Avbd
{
    /// <summary>
    /// Casts a ray against the boxes of an <see cref="AvbdBodyBuffers"/> on the GPU. Every live body is tested (ground, blocks, shots; any
    /// slot that is not free), against its pose and size in the buffers, so the CPU never reads the poses. Only the nearest hit distance
    /// (four bytes) comes back, asynchronously: <see cref="Cast"/> returns at once and calls its callback a few frames later.
    /// </summary>
    public sealed class AvbdRaycaster : IDisposable
    {
        readonly AvbdBodyBuffers bodies;
        readonly ComputeShader shader;
        readonly int clearKernel;
        readonly int castKernel;
        GraphicsBuffer result;

        static readonly int RayHitId = Shader.PropertyToID("_RayHit");
        static readonly int RayOriginId = Shader.PropertyToID("_RayOrigin");
        static readonly int RayDirectionId = Shader.PropertyToID("_RayDirection");

        public AvbdRaycaster(AvbdBodyBuffers bodies, AvbdShaderSet shaderSet = null)
        {
            this.bodies = bodies ?? throw new ArgumentNullException(nameof(bodies));
            AvbdShaderSet shaders = shaderSet != null ? shaderSet : AvbdShaderSet.Load();
            if (shaders == null)
                throw new InvalidOperationException("The AvbdShaderSet asset was not found.");
            shader = shaders.Raycast != null ? shaders.Raycast : throw new InvalidOperationException("The AvbdShaderSet has no raycast shader.");
            clearKernel = shader.FindKernel("Clear");
            castKernel = shader.FindKernel("Cast");
            result = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(uint));
        }

        /// <summary>
        /// Finds the nearest body hit by the ray, as of the work already submitted to the GPU (a step or an edit queued before this call is
        /// included, because the GPU runs in order). The poses are the ones the last step left.
        /// </summary>
        /// <param name="origin">Start of the ray.</param>
        /// <param name="direction">Direction; does not have to be unit length, but not zero.</param>
        /// <param name="maxDistance">Farthest distance that counts, above zero.</param>
        /// <param name="onResult">Called on the main thread once the result is back: whether something was hit, and the distance along the ray
        /// (in units of the normalized direction, 0 if the ray starts inside a box). Not called if the raycaster is disposed first.</param>
        public void Cast(float3 origin, float3 direction, float maxDistance, Action<bool, float> onResult)
        {
            if (onResult == null)
                throw new ArgumentNullException(nameof(onResult));
            DispatchCast(origin, direction, maxDistance);

            float max = maxDistance;
            AsyncGPUReadback.Request(result, request =>
            {
                if (result == null || request.hasError)
                    return; // disposed, or the buffer is gone
                uint bits = request.GetData<uint>()[0];
                float distance = math.asfloat(bits);
                onResult(bits != math.asuint(max), distance);
            });
        }

        // The dispatches, without the readback (the tests read the result buffer themselves).
        internal void DispatchCast(float3 origin, float3 direction, float maxDistance)
        {
            if (result == null)
                throw new ObjectDisposedException(nameof(AvbdRaycaster));
            if (!math.all(math.isfinite(origin)) || !math.all(math.isfinite(direction)) || !float.IsFinite(maxDistance))
                throw new ArgumentException("A ray needs finite values.");
            if (!(maxDistance > 0f))
                throw new ArgumentException("The maximum distance of a ray must be above zero.", nameof(maxDistance));
            float length = math.length(direction);
            if (!(length > 1e-9f))
                throw new ArgumentException("The direction of a ray must not be zero.", nameof(direction));

            bodies.ApplyEdits();
            shader.SetVector(RayOriginId, new Vector4(origin.x, origin.y, origin.z, maxDistance));
            float3 unit = direction / length;
            shader.SetVector(RayDirectionId, new Vector4(unit.x, unit.y, unit.z, 0f));

            shader.SetBuffer(clearKernel, RayHitId, result);
            shader.Dispatch(clearKernel, 1, 1, 1);

            bodies.BindTo(shader, castKernel);
            shader.SetBuffer(castKernel, RayHitId, result);
            AvbdDispatch.Linear(shader, castKernel, bodies.HighWaterMark);
        }

        internal GraphicsBuffer ResultBuffer => result;

        public void Dispose()
        {
            result?.Release();
            result = null;
        }
    }
}
