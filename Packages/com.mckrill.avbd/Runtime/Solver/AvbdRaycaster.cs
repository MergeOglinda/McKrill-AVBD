using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace McKrill.Avbd
{
    /// <summary>
    /// Casts a ray against the boxes of an <see cref="AvbdBodyBuffers"/> on the GPU. Every live body is tested (ground, blocks, shots; any
    /// slot that is not free), against its pose and size in the buffers, so the CPU never reads the poses. Only the answer (32 bytes: the nearest hit distance,
    /// the body and the hit point in its frame) comes back, asynchronously: <see cref="Cast"/> returns at once and calls its callback a few frames later.
    /// </summary>
    public sealed class AvbdRaycaster : IDisposable
    {
        /// <summary>Words of the result buffer: distance, slot, local point (3), local direction (3). Mirrors AvbdRaycast.compute.</summary>
        internal const int ResultWords = 8;
        internal const uint NoBody = 0xFFFFFFFFu;

        readonly AvbdBodyBuffers bodies;
        readonly ComputeShader shader;
        readonly int clearKernel;
        readonly int castKernel;
        readonly int castBodyKernel;
        readonly int finishKernel;
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
            castBodyKernel = shader.FindKernel("CastBody");
            finishKernel = shader.FindKernel("Finish");
            result = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ResultWords, sizeof(uint));
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
            Cast(origin, direction, maxDistance, hit => onResult(hit.hit, hit.distance));
        }

        /// <summary>
        /// As <see cref="Cast(float3, float3, float, Action{bool, float})"/>, and the answer also names the body that was hit and where, in the frame of that body
        /// (see <see cref="AvbdRayHit"/>), so the caller can follow the body: the answer arrives a few frames after the ray was cast, and the body has moved by then.
        /// The body is the one that held the slot when the ray was cast; if it was removed before the answer arrived, <see cref="AvbdRayHit.body"/> is invalid.
        /// </summary>
        /// <param name="onResult">Called on the main thread once the result is back. Not called if the raycaster is disposed first.</param>
        public void Cast(float3 origin, float3 direction, float maxDistance, Action<AvbdRayHit> onResult)
        {
            if (onResult == null)
                throw new ArgumentNullException(nameof(onResult));
            DispatchCast(origin, direction, maxDistance);

            float max = maxDistance;
            uint removals = bodies.RemovalSerial; // a body removed after this is not the one the GPU found
            AsyncGPUReadback.Request(result, request =>
            {
                if (result == null || request.hasError)
                    return; // disposed, or the buffer is gone
                Unity.Collections.NativeArray<uint> words = request.GetData<uint>();
                onResult(Decode(words, max, bodies, removals));
            });
        }

        // The words of the result buffer as an answer. A hit whose body is gone (or that the kernel could not name) has an invalid body.
        internal static AvbdRayHit Decode(Unity.Collections.NativeArray<uint> words, float maxDistance, AvbdBodyBuffers bodies, uint removalsAtCast)
        {
            uint distanceBits = words[0];
            if (distanceBits == math.asuint(maxDistance))
                return new AvbdRayHit(false, maxDistance, BodyHandle.Invalid, float3.zero, float3.zero);

            BodyHandle body = BodyHandle.Invalid;
            float3 localPoint = float3.zero;
            float3 localDirection = float3.zero;
            if (words[1] != NoBody && bodies != null && bodies.TryGetHandleSince((int)words[1], removalsAtCast, out BodyHandle found))
            {
                body = found;
                localPoint = new float3(math.asfloat(words[2]), math.asfloat(words[3]), math.asfloat(words[4]));
                localDirection = new float3(math.asfloat(words[5]), math.asfloat(words[6]), math.asfloat(words[7]));
            }
            return new AvbdRayHit(true, math.asfloat(distanceBits), body, localPoint, localDirection);
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

            // Which body gave the nearest distance, then its local hit point: each reads what the one before wrote.
            bodies.BindTo(shader, castBodyKernel);
            shader.SetBuffer(castBodyKernel, RayHitId, result);
            AvbdDispatch.Linear(shader, castBodyKernel, bodies.HighWaterMark);

            bodies.BindTo(shader, finishKernel);
            shader.SetBuffer(finishKernel, RayHitId, result);
            shader.Dispatch(finishKernel, 1, 1, 1);
        }

        internal GraphicsBuffer ResultBuffer => result;

        public void Dispose()
        {
            result?.Release();
            result = null;
        }
    }

    /// <summary>The answer of <see cref="AvbdRaycaster.Cast(float3, float3, float, System.Action{AvbdRayHit})"/>.</summary>
    public readonly struct AvbdRayHit
    {
        /// <summary>Whether the ray hit a body within the maximum distance.</summary>
        public readonly bool hit;

        /// <summary>Distance along the ray to the nearest hit, in meters (the maximum distance for a miss); 0 if the ray starts inside a box.</summary>
        public readonly float distance;

        /// <summary>The body that was hit; <see cref="BodyHandle.Invalid"/> for a miss, or if the body was removed before the answer arrived.</summary>
        public readonly BodyHandle body;

        /// <summary>The hit point in the frame of <see cref="body"/>: relative to its center of mass, along its axes (so it turns and moves with it). Zero without a body.</summary>
        public readonly float3 localPoint;

        /// <summary>The unit direction of the ray in the frame of <see cref="body"/>. Zero without a body.</summary>
        public readonly float3 localDirection;

        internal AvbdRayHit(bool hit, float distance, BodyHandle body, float3 localPoint, float3 localDirection)
        {
            this.hit = hit;
            this.distance = distance;
            this.body = body;
            this.localPoint = localPoint;
            this.localDirection = localDirection;
        }
    }
}
