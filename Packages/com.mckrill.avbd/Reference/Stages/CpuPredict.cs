using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle for stage P: the same arithmetic as Shaders/Compute/AvbdPredict.compute, on <see cref="CpuBodyArrays"/>.
    /// Tests check it against <see cref="ReferenceSolver"/> and check the GPU kernel against it.
    /// </summary>
    internal static class CpuPredict
    {
        public static void Run(CpuBodyArrays bodies, in AvbdSolverSettings settings, int count)
        {
            float dt = settings.dt;
            float3 gravity = settings.gravity;

            for (int slot = 0; slot < count; slot++)
            {
                uint flags = bodies.Meta[slot].ident.x;
                if ((flags & ((uint)BodyFlags.Free | (uint)BodyFlags.Kinematic)) != 0u)
                    continue;
                if (!(bodies.Mass[slot].momentMass.w > 0f))
                    continue;

                BodyStateRecord s = bodies.State[slot];
                BodyMotionRecord v = bodies.Motion[slot];
                float3 position = s.position.xyz;
                float3 linearVel = v.linearVel.xyz;
                float3 angularVel = v.angularVel.xyz;

                float3 inertialPos = position + linearVel * dt;
                inertialPos += gravity * (dt * dt);
                float4 predictedRot = AvbdQuat.IntegrateAngular(new quaternion(s.rotation), angularVel * dt).value;

                bodies.Inertial[slot] = new BodyInertialRecord { position = new float4(inertialPos, 0f), rotation = predictedRot };

                float3 accel = (linearVel - v.prevLinearVel.xyz) / dt;
                float gravityLength = math.length(gravity);
                float accelWeight = gravityLength > 0f
                    ? math.clamp(math.dot(accel, gravity) / (gravityLength * gravityLength), 0f, 1f)
                    : 0f;
                if (!math.isfinite(accelWeight))
                    accelWeight = 0f;

                s.startPosition = s.position;
                s.startRotation = s.rotation;
                s.position = new float4(position + linearVel * dt + gravity * (accelWeight * dt * dt), 0f);
                s.rotation = predictedRot;
                bodies.State[slot] = s;
            }
        }
    }
}
