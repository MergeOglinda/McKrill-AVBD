using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle for stage P: the same arithmetic as Shaders/Compute/AvbdPredict.compute, on <see cref="CpuBodyArrays"/>.
    /// Tests check it against <see cref="ReferenceSolver"/> and check the GPU kernel against it.
    /// </summary>
    internal static class CpuPredict
    {
        // Joints and springs in the body's constraint list: the entries (range.y) after the contacts (range.z). Mirrors the shielding of a drag field in the kernel.
        static uint JointCount(CpuBodyArrays bodies, int slot)
        {
            uint4 range = bodies.Range[slot].range;
            return range.y > range.z ? range.y - range.z : 0u;
        }

        /// <param name="explosions">The explosions queued for the step (<see cref="AvbdSolver.Explode"/>), or null for none.</param>
        /// <param name="forceFields">The table of force fields (<see cref="AvbdSolver.AddForceField"/>), or null for none; looked at in [0, forceFieldSlots).</param>
        public static void Run(CpuBodyArrays bodies, in AvbdSolverSettings settings, int count, CpuExplosion[] explosions = null, int explosionCount = 0,
            ForceFieldRecord[] forceFields = null, int forceFieldSlots = 0)
        {
            float dt = settings.dt;
            float3 gravity = settings.gravity;
            // The centroids of the springs, over the poses at the start of the step (the kernel ForceFieldAccumulate runs before the predict kernel).
            int[] springSums = forceFields != null && forceFieldSlots > 0 ? CpuForceFieldMath.SpringSums(bodies, count, forceFields, forceFieldSlots) : null;

            for (int slot = 0; slot < count; slot++)
            {
                uint flags = bodies.Flags[slot];
                if ((flags & ((uint)BodyFlags.Free | (uint)BodyFlags.Kinematic)) != 0u)
                    continue;
                if (!(bodies.Mass[slot].momentMass.w > 0f))
                    continue;

                BodyStateRecord s = bodies.State[slot];
                BodyMotionRecord v = bodies.Motion[slot];
                float3 position = s.position.xyz;
                if (explosions != null && explosionCount > 0)
                {
                    float3 explodedLinear = v.linearVel.xyz;
                    float3 explodedPrev = v.prevLinearVel.xyz;
                    float3 explodedAngular = v.angularVel.xyz;
                    if (CpuExplosionMath.Apply((uint)slot, bodies.Mass[slot].momentMass.w, bodies.Mass[slot].momentMass.xyz, bodies.Shape[slot].sizeFriction.xyz,
                            new quaternion(s.rotation), position, explosions, explosionCount, ref explodedLinear, ref explodedPrev, ref explodedAngular))
                    {
                        v.linearVel = new float4(explodedLinear, 0f);
                        v.prevLinearVel = new float4(explodedPrev, 0f);
                        v.angularVel = new float4(explodedAngular, 0f);
                        bodies.Motion[slot] = v;
                    }
                }
                float3 linearVel = v.linearVel.xyz;
                float3 angularVel = v.angularVel.xyz;

                float3 inertialPos = position + linearVel * dt;
                inertialPos += gravity * (dt * dt);
                if (forceFields != null && forceFieldSlots > 0)
                {
                    inertialPos += CpuForceFieldMath.Acceleration(bodies.Meta[slot].ident.y, JointCount(bodies, slot), bodies.Mass[slot].momentMass.w, new quaternion(s.rotation),
                        linearVel, gravity, dt, forceFields, forceFieldSlots, springSums) * (dt * dt);
                }
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
