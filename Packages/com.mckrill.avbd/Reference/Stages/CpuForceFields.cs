using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle for the force fields of stage P: the same arithmetic as ForceFieldAcceleration in Shaders/Compute/AvbdPredict.compute.
    /// </summary>
    internal static class CpuForceFieldMath
    {
        // Mirror the kinds and the flag of Shaders/Include/AvbdForceFields.hlsl.
        const uint None = 0u;
        const uint Constant = 1u;
        const uint GravityScale = 2u;
        const uint Drag = 3u;
        const uint Spring = 5u;
        const uint BodySpace = 1u;
        const float FixedPoint = 1024f; // AVBD_FORCE_FIXED_POINT

        /// <summary>
        /// What ForceFieldClear and ForceFieldAccumulate compute: 4 ints per field, for each spring the sum of the offsets (position - target) of the bodies of its
        /// group in 1/1024 m, and their number. Run before the bodies are predicted.
        /// </summary>
        public static int[] SpringSums(CpuBodyArrays bodies, int count, ForceFieldRecord[] fields, int slots)
        {
            var sums = new int[fields.Length * 4];
            for (int slot = 0; slot < count; slot++)
            {
                if ((bodies.Flags[slot] & ((uint)BodyFlags.Free | (uint)BodyFlags.Kinematic)) != 0u)
                    continue;
                if (!(bodies.Mass[slot].momentMass.w > 0f))
                    continue;

                uint groupMask = bodies.Meta[slot].ident.y;
                float3 position = bodies.State[slot].position.xyz;
                for (int i = 0; i < slots; i++)
                {
                    if (fields[i].ident.x != Spring || (groupMask & fields[i].ident.y) == 0u)
                        continue;
                    int3 offset = (int3)math.floor((position - fields[i].value.xyz) * FixedPoint + 0.5f);
                    sums[i * 4] += offset.x;
                    sums[i * 4 + 1] += offset.y;
                    sums[i * 4 + 2] += offset.z;
                    sums[i * 4 + 3]++;
                }
            }
            return sums;
        }

        /// <summary>
        /// The acceleration the fields give a body, in m/s^2. <paramref name="jointCount"/> is the number of joints and springs in its constraint list (the
        /// entries after the contacts), which a drag field with shielding turns into an exposure.
        /// </summary>
        /// <param name="fields">The table (<see cref="AvbdSolver.AddForceField"/>), looked at in [0, slots); empty slots have kind 0.</param>
        /// <param name="springSums">The result of <see cref="SpringSums"/>; may be null if there is no spring field.</param>
        public static float3 Acceleration(uint groupMask, uint jointCount, float mass, quaternion rotation, float3 linearVel, float3 gravity, float dt, ForceFieldRecord[] fields, int slots,
            int[] springSums = null)
        {
            float3 acceleration = float3.zero;
            for (int i = 0; i < slots; i++)
            {
                ForceFieldRecord f = fields[i];
                uint kind = f.ident.x;
                if (kind == None || (groupMask & f.ident.y) == 0u)
                    continue;

                if (kind == Constant)
                {
                    float3 force = f.value.xyz;
                    if ((f.ident.z & BodySpace) != 0u)
                        force = AvbdQuat.Rotate(rotation, force);
                    acceleration += force / mass;
                }
                else if (kind == GravityScale)
                {
                    acceleration += gravity * (f.coefficients.x - 1f);
                }
                else if (kind == Spring)
                {
                    int members = springSums[i * 4 + 3];
                    if (members == 0)
                        continue;
                    float3 offsetSum = new float3(springSums[i * 4], springSums[i * 4 + 1], springSums[i * 4 + 2]);
                    float3 toTarget = -offsetSum / (FixedPoint * members);

                    float stiffness = f.coefficients.x;
                    if (math.dot(f.axis.xyz, f.axis.xyz) > 0f)
                    {
                        float3 fromTarget = -toTarget;
                        float3 across = fromTarget - math.dot(fromTarget, f.axis.xyz) * f.axis.xyz;
                        stiffness *= 1f + f.coefficients.z * math.length(across);
                    }

                    float3 springAcceleration = stiffness * toTarget / mass;
                    float springLimit = math.length(toTarget) / (dt * dt);
                    float springLength = math.length(springAcceleration);
                    if (springLength > springLimit)
                        springAcceleration *= springLimit / springLength;

                    float3 dampingAcceleration = -2f * f.coefficients.y * math.sqrt(stiffness / mass) * linearVel;
                    float dampingLimit = math.length(linearVel) / dt;
                    float dampingLength = math.length(dampingAcceleration);
                    if (dampingLength > dampingLimit)
                        dampingAcceleration *= dampingLimit / dampingLength;

                    acceleration += springAcceleration + dampingAcceleration;
                }
                else
                {
                    float3 relative = linearVel - f.value.xyz;
                    float speedSq = math.dot(relative, relative);
                    float speed = math.sqrt(speedSq);
                    if (!(speed > 1e-6f))
                        continue;
                    float3 direction = relative / speed;

                    float3 force;
                    if (kind == Drag)
                    {
                        force = -(f.coefficients.x + f.coefficients.y * speed) * relative;
                        if (f.coefficients.z > 0f)
                            force *= math.saturate((f.coefficients.z - jointCount) / math.max(f.coefficients.z - f.coefficients.w, 1e-3f));
                    }
                    else
                    {
                        float3 wingNormal = AvbdQuat.Rotate(rotation, f.axis.xyz);
                        float pressure = f.coefficients.z * speedSq;
                        force = -pressure * (f.coefficients.x * math.dot(direction, wingNormal) * wingNormal + f.coefficients.y * direction);
                    }

                    float3 fieldAcceleration = force / mass;
                    float limit = speed / dt;
                    float fieldLength = math.length(fieldAcceleration);
                    if (fieldLength > limit)
                        fieldAcceleration *= limit / fieldLength;
                    acceleration += fieldAcceleration;
                }
            }
            return acceleration;
        }
    }
}
