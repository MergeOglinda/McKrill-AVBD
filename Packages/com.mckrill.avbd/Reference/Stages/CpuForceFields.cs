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
        const uint BodySpace = 1u;

        /// <summary>
        /// The acceleration the fields give a body, in m/s^2.
        /// </summary>
        /// <param name="fields">The table (<see cref="AvbdSolver.AddForceField"/>), looked at in [0, slots); empty slots have kind 0.</param>
        public static float3 Acceleration(uint groupMask, float mass, quaternion rotation, float3 linearVel, float3 gravity, float dt, ForceFieldRecord[] fields, int slots)
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
