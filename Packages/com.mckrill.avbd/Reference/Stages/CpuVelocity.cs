using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle for stage U: the same arithmetic as Shaders/Compute/AvbdVelocity.compute, on <see cref="CpuBodyArrays"/>.
    /// Tests check it against <see cref="ReferenceSolver"/> and check the GPU kernel against it.
    /// </summary>
    internal static class CpuVelocity
    {
        public static void Run(CpuBodyArrays bodies, in AvbdSolverSettings settings, int count)
        {
            float dt = settings.dt;

            for (int slot = 0; slot < count; slot++)
            {
                uint flags = bodies.Meta[slot].ident.x;
                if ((flags & (uint)BodyFlags.Free) != 0u)
                    continue;

                BodyMotionRecord v = bodies.Motion[slot];
                v.prevLinearVel = v.linearVel;

                if ((flags & (uint)BodyFlags.Kinematic) == 0u && bodies.Mass[slot].momentMass.w > 0f)
                {
                    BodyStateRecord s = bodies.State[slot];
                    v.linearVel = new float4((s.position.xyz - s.startPosition.xyz) / dt, 0f);
                    v.angularVel = new float4(AvbdQuat.DifferenceRaw(new quaternion(s.rotation), new quaternion(s.startRotation)) / dt, 0f);
                }

                bodies.Motion[slot] = v;
            }
        }
    }
}
