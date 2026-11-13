using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle of the per-step constraint warm start (<c>Joint.Initialize</c> in the reference solver): evaluates the constraint at the
    /// start-of-step pose (C0) and decays the multipliers and penalties. It must run after stage P, which writes the start pose, or
    /// before it, while the pose is still the start pose. The GPU version is added with stage D; stage V's tests use this oracle to
    /// prepare records, so that the primal kernel is tested against the reference solver on a fully initialized state.
    /// Springs have no state and are left alone. Broken joints are left alone.
    /// </summary>
    internal static class CpuConstraintInit
    {
        public static void Run(CpuBodyArrays bodies, ConstraintRecord[] constraints, int count, in AvbdSolverSettings settings)
        {
            for (int i = 0; i < count; i++)
            {
                ConstraintRecord c = constraints[i];
                if (ConstraintFlagLayout.KindOf(c.ident.z) != ConstraintKind.Joint)
                    continue;
                ConstraintFlags flags = ConstraintFlagLayout.FlagsOf(c.ident.z);
                if ((flags & ConstraintFlags.Broken) != 0)
                    continue;

                bool hasA = c.ident.x != AvbdConst.Invalid;
                BodyStateRecord b = bodies.State[c.ident.y];
                float3 rA = c.anchorA.xyz;
                float3 rB = c.anchorB.xyz;

                float3 anchorA = rA;
                quaternion rotA = quaternion.identity;
                if (hasA)
                {
                    BodyStateRecord a = bodies.State[c.ident.x];
                    rotA = new quaternion(a.startRotation);
                    anchorA = AvbdQuat.Transform(a.startPosition.xyz, rotA, rA);
                }
                quaternion rotB = new quaternion(b.startRotation);
                float3 anchorB = AvbdQuat.Transform(b.startPosition.xyz, rotB, rB);

                float torqueArm = c.c0Lin.w;
                c.c0Lin = new float4(anchorA - anchorB, torqueArm);
                c.c0Ang = new float4(AvbdQuat.DifferenceRaw(rotA, rotB) * torqueArm, c.c0Ang.w);

                // Warm start (Eq. 19), penalties clamped to a safe range.
                float3 lambdaLin = c.lambdaLin.xyz * settings.alpha * settings.gamma;
                float3 lambdaAng = c.lambdaAng.xyz * settings.alpha * settings.gamma;
                float3 penaltyLin = AvbdMat3.Clamp(c.penaltyLin.xyz * settings.gamma, settings.penaltyMin, settings.penaltyMax);
                float3 penaltyAng = AvbdMat3.Clamp(c.penaltyAng.xyz * settings.gamma, settings.penaltyMin, settings.penaltyMax);

                // Clamp to the material stiffness. A hard constraint has no cap here (its penalty is already limited by penaltyMax).
                if ((flags & ConstraintFlags.HardLinear) == 0)
                    penaltyLin = math.min(penaltyLin, new float3(c.anchorA.w));
                if ((flags & ConstraintFlags.HardAngular) == 0)
                    penaltyAng = math.min(penaltyAng, new float3(c.anchorB.w));

                c.lambdaLin = new float4(lambdaLin, 0f);
                c.lambdaAng = new float4(lambdaAng, 0f);
                c.penaltyLin = new float4(penaltyLin, 0f);
                c.penaltyAng = new float4(penaltyAng, 0f);
                constraints[i] = c;
            }
        }
    }
}
