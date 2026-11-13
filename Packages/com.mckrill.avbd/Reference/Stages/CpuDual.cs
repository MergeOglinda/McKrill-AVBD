using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle for the dual update of stage D (<c>Joint.UpdateDual</c> in the reference solver): the same arithmetic as the
    /// <c>Dual</c> kernel in Shaders/Compute/AvbdDual.compute. It evaluates the constraints at the current body poses, so it runs after
    /// a primal sweep. The warm start that precedes the iterations is <see cref="CpuConstraintInit"/>. Springs and broken joints are left alone.
    /// </summary>
    internal static class CpuDual
    {
        public static void Run(CpuBodyArrays bodies, ConstraintRecord[] constraints, int count, in AvbdSolverSettings settings)
        {
            for (int i = 0; i < count; i++)
                Update(bodies, ref constraints[i], settings);
        }

        public static void Update(CpuBodyArrays bodies, ref ConstraintRecord c, in AvbdSolverSettings settings)
        {
            if (ConstraintFlagLayout.KindOf(c.ident.z) != ConstraintKind.Joint)
                return;
            ConstraintFlags flags = ConstraintFlagLayout.FlagsOf(c.ident.z);
            if ((flags & ConstraintFlags.Broken) != 0)
                return;

            float alpha = settings.alpha;
            bool hasA = c.ident.x != AvbdConst.Invalid;

            float3 posB = bodies.State[c.ident.y].position.xyz;
            quaternion rotB = new quaternion(bodies.State[c.ident.y].rotation);
            float3 anchorA = c.anchorA.xyz;
            quaternion rotA = quaternion.identity;
            if (hasA)
            {
                BodyStateRecord a = bodies.State[c.ident.x];
                rotA = new quaternion(a.rotation);
                anchorA = AvbdQuat.Transform(a.position.xyz, rotA, c.anchorA.xyz);
            }
            float3 anchorB = AvbdQuat.Transform(posB, rotB, c.anchorB.xyz);

            float3 penaltyLin = c.penaltyLin.xyz;
            float3 penaltyAng = c.penaltyAng.xyz;
            float3 lambdaLin = c.lambdaLin.xyz;
            float3 lambdaAng = c.lambdaAng.xyz;

            // Linear constraint
            if (math.lengthsq(penaltyLin) > 0f)
            {
                float3 C = anchorA - anchorB;
                float cap = settings.penaltyMax; // a hard constraint is limited by the maximum penalty only

                if ((flags & ConstraintFlags.HardLinear) != 0)
                {
                    C -= c.c0Lin.xyz * alpha;
                    lambdaLin = penaltyLin * C + lambdaLin;
                }
                else
                {
                    cap = math.min(c.anchorA.w, settings.penaltyMax);
                }

                penaltyLin = math.min(penaltyLin + math.abs(C) * settings.betaLin, new float3(cap));
            }

            // Angular constraint
            if (math.lengthsq(penaltyAng) > 0f)
            {
                float3 C = AvbdQuat.DifferenceRaw(rotA, rotB) * c.c0Lin.w;
                float cap = settings.penaltyMax;

                if ((flags & ConstraintFlags.HardAngular) != 0)
                {
                    C -= c.c0Ang.xyz * alpha;
                    lambdaAng = penaltyAng * C + lambdaAng;
                }
                else
                {
                    cap = math.min(c.anchorB.w, settings.penaltyMax);
                }

                penaltyAng = math.min(penaltyAng + math.abs(C) * settings.betaAng, new float3(cap));
            }

            // Fracture test
            uint packed = c.ident.z;
            if ((flags & ConstraintFlags.Fracturable) != 0 && math.lengthsq(lambdaAng) > c.c0Ang.w * c.c0Ang.w)
            {
                penaltyLin = float3.zero;
                penaltyAng = float3.zero;
                lambdaLin = float3.zero;
                lambdaAng = float3.zero;
                packed |= (uint)ConstraintFlags.Broken;
            }

            c.ident.z = packed;
            c.penaltyLin = new float4(penaltyLin, c.penaltyLin.w);
            c.penaltyAng = new float4(penaltyAng, c.penaltyAng.w);
            c.lambdaLin = new float4(lambdaLin, c.lambdaLin.w);
            c.lambdaAng = new float4(lambdaAng, c.lambdaAng.w);
        }
    }
}
