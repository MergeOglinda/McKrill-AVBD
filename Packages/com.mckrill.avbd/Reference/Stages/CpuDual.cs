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
        /// <param name="plastic">The plastic array, for joints with the Plastic flag; null if there are none.</param>
        public static void Run(CpuBodyArrays bodies, ConstraintRecord[] constraints, int count, in AvbdSolverSettings settings, PlasticRecord[] plastic = null)
        {
            for (int i = 0; i < count; i++)
                Update(bodies, ref constraints[i], settings, plastic);
        }

        public static void Update(CpuBodyArrays bodies, ref ConstraintRecord c, in AvbdSolverSettings settings, PlasticRecord[] plastic = null)
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
            float3 posA = float3.zero;
            if (hasA)
            {
                BodyStateRecord a = bodies.State[c.ident.x];
                rotA = new quaternion(a.rotation);
                posA = a.position.xyz;
                anchorA = AvbdQuat.Transform(posA, rotA, c.anchorA.xyz);
            }
            float3 anchorB = AvbdQuat.Transform(posB, rotB, c.anchorB.xyz);

            float3 penaltyLin = c.penaltyLin.xyz;
            float3 penaltyAng = c.penaltyAng.xyz;
            float3 lambdaLin = c.lambdaLin.xyz;
            float3 lambdaAng = c.lambdaAng.xyz;

            bool isPlastic = (flags & ConstraintFlags.Plastic) != 0;
            PlasticRecord plasticRecord = default;
            quaternion rest = quaternion.identity;
            if ((flags & (ConstraintFlags.Plastic | ConstraintFlags.PlasticLinear)) != 0)
            {
                plasticRecord = CpuPlastic.Of(plastic, c);
                if (isPlastic)
                    rest = new quaternion(plasticRecord.rest);
            }
            bool yielded = false;
            bool yieldedLin = false;

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

                // Yield of the linear row: a multiplier above the yield value is clamped to it and the joint keeps the stretch it has (the anchor on
                // body A moves below). A multiplier above the fracture threshold does not yield: the joint breaks.
                if ((flags & ConstraintFlags.PlasticLinear) != 0)
                {
                    float force = math.length(lambdaLin);
                    bool fractures = (flags & ConstraintFlags.FracturableLinear) != 0 && force > c.reserved.x;
                    if (force > plasticRecord.limits.z && !fractures)
                    {
                        lambdaLin *= plasticRecord.limits.z / force;
                        yieldedLin = true;
                    }
                }

                // A yielding row does not ramp its penalty.
                if (!yieldedLin)
                    penaltyLin = math.min(penaltyLin + math.abs(C) * settings.betaLin, new float3(cap));
            }

            // Angular constraint
            if (math.lengthsq(penaltyAng) > 0f)
            {
                float3 C = AvbdQuat.JointAngularError(rotA, rotB, rest) * c.c0Lin.w;
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

                // Yield: a multiplier above the yield value is clamped to it and the joint keeps the bend it has (the rest rotation moves below). A
                // multiplier above the fracture threshold does not yield: the joint breaks.
                if (isPlastic)
                {
                    float force = math.length(lambdaAng);
                    bool fractures = (flags & ConstraintFlags.Fracturable) != 0 && force > c.c0Ang.w;
                    if (force > plasticRecord.limits.x && !fractures)
                    {
                        lambdaAng *= plasticRecord.limits.x / force;
                        yielded = true;
                    }
                }

                // A yielding row does not ramp its penalty.
                if (!yielded)
                    penaltyAng = math.min(penaltyAng + math.abs(C) * settings.betaAng, new float3(cap));
            }

            bool bentOff = false;
            if (yielded)
            {
                rest = AvbdQuat.RestOf(rotA, rotB);
                if ((flags & ConstraintFlags.BendBreak) != 0)
                    bentOff = AvbdQuat.BendAngle(new quaternion(plasticRecord.restStart), rest) > plasticRecord.limits.y;
            }

            bool stretchedOff = false;
            if (yieldedLin)
            {
                // The anchor on body A moves to where the anchor on body B is now (in the body space of A; a joint to the world keeps a world point).
                float3 moved = hasA ? AvbdQuat.Rotate(math.conjugate(rotA), anchorB - posA) : anchorB;
                c.anchorA = new float4(moved, c.anchorA.w);
                if ((flags & ConstraintFlags.StretchBreak) != 0)
                    stretchedOff = math.length(moved - plasticRecord.reserved.xyz) > plasticRecord.limits.w;
                c.c0Lin = new float4(0f, 0f, 0f, c.c0Lin.w); // the stabilization reference of this step must not pull the stretch back
            }

            // Fracture tests of the two rows, and the bend and stretch limits of a plastic joint
            uint packed = c.ident.z;
            if (bentOff || stretchedOff
                || ((flags & ConstraintFlags.Fracturable) != 0 && math.lengthsq(lambdaAng) > c.c0Ang.w * c.c0Ang.w)
                || ((flags & ConstraintFlags.FracturableLinear) != 0 && math.lengthsq(lambdaLin) > c.reserved.x * c.reserved.x))
            {
                penaltyLin = float3.zero;
                penaltyAng = float3.zero;
                lambdaLin = float3.zero;
                lambdaAng = float3.zero;
                packed |= (uint)ConstraintFlags.Broken;
            }

            if (yielded)
            {
                // The joint holds the bend it has now: C0, the stabilization reference of this step, must not pull it back.
                c.c0Ang = new float4(0f, 0f, 0f, c.c0Ang.w);
                CpuPlastic.Of(plastic, c).rest = rest.value;
            }

            c.ident.z = packed;
            c.penaltyLin = new float4(penaltyLin, c.penaltyLin.w);
            c.penaltyAng = new float4(penaltyAng, c.penaltyAng.w);
            c.lambdaLin = new float4(lambdaLin, c.lambdaLin.w);
            c.lambdaAng = new float4(lambdaAng, c.lambdaAng.w);
        }
    }
}
