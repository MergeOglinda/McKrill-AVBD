using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// Revolute (ball-socket) joint plus optional angle constraint between two bodies, with optional
    /// fracture. A port of upstream's Joint. If BodyA is null the joint pins BodyB to the world-space
    /// point rA.
    /// </summary>
    public sealed class AvbdJoint : AvbdForce
    {
        public float3 rA;
        public float3 rB;
        public float3 C0Lin;
        public float3 C0Ang;
        public float3 penaltyLin;
        public float3 penaltyAng;
        public float3 lambdaLin;
        public float3 lambdaAng;

        /// <summary>Linear stiffness. Infinity means a hard constraint (stabilized, with a multiplier).</summary>
        public float stiffnessLin;

        /// <summary>Angular stiffness. 0 disables the angle constraint; infinity makes it hard.</summary>
        public float stiffnessAng;

        /// <summary>Multiplier norm above which the joint breaks. Infinity = never.</summary>
        public float fracture;

        public float torqueArm;
        public bool broken;

        public AvbdJoint(
            ReferenceSolver solver, AvbdBody bodyA, AvbdBody bodyB, float3 rA, float3 rB,
            float stiffnessLin = float.PositiveInfinity, float stiffnessAng = 0f, float fracture = float.PositiveInfinity)
            : base(solver, bodyA, bodyB)
        {
            this.rA = rA;
            this.rB = rB;
            this.stiffnessLin = stiffnessLin;
            this.stiffnessAng = stiffnessAng;
            this.fracture = fracture;
            broken = false;

            penaltyLin = penaltyAng = float3.zero;
            lambdaLin = lambdaAng = float3.zero;
            torqueArm = math.lengthsq((bodyA != null ? bodyA.Size : float3.zero) + bodyB.Size);
        }

        // World-space anchor of body A (or the fixed world point) and of body B.
        float3 AnchorA() => BodyA != null ? AvbdQuat.Transform(BodyA.PositionLin, BodyA.PositionAng, rA) : rA;
        float3 AnchorB() => AvbdQuat.Transform(BodyB.PositionLin, BodyB.PositionAng, rB);
        quaternion RotationA() => BodyA != null ? BodyA.PositionAng : quaternion.identity;

        // Diagonal approximation of the geometric stiffness of a ball-socket joint (upstream's
        // geometricStiffnessBallSocket): diag(-v[k]) plus v added to column k.
        static float3x3 GeometricStiffnessBallSocket(int k, float3 v)
        {
            // Upstream: m[i][k] += v[i] for rows i = 0..2, i.e. v is added to column k.
            float d = -v[k];
            float3 c0 = new float3(d, 0f, 0f);
            float3 c1 = new float3(0f, d, 0f);
            float3 c2 = new float3(0f, 0f, d);
            if (k == 0) c0 += v;
            else if (k == 1) c1 += v;
            else c2 += v;
            return new float3x3(c0, c1, c2);
        }

        public override bool Initialize()
        {
            AvbdSolverParams p = Solver.Params;

            // Constraint function at the start of the step, C(x-).
            C0Lin = AnchorA() - AnchorB();
            C0Ang = Solver.QuatDifference(RotationA(), BodyB.PositionAng) * torqueArm;

            // Warm-start the dual variables and penalties (Eq. 19), penalties clamped to a safe range.
            lambdaLin = lambdaLin * p.alpha * p.gamma;
            lambdaAng = lambdaAng * p.alpha * p.gamma;
            penaltyLin = AvbdMat3.Clamp(penaltyLin * p.gamma, AvbdSolverParams.PenaltyMin, AvbdSolverParams.PenaltyMax);
            penaltyAng = AvbdMat3.Clamp(penaltyAng * p.gamma, AvbdSolverParams.PenaltyMin, AvbdSolverParams.PenaltyMax);

            // Clamp the penalty to the material stiffness.
            penaltyLin = math.min(penaltyLin, new float3(stiffnessLin));
            penaltyAng = math.min(penaltyAng, new float3(stiffnessAng));

            return !broken;
        }

        public override void UpdatePrimal(AvbdBody body, float alpha, ref AvbdPrimalSystem system)
        {
            bool isA = body == BodyA;

            // Linear constraint
            if (math.lengthsq(penaltyLin) > 0f)
            {
                float3x3 K = AvbdMat3.Diagonal(penaltyLin);
                float3 C = AnchorA() - AnchorB();

                // Stabilization
                if (float.IsInfinity(stiffnessLin))
                    C -= C0Lin * alpha;

                float3 F = penaltyLin * C + lambdaLin;

                // Jacobians for the body being solved.
                float3x3 jLin = isA ? float3x3.identity : -float3x3.identity;
                float3x3 jAng = isA
                    ? AvbdQuat.Skew(-AvbdQuat.Rotate(BodyA.PositionAng, rA))
                    : AvbdQuat.Skew(AvbdQuat.Rotate(BodyB.PositionAng, rB));

                float3x3 jLinT = math.transpose(jLin);
                float3x3 jAngT = math.transpose(jAng);
                float3x3 jAngTk = math.mul(jAngT, K);

                system.lhsLin += math.mul(math.mul(jLinT, K), jLin);
                system.lhsAng += math.mul(jAngTk, jAng);
                system.lhsCross += math.mul(jAngTk, jLin);

                // Diagonal approximation for higher-order terms.
                float3 r = isA ? AvbdQuat.Rotate(BodyA.PositionAng, rA) : -AvbdQuat.Rotate(BodyB.PositionAng, rB);
                float3x3 H =
                    GeometricStiffnessBallSocket(0, r) * F.x +
                    GeometricStiffnessBallSocket(1, r) * F.y +
                    GeometricStiffnessBallSocket(2, r) * F.z;
                system.lhsAng += AvbdMat3.Diagonalize(H);

                system.rhsLin += math.mul(jLinT, F);
                system.rhsAng += math.mul(jAngT, F);
            }

            // Angular constraint
            if (math.lengthsq(penaltyAng) > 0f)
            {
                float3x3 K = AvbdMat3.Diagonal(penaltyAng);
                float3 C = Solver.QuatDifference(RotationA(), BodyB.PositionAng) * torqueArm;

                // Stabilization
                if (float.IsInfinity(stiffnessAng))
                    C -= C0Ang * alpha;

                float3 F = penaltyAng * C + lambdaAng;

                float3x3 jAng = (isA ? float3x3.identity : -float3x3.identity) * torqueArm;
                float3x3 jAngT = math.transpose(jAng);

                system.lhsAng += math.mul(math.mul(jAngT, K), jAng);
                system.rhsAng += math.mul(jAngT, F);
            }
        }

        public override void UpdateDual(float alpha)
        {
            AvbdSolverParams p = Solver.Params;

            // Linear constraint
            if (math.lengthsq(penaltyLin) > 0f)
            {
                float3 C = AnchorA() - AnchorB();

                if (float.IsInfinity(stiffnessLin))
                {
                    // Stabilization
                    C -= C0Lin * alpha;

                    // Store the updated force as the multiplier.
                    lambdaLin = penaltyLin * C + lambdaLin;
                }

                // Ramp the penalty, clamped to the material stiffness (Eq. 16).
                penaltyLin = math.min(penaltyLin + math.abs(C) * p.betaLin,
                    new float3(math.min(stiffnessLin, AvbdSolverParams.PenaltyMax)));
            }

            // Angular constraint
            if (math.lengthsq(penaltyAng) > 0f)
            {
                float3 C = Solver.QuatDifference(RotationA(), BodyB.PositionAng) * torqueArm;

                if (float.IsInfinity(stiffnessAng))
                {
                    C -= C0Ang * alpha;
                    lambdaAng = penaltyAng * C + lambdaAng;
                }

                penaltyAng = math.min(penaltyAng + math.abs(C) * p.betaAng,
                    new float3(math.min(stiffnessAng, AvbdSolverParams.PenaltyMax)));
            }

            // Fracture test
            if (math.lengthsq(lambdaAng) > fracture * fracture)
            {
                penaltyLin = float3.zero;
                penaltyAng = float3.zero;
                lambdaLin = float3.zero;
                lambdaAng = float3.zero;
                broken = true;
            }
        }
    }
}
