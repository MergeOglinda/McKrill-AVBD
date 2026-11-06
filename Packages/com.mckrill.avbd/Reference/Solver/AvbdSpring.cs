using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// Standard spring between a point on A and a point on B (both in body space), a port of upstream's
    /// Spring. It is a plain force: no multiplier, no penalty, nothing to update in the dual step.
    /// </summary>
    public sealed class AvbdSpring : AvbdForce
    {
        public float3 rA;
        public float3 rB;
        public float rest;
        public float stiffness;

        /// <param name="rest">Rest length. A negative value means "use the current distance".</param>
        public AvbdSpring(ReferenceSolver solver, AvbdBody bodyA, AvbdBody bodyB, float3 rA, float3 rB, float stiffness, float rest = -1f)
            : base(solver, bodyA, bodyB)
        {
            this.rA = rA;
            this.rB = rB;
            this.stiffness = stiffness;
            this.rest = rest;

            if (this.rest < 0f)
            {
                float3 pA = AvbdQuat.Transform(bodyA.PositionLin, bodyA.PositionAng, this.rA);
                float3 pB = AvbdQuat.Transform(bodyB.PositionLin, bodyB.PositionAng, this.rB);
                this.rest = math.length(pA - pB);
            }
        }

        public override bool Initialize() => true;

        public override void UpdatePrimal(AvbdBody body, float alpha, ref AvbdPrimalSystem system)
        {
            float3 pA = AvbdQuat.Transform(BodyA.PositionLin, BodyA.PositionAng, rA);
            float3 pB = AvbdQuat.Transform(BodyB.PositionLin, BodyB.PositionAng, rB);
            float3 d = pA - pB;
            float dLen = math.length(d);
            if (dLen <= 1.0e-6f)
                return;

            float3 n = d / dLen;
            float C = dLen - rest;
            float f = stiffness * C;

            float3 jLin;
            float3 jAng;
            if (body == BodyA)
            {
                float3 rWorld = AvbdQuat.Rotate(BodyA.PositionAng, rA);
                jLin = n;
                jAng = math.cross(rWorld, n);
            }
            else
            {
                float3 rWorld = AvbdQuat.Rotate(BodyB.PositionAng, rB);
                jLin = -n;
                jAng = -math.cross(rWorld, n);
            }

            float3 F = jLin * f;
            float3 tau = jAng * f;
            float3x3 kLL = AvbdQuat.Outer(jLin, jLin) * stiffness;
            float3x3 kLA = AvbdQuat.Outer(jAng, jLin) * stiffness;
            float3x3 kAA = AvbdQuat.Outer(jAng, jAng) * stiffness;

            system.lhsLin += kLL;
            system.lhsAng += kAA;
            system.lhsCross += kLA;
            system.rhsLin += F;
            system.rhsAng += tau;
        }

        public override void UpdateDual(float alpha)
        {
        }
    }
}
