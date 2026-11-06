using System;
using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// Collision manifold between two bodies: up to eight frictional contact points sharing one contact
    /// basis. A port of upstream's Manifold. Contact multipliers and penalties persist from step to step
    /// for contacts whose feature key matches, which is how warm starting works; the manifold object
    /// itself is removed as soon as a step finds no contacts.
    /// </summary>
    public sealed class AvbdManifold : AvbdForce
    {
        public const int MaxContacts = BoxCollision.MaxContacts;

        /// <summary>Contact point information for a single contact.</summary>
        public struct Contact
        {
            /// <summary>Identifies the contact feature across frames (packed face/edge ids).</summary>
            public int featureKey;

            /// <summary>Contact offset in A's local space, relative to its center.</summary>
            public float3 rA;

            /// <summary>Contact offset in B's local space, relative to its center.</summary>
            public float3 rB;

            /// <summary>Constraint value at the start of the step, in the contact basis, including the margin.</summary>
            public float3 C0;

            /// <summary>Penalty per row: normal, tangent 1, tangent 2.</summary>
            public float3 penalty;

            /// <summary>Multiplier per row: normal, tangent 1, tangent 2.</summary>
            public float3 lambda;

            /// <summary>True if the contact held static friction in the last dual update.</summary>
            public bool stick;
        }

        public readonly Contact[] Contacts = new Contact[MaxContacts];

        /// <summary>
        /// Contact basis as a matrix: row 0 is the normal (pointing from B to A), rows 1 and 2 are the
        /// tangents. Left unchanged when collision exits early, like upstream.
        /// </summary>
        public float3x3 Basis;

        public int NumContacts;
        public float Friction;

        readonly Contact[] scratch = new Contact[MaxContacts];

        public AvbdManifold(ReferenceSolver solver, AvbdBody bodyA, AvbdBody bodyB)
            : base(solver, bodyA, bodyB)
        {
            NumContacts = 0;
        }

        public override bool Initialize()
        {
            AvbdSolverParams p = Solver.Params;

            Friction = math.sqrt(BodyA.Friction * BodyB.Friction);

            // Compute new contacts.
            Array.Clear(scratch, 0, scratch.Length);
            int newNumContacts = BoxCollision.Collide(BodyA, BodyB, scratch, ref Basis);

            // Merge old contact data with new contacts.
            for (int i = 0; i < newNumContacts; i++)
            {
                for (int j = 0; j < NumContacts; j++)
                {
                    if (scratch[i].featureKey == Contacts[j].featureKey)
                    {
                        float3 newRA = scratch[i].rA;
                        float3 newRB = scratch[i].rB;
                        scratch[i] = Contacts[j];

                        // If there was no static friction last frame, use the new contact point locations.
                        if (!Contacts[j].stick)
                        {
                            scratch[i].rA = newRA;
                            scratch[i].rB = newRB;
                        }
                        break;
                    }
                }
            }

            // Copy new contacts into the manifold.
            NumContacts = newNumContacts;
            for (int i = 0; i < NumContacts; i++)
                Contacts[i] = scratch[i];

            // Compute the error at q- and warm-start the dual variables and penalties.
            for (int i = 0; i < NumContacts; i++)
            {
                ref Contact c = ref Contacts[i];

                float3 xA = AvbdQuat.Transform(BodyA.PositionLin, BodyA.PositionAng, c.rA);
                float3 xB = AvbdQuat.Transform(BodyB.PositionLin, BodyB.PositionAng, c.rB);
                c.C0 = math.mul(Basis, xA - xB) + new float3(AvbdSolverParams.CollisionMargin, 0f, 0f);

                // Eq. 19; penalty clamped to a safe range.
                c.lambda = c.lambda * p.alpha * p.gamma;
                c.penalty = AvbdMat3.Clamp(c.penalty * p.gamma, AvbdSolverParams.PenaltyMin, AvbdSolverParams.PenaltyMax);
            }

            return NumContacts > 0;
        }

        // Shared by the primal and dual updates: the Taylor-approximated constraint C, the clamped
        // force F = K C + lambda (normal force <= 0, friction clamped to the cone), and the data the
        // dual update needs to decide whether the friction rows are inside the cone.
        void EvaluateContact(
            in Contact c, float alpha,
            float3 dqALin, float3 dqAAng, float3 dqBLin, float3 dqBAng,
            out float3x3 jALin, out float3x3 jBLin, out float3x3 jAAng, out float3x3 jBAng,
            out float3 C, out float3 F, out float bounds, out float frictionScale)
        {
            float3 rAWorld = AvbdQuat.Rotate(BodyA.PositionAng, c.rA);
            float3 rBWorld = AvbdQuat.Rotate(BodyB.PositionAng, c.rB);

            // Jacobians (Sec 4). Linear parts are +/- the basis; angular rows are r x (linear row).
            jALin = Basis;
            jBLin = -Basis;
            jAAng = AvbdMat3.FromRows(
                math.cross(rAWorld, AvbdMat3.Row(jALin, 0)),
                math.cross(rAWorld, AvbdMat3.Row(jALin, 1)),
                math.cross(rAWorld, AvbdMat3.Row(jALin, 2)));
            jBAng = AvbdMat3.FromRows(
                math.cross(rBWorld, AvbdMat3.Row(jBLin, 0)),
                math.cross(rBWorld, AvbdMat3.Row(jBLin, 1)),
                math.cross(rBWorld, AvbdMat3.Row(jBLin, 2)));

            C = c.C0 * (1f - alpha)
                + math.mul(jALin, dqALin) + math.mul(jBLin, dqBLin)
                + math.mul(jAAng, dqAAng) + math.mul(jBAng, dqBAng);

            F = c.penalty * C + c.lambda;

            // Clamp the normal force.
            F.x = math.min(F.x, 0f);

            // Clamp the norm of the friction force to the friction cone.
            bounds = math.abs(F.x) * Friction;
            frictionScale = math.length(F.yz);
            if (frictionScale > bounds && frictionScale > 0f)
            {
                F.y *= bounds / frictionScale;
                F.z *= bounds / frictionScale;
            }
        }

        public override void UpdatePrimal(AvbdBody body, float alpha, ref AvbdPrimalSystem system)
        {
            float3 dqALin = BodyA.PositionLin - BodyA.InitialLin;
            float3 dqAAng = Solver.QuatDifference(BodyA.PositionAng, BodyA.InitialAng);
            float3 dqBLin = BodyB.PositionLin - BodyB.InitialLin;
            float3 dqBAng = Solver.QuatDifference(BodyB.PositionAng, BodyB.InitialAng);

            bool isA = body == BodyA;

            for (int i = 0; i < NumContacts; i++)
            {
                EvaluateContact(
                    in Contacts[i], alpha, dqALin, dqAAng, dqBLin, dqBAng,
                    out float3x3 jALin, out float3x3 jBLin, out float3x3 jAAng, out float3x3 jBAng,
                    out _, out float3 F, out _, out _);

                float3x3 K = AvbdMat3.Diagonal(Contacts[i].penalty);

                // Jacobian for the body being solved.
                float3x3 jLin = isA ? jALin : jBLin;
                float3x3 jAng = isA ? jAAng : jBAng;

                float3x3 jLinT = math.transpose(jLin);
                float3x3 jAngT = math.transpose(jAng);
                float3x3 jAngTk = math.mul(jAngT, K);

                // Stamp into the left-hand side.
                system.lhsLin += math.mul(math.mul(jLinT, K), jLin);
                system.lhsAng += math.mul(jAngTk, jAng);
                system.lhsCross += math.mul(jAngTk, jLin);

                // Stamp into the right-hand side.
                system.rhsLin += math.mul(jLinT, F);
                system.rhsAng += math.mul(jAngT, F);
            }
        }

        public override void UpdateDual(float alpha)
        {
            AvbdSolverParams p = Solver.Params;
            float3 dqALin = BodyA.PositionLin - BodyA.InitialLin;
            float3 dqAAng = Solver.QuatDifference(BodyA.PositionAng, BodyA.InitialAng);
            float3 dqBLin = BodyB.PositionLin - BodyB.InitialLin;
            float3 dqBAng = Solver.QuatDifference(BodyB.PositionAng, BodyB.InitialAng);

            for (int i = 0; i < NumContacts; i++)
            {
                EvaluateContact(
                    in Contacts[i], alpha, dqALin, dqAAng, dqBLin, dqBAng,
                    out _, out _, out _, out _,
                    out float3 C, out float3 F, out float bounds, out float frictionScale);

                ref Contact c = ref Contacts[i];

                // Store the updated force.
                c.lambda = F;

                // Ramp the penalties and clamp them to the material stiffness if we are within the
                // force bounds (Eq. 16). frictionScale is the value before the cone clamp, as upstream.
                if (F.x < 0f)
                    c.penalty.x = math.min(c.penalty.x + p.betaLin * math.abs(C.x), AvbdSolverParams.PenaltyMax);
                if (frictionScale <= bounds)
                {
                    c.penalty.y = math.min(c.penalty.y + p.betaLin * math.abs(C.y), AvbdSolverParams.PenaltyMax);
                    c.penalty.z = math.min(c.penalty.z + p.betaLin * math.abs(C.z), AvbdSolverParams.PenaltyMax);
                    c.stick = math.length(C.yz) < AvbdSolverParams.StickThreshold;
                }
            }
        }
    }
}
