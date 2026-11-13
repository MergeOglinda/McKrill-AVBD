using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle for stage V: the same arithmetic as Shaders/Compute/AvbdPrimal.compute, on <see cref="CpuBodyArrays"/> and the
    /// constraint records, in the order of the colored sweep. Tests check it against <see cref="ReferenceSolver"/> (one iteration in
    /// <see cref="GaussSeidelMode.Colored"/>) and check the GPU kernel against it.
    /// </summary>
    internal static class CpuPrimal
    {
        /// <summary>Runs one primal sweep: every color in turn, every body of the color. Returns the number of failed solves.</summary>
        public static int Sweep(CpuBodyArrays bodies, ConstraintRecord[] constraints, ConstraintTopology topology, in AvbdSolverSettings settings)
        {
            int failures = 0;
            for (int color = 0; color < topology.ColorCount; color++)
            {
                int end = topology.ColorOffsets[color + 1];
                for (int k = topology.ColorOffsets[color]; k < end; k++)
                    if (!SolveBody(bodies, constraints, topology, (int)topology.ColorBodies[k], settings))
                        failures++;
            }
            return failures;
        }

        // Returns false only if the 6x6 solve failed. A skipped body (free, kinematic, massless) counts as success.
        public static bool SolveBody(CpuBodyArrays bodies, ConstraintRecord[] constraints, ConstraintTopology topology, int slot, in AvbdSolverSettings settings)
        {
            uint flagsWord = bodies.Meta[slot].ident.x;
            if ((flagsWord & ((uint)BodyFlags.Free | (uint)BodyFlags.Kinematic)) != 0u)
                return true;
            float4 momentMass = bodies.Mass[slot].momentMass;
            float mass = momentMass.w;
            if (!(mass > 0f))
                return true;

            float dt = settings.dt;
            float dtSq = dt * dt;

            BodyStateRecord state = bodies.State[slot];
            BodyInertialRecord inertial = bodies.Inertial[slot];
            float3 position = state.position.xyz;
            quaternion rotation = new quaternion(state.rotation);

            var sys = new AvbdPrimalSystem
            {
                lhsLin = AvbdMat3.Diagonal(mass / dtSq, mass / dtSq, mass / dtSq),
                lhsAng = AvbdMat3.Diagonal(momentMass.x / dtSq, momentMass.y / dtSq, momentMass.z / dtSq),
                lhsCross = float3x3.zero,
                rhsLin = (mass / dtSq) * (position - inertial.position.xyz),
                rhsAng = (momentMass.xyz / dtSq) * AvbdQuat.DifferenceRaw(rotation, new quaternion(inertial.rotation)),
            };

            uint offset = bodies.Range[slot].range.x;
            uint count = bodies.Range[slot].range.y;
            for (uint e = 0; e < count; e++)
            {
                ConstraintRecord c = constraints[topology.List[offset + e]];
                uint flags = c.ident.z & ConstraintFlagLayout.FlagsMask;
                switch (ConstraintFlagLayout.KindOf(c.ident.z))
                {
                    case ConstraintKind.Joint:
                        StampJoint(bodies, c, (ConstraintFlags)flags, (uint)slot, position, rotation, settings, ref sys);
                        break;
                    case ConstraintKind.Spring:
                        StampSpring(bodies, c, (uint)slot, position, rotation, ref sys);
                        break;
                }
            }

            var a = new Mat6
            {
                ll = sys.lhsLin,
                aa = sys.lhsAng,
                al = sys.lhsCross,
                la = math.transpose(sys.lhsCross),
            };
            if (!a.TrySolveLDLT(new Vec6(-sys.rhsLin, -sys.rhsAng), out Vec6 dx))
                return false;

            state.position = new float4(position + dx.lin, 0f);
            state.rotation = AvbdQuat.IntegrateAngular(rotation, dx.ang).value;
            bodies.State[slot] = state;
            return true;
        }

        static void LoadPose(CpuBodyArrays bodies, uint id, uint self, float3 selfPos, quaternion selfRot, out float3 pos, out quaternion rot)
        {
            if (id == self)
            {
                pos = selfPos;
                rot = selfRot;
            }
            else if (id == AvbdConst.Invalid)
            {
                pos = float3.zero;
                rot = quaternion.identity;
            }
            else
            {
                pos = bodies.State[id].position.xyz;
                rot = new quaternion(bodies.State[id].rotation);
            }
        }

        static void StampJoint(
            CpuBodyArrays bodies, in ConstraintRecord c, ConstraintFlags flags, uint slot, float3 selfPos, quaternion selfRot,
            in AvbdSolverSettings settings, ref AvbdPrimalSystem sys)
        {
            float alpha = settings.alpha;
            bool hasA = c.ident.x != AvbdConst.Invalid;
            bool isA = c.ident.x == slot;

            LoadPose(bodies, c.ident.x, slot, selfPos, selfRot, out float3 posA, out quaternion rotA);
            LoadPose(bodies, c.ident.y, slot, selfPos, selfRot, out float3 posB, out quaternion rotB);

            float3 rA = c.anchorA.xyz;
            float3 rB = c.anchorB.xyz;
            float torqueArm = c.c0Lin.w;

            float3 rWorldA = AvbdQuat.Rotate(rotA, rA);
            float3 rWorldB = AvbdQuat.Rotate(rotB, rB);
            float3 anchorA = hasA ? rWorldA + posA : rA;
            float3 anchorB = rWorldB + posB;

            float3 penaltyLin = c.penaltyLin.xyz;
            float3 penaltyAng = c.penaltyAng.xyz;

            // Linear constraint
            if (math.lengthsq(penaltyLin) > 0f)
            {
                float3x3 K = AvbdMat3.Diagonal(penaltyLin);
                float3 C = anchorA - anchorB;

                if ((flags & ConstraintFlags.HardLinear) != 0)
                    C -= c.c0Lin.xyz * alpha;

                float3 F = penaltyLin * C + c.lambdaLin.xyz;

                float3x3 jLin = isA ? float3x3.identity : -float3x3.identity;
                float3x3 jAng = isA ? AvbdQuat.Skew(-rWorldA) : AvbdQuat.Skew(rWorldB);

                float3x3 jLinT = math.transpose(jLin);
                float3x3 jAngT = math.transpose(jAng);
                float3x3 jAngTk = math.mul(jAngT, K);

                sys.lhsLin += math.mul(math.mul(jLinT, K), jLin);
                sys.lhsAng += math.mul(jAngTk, jAng);
                sys.lhsCross += math.mul(jAngTk, jLin);

                // -dot(F, r) I + r F^T: the sum of the geometric stiffness terms (see the kernel).
                float3 r = isA ? rWorldA : -rWorldB;
                float3x3 H = -math.dot(F, r) * float3x3.identity + AvbdQuat.Outer(r, F);
                sys.lhsAng += AvbdMat3.Diagonalize(H);

                sys.rhsLin += math.mul(jLinT, F);
                sys.rhsAng += math.mul(jAngT, F);
            }

            // Angular constraint
            if (math.lengthsq(penaltyAng) > 0f)
            {
                float3x3 K = AvbdMat3.Diagonal(penaltyAng);
                float3 C = AvbdQuat.DifferenceRaw(rotA, rotB) * torqueArm;

                if ((flags & ConstraintFlags.HardAngular) != 0)
                    C -= c.c0Ang.xyz * alpha;

                float3 F = penaltyAng * C + c.lambdaAng.xyz;

                float3x3 jAng = (isA ? float3x3.identity : -float3x3.identity) * torqueArm;
                float3x3 jAngT = math.transpose(jAng);

                sys.lhsAng += math.mul(math.mul(jAngT, K), jAng);
                sys.rhsAng += math.mul(jAngT, F);
            }
        }

        static void StampSpring(CpuBodyArrays bodies, in ConstraintRecord c, uint slot, float3 selfPos, quaternion selfRot, ref AvbdPrimalSystem sys)
        {
            bool isA = c.ident.x == slot;

            LoadPose(bodies, c.ident.x, slot, selfPos, selfRot, out float3 posA, out quaternion rotA);
            LoadPose(bodies, c.ident.y, slot, selfPos, selfRot, out float3 posB, out quaternion rotB);

            float stiffness = c.anchorA.w;
            float rest = c.anchorB.w;

            float3 rWorldA = AvbdQuat.Rotate(rotA, c.anchorA.xyz);
            float3 rWorldB = AvbdQuat.Rotate(rotB, c.anchorB.xyz);
            float3 d = (rWorldA + posA) - (rWorldB + posB);
            float dLen = math.length(d);
            if (dLen <= 1.0e-6f)
                return;

            float3 n = d / dLen;
            float f = stiffness * (dLen - rest);

            float3 jLin;
            float3 jAng;
            if (isA)
            {
                jLin = n;
                jAng = math.cross(rWorldA, n);
            }
            else
            {
                jLin = -n;
                jAng = -math.cross(rWorldB, n);
            }

            sys.lhsLin += AvbdQuat.Outer(jLin, jLin) * stiffness;
            sys.lhsAng += AvbdQuat.Outer(jAng, jAng) * stiffness;
            sys.lhsCross += AvbdQuat.Outer(jAng, jLin) * stiffness;
            sys.rhsLin += jLin * f;
            sys.rhsAng += jAng * f;
        }
    }
}
