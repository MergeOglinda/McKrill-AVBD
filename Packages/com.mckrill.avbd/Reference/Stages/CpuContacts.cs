using System;
using System.Collections.Generic;
using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle of the contact stages: the narrow phase over all pairs (Shaders/Compute/AvbdNarrowphase.compute), the topology with
    /// contacts, and the contact dual update (the DualContact kernel of AvbdDual.compute). The contact records live in the same array as
    /// the joints and springs, from <see cref="ContactBase"/> on, in the pair layout of <see cref="ContactLayout"/>.
    /// The collision itself is <see cref="BoxCollision"/>, the reference solver's port of upstream's collide(); the merge with the old
    /// manifold and the warm start follow <see cref="AvbdManifold.Initialize"/>.
    /// </summary>
    internal sealed class CpuContacts
    {
        public readonly int ContactBase;
        public readonly int ContactBodies;

        /// <summary>Contacts per pair, indexed like <see cref="ContactLayout.PairIndex"/>.</summary>
        public readonly uint[] PairCounts;

        readonly bool[] excluded;
        readonly AvbdBody bodyA = new AvbdBody(new float3(1f), 1f, 0.5f, float3.zero, float3.zero);
        readonly AvbdBody bodyB = new AvbdBody(new float3(1f), 1f, 0.5f, float3.zero, float3.zero);
        readonly AvbdManifold.Contact[] scratch = new AvbdManifold.Contact[BoxCollision.MaxContacts];

        public CpuContacts(int contactBodies, int contactBase)
        {
            ContactBodies = contactBodies;
            ContactBase = contactBase;
            long pairs = ContactLayout.PairCount(contactBodies);
            PairCounts = new uint[pairs];
            excluded = new bool[pairs];
        }

        /// <summary>Length of a record array with room for <paramref name="jointCapacity"/> joints and springs and the contacts of <paramref name="contactBodies"/> bodies.</summary>
        public static int RecordCapacity(int jointCapacity, int contactBodies)
        {
            return jointCapacity + (int)ContactLayout.PairCount(contactBodies) * ContactLayout.MaxContactsPerPair;
        }

        /// <summary>The reference solver creates no manifold between bodies that a joint or a spring joins: mark those pairs.</summary>
        public void ExcludeJoinedPairs(ConstraintRecord[] constraints, int jointCount)
        {
            Array.Clear(excluded, 0, excluded.Length);
            for (int i = 0; i < jointCount; i++)
            {
                if (ConstraintTopology.IsBroken(constraints[i]))
                    continue; // a fractured joint no longer joins the pair
                uint a = constraints[i].ident.x;
                uint b = constraints[i].ident.y;
                if (a == AvbdConst.Invalid || a >= (uint)ContactBodies || b >= (uint)ContactBodies)
                    continue;
                excluded[ContactLayout.PairIndex((int)a, (int)b)] = true;
            }
        }

        /// <summary>A spawned or removed slot does not keep the manifolds of its previous body.</summary>
        public void ForgetSlot(int slot)
        {
            if (slot >= ContactBodies)
                return;
            for (int other = 0; other < ContactBodies; other++)
                if (other != slot)
                    PairCounts[ContactLayout.PairIndex(slot, other)] = 0;
        }

        static bool IsSolved(CpuBodyArrays bodies, int slot)
        {
            uint flags = bodies.Flags[slot];
            return (flags & ((uint)BodyFlags.Free | (uint)BodyFlags.Kinematic)) == 0u && bodies.Mass[slot].momentMass.w > 0f;
        }

        /// <summary>The Narrowphase kernel for every pair of the first <paramref name="bodyCount"/> slots (up to the contact bodies).</summary>
        public void Narrowphase(CpuBodyArrays bodies, ConstraintRecord[] constraints, in AvbdSolverSettings settings, int bodyCount)
        {
            int n = Math.Min(bodyCount, ContactBodies);
            for (int a = 1; a < n; a++)
                for (int b = 0; b < a; b++)
                    NarrowphasePair(bodies, constraints, settings, a, b);
        }

        void NarrowphasePair(CpuBodyArrays bodies, ConstraintRecord[] constraints, in AvbdSolverSettings settings, int a, int b)
        {
            int pair = (int)ContactLayout.PairIndex(a, b);
            uint oldCount = PairCounts[pair];

            bool freeA = (bodies.Flags[a] & (uint)BodyFlags.Free) != 0u;
            bool freeB = (bodies.Flags[b] & (uint)BodyFlags.Free) != 0u;
            if (freeA || freeB || (!IsSolved(bodies, a) && !IsSolved(bodies, b)))
            {
                PairCounts[pair] = 0;
                return;
            }

            BodyStateRecord stateA = bodies.State[a];
            BodyStateRecord stateB = bodies.State[b];
            float4 shapeA = bodies.Shape[a].sizeFriction;
            float4 shapeB = bodies.Shape[b].sizeFriction;
            float3 posA = stateA.startPosition.xyz;
            float3 posB = stateB.startPosition.xyz;

            // A new manifold needs overlapping bounding spheres and no force between the bodies; an existing one skips both tests.
            if (oldCount == 0)
            {
                if (excluded[pair])
                {
                    PairCounts[pair] = 0;
                    return;
                }
                float3 dp = posA - posB;
                float r = math.length(shapeA.xyz * 0.5f) + math.length(shapeB.xyz * 0.5f);
                if (!(math.dot(dp, dp) <= r * r))
                {
                    PairCounts[pair] = 0;
                    return;
                }
            }

            bodyA.PositionLin = posA;
            bodyA.PositionAng = new quaternion(stateA.startRotation);
            bodyA.Size = shapeA.xyz;
            bodyB.PositionLin = posB;
            bodyB.PositionAng = new quaternion(stateB.startRotation);
            bodyB.Size = shapeB.xyz;

            Array.Clear(scratch, 0, scratch.Length);
            float3x3 basis = float3x3.identity;
            int count = BoxCollision.Collide(bodyA, bodyB, scratch, ref basis);
            if (count == 0)
            {
                PairCounts[pair] = 0;
                return;
            }

            int first = ContactBase + pair * ContactLayout.MaxContactsPerPair;
            var old = new ConstraintRecord[ContactLayout.MaxContactsPerPair];
            for (int k = 0; k < oldCount; k++)
                old[k] = constraints[first + k];

            float3 normal = AvbdMat3.Row(basis, 0);
            float friction = math.sqrt(shapeA.w * shapeB.w);
            for (int i = 0; i < count; i++)
            {
                float3 rA = scratch[i].rA;
                float3 rB = scratch[i].rB;
                float3 penalty = float3.zero;
                float3 lambda = float3.zero;
                bool stick = false;

                // Merge with the old contact of the same feature.
                for (int j = 0; j < oldCount; j++)
                {
                    if ((int)old[j].ident.w != scratch[i].featureKey)
                        continue;
                    penalty = old[j].penaltyLin.xyz;
                    lambda = old[j].lambdaLin.xyz;
                    stick = (old[j].ident.z & (uint)ConstraintFlags.Stick) != 0u;
                    if (stick)
                    {
                        rA = old[j].anchorA.xyz;
                        rB = old[j].anchorB.xyz;
                    }
                    break;
                }

                // Error at the start-of-step pose, and the warm start.
                float3 xA = AvbdQuat.Transform(posA, bodyA.PositionAng, rA);
                float3 xB = AvbdQuat.Transform(posB, bodyB.PositionAng, rB);
                float3 c0 = math.mul(basis, xA - xB) + new float3(ContactLayout.CollisionMargin, 0f, 0f);
                lambda = lambda * settings.alpha * settings.gamma;
                penalty = AvbdMat3.Clamp(penalty * settings.gamma, settings.penaltyMin, settings.penaltyMax);

                constraints[first + i] = new ConstraintRecord
                {
                    ident = new uint4((uint)a, (uint)b,
                        ConstraintFlagLayout.Pack(stick ? ConstraintFlags.Stick : ConstraintFlags.None, ConstraintKind.Contact),
                        (uint)scratch[i].featureKey),
                    anchorA = new float4(rA, friction),
                    anchorB = new float4(rB, 0f),
                    c0Lin = new float4(c0, 0f),
                    c0Ang = new float4(normal, 0f),
                    penaltyLin = new float4(penalty, 0f),
                    lambdaLin = new float4(lambda, 0f),
                };
            }
            PairCounts[pair] = (uint)count;
        }

        /// <summary>Builds the lists and the coloring with the contacts of the last narrow phase, and stores the ranges in the body arrays.</summary>
        public ConstraintTopology BuildTopology(CpuBodyArrays bodies, ConstraintRecord[] constraints, int jointCount, int bodyCount)
        {
            var solved = new bool[bodyCount];
            for (int s = 0; s < bodyCount; s++)
                solved[s] = IsSolved(bodies, s);

            var topology = new ConstraintTopology();
            topology.Build(constraints, jointCount, solved, bodyCount, true, PairCounts, Math.Min(bodyCount, ContactBodies), ContactBase);
            Array.Copy(topology.Ranges, bodies.Range, bodyCount);
            return topology;
        }

        /// <summary>The DualContact kernel for every contact of every pair.</summary>
        public void Dual(CpuBodyArrays bodies, ConstraintRecord[] constraints, in AvbdSolverSettings settings, int bodyCount)
        {
            int n = Math.Min(bodyCount, ContactBodies);
            for (int a = 1; a < n; a++)
                for (int b = 0; b < a; b++)
                {
                    int pair = (int)ContactLayout.PairIndex(a, b);
                    for (int k = 0; k < (int)PairCounts[pair]; k++)
                        CpuContactMath.Dual(bodies, ref constraints[ContactBase + pair * ContactLayout.MaxContactsPerPair + k], settings);
                }
        }
    }

    /// <summary>The math of one contact in the primal and dual stages, on a contact record: the arithmetic of Include/AvbdContactSolve.hlsl.</summary>
    internal static class CpuContactMath
    {
        public struct Eval
        {
            public float3x3 jALin, jBLin, jAAng, jBAng;
            public float3 C;
            public float3 F;
            public float bounds;
            public float frictionScale;
        }

        // Rows: the normal, then two tangents (upstream's orthonormal()).
        public static float3x3 Basis(float3 normal)
        {
            float3 t1 = math.abs(normal.x) > math.abs(normal.z)
                ? new float3(-normal.y, normal.x, 0f)
                : new float3(0f, -normal.z, normal.y);
            t1 = math.normalize(t1);
            float3 t2 = math.cross(normal, t1);
            return AvbdMat3.FromRows(normal, t1, t2);
        }

        public static Eval Evaluate(
            in ConstraintRecord c, quaternion rotA, quaternion rotB,
            float3 dqALin, float3 dqAAng, float3 dqBLin, float3 dqBAng, float alpha)
        {
            var e = new Eval();
            float3x3 basis = Basis(c.c0Ang.xyz);
            float3 rAWorld = AvbdQuat.Rotate(rotA, c.anchorA.xyz);
            float3 rBWorld = AvbdQuat.Rotate(rotB, c.anchorB.xyz);

            e.jALin = basis;
            e.jBLin = -basis;
            e.jAAng = AvbdMat3.FromRows(
                math.cross(rAWorld, AvbdMat3.Row(e.jALin, 0)), math.cross(rAWorld, AvbdMat3.Row(e.jALin, 1)), math.cross(rAWorld, AvbdMat3.Row(e.jALin, 2)));
            e.jBAng = AvbdMat3.FromRows(
                math.cross(rBWorld, AvbdMat3.Row(e.jBLin, 0)), math.cross(rBWorld, AvbdMat3.Row(e.jBLin, 1)), math.cross(rBWorld, AvbdMat3.Row(e.jBLin, 2)));

            e.C = c.c0Lin.xyz * (1f - alpha)
                + math.mul(e.jALin, dqALin) + math.mul(e.jBLin, dqBLin)
                + math.mul(e.jAAng, dqAAng) + math.mul(e.jBAng, dqBAng);

            float3 F = c.penaltyLin.xyz * e.C + c.lambdaLin.xyz;
            F.x = math.min(F.x, 0f);
            e.bounds = math.abs(F.x) * c.anchorA.w;
            e.frictionScale = math.sqrt(F.y * F.y + F.z * F.z);
            if (e.frictionScale > e.bounds && e.frictionScale > 0f)
            {
                F.y *= e.bounds / e.frictionScale;
                F.z *= e.bounds / e.frictionScale;
            }
            e.F = F;
            return e;
        }

        // Displacements of both bodies from the start of the step, at the poses given for the two bodies.
        public static Eval EvaluateAt(
            CpuBodyArrays bodies, in ConstraintRecord c, float3 posA, quaternion rotA, float3 posB, quaternion rotB, float alpha)
        {
            BodyStateRecord a = bodies.State[c.ident.x];
            BodyStateRecord b = bodies.State[c.ident.y];
            return Evaluate(
                c, rotA, rotB,
                posA - a.startPosition.xyz, AvbdQuat.DifferenceRaw(rotA, new quaternion(a.startRotation)),
                posB - b.startPosition.xyz, AvbdQuat.DifferenceRaw(rotB, new quaternion(b.startRotation)), alpha);
        }

        /// <summary>Adds the terms of the contact for the body being solved (A if <paramref name="isA"/>) to its system.</summary>
        public static void Stamp(in Eval e, float3 penalty, bool isA, ref AvbdPrimalSystem sys)
        {
            float3x3 K = AvbdMat3.Diagonal(penalty);
            float3x3 jLin = isA ? e.jALin : e.jBLin;
            float3x3 jAng = isA ? e.jAAng : e.jBAng;

            float3x3 jLinT = math.transpose(jLin);
            float3x3 jAngT = math.transpose(jAng);
            float3x3 jAngTk = math.mul(jAngT, K);

            sys.lhsLin += math.mul(math.mul(jLinT, K), jLin);
            sys.lhsAng += math.mul(jAngTk, jAng);
            sys.lhsCross += math.mul(jAngTk, jLin);
            sys.rhsLin += math.mul(jLinT, e.F);
            sys.rhsAng += math.mul(jAngT, e.F);
        }

        /// <summary>The dual update of one contact record at the current body poses.</summary>
        public static void Dual(CpuBodyArrays bodies, ref ConstraintRecord c, in AvbdSolverSettings settings)
        {
            BodyStateRecord a = bodies.State[c.ident.x];
            BodyStateRecord b = bodies.State[c.ident.y];
            Eval e = EvaluateAt(bodies, c, a.position.xyz, new quaternion(a.rotation), b.position.xyz, new quaternion(b.rotation), settings.alpha);

            float3 penalty = c.penaltyLin.xyz;
            bool stick = (c.ident.z & (uint)ConstraintFlags.Stick) != 0u;

            if (e.F.x < 0f)
                penalty.x = math.min(penalty.x + settings.betaLin * math.abs(e.C.x), settings.penaltyMax);
            if (e.frictionScale <= e.bounds)
            {
                penalty.y = math.min(penalty.y + settings.betaLin * math.abs(e.C.y), settings.penaltyMax);
                penalty.z = math.min(penalty.z + settings.betaLin * math.abs(e.C.z), settings.penaltyMax);
                stick = math.sqrt(e.C.y * e.C.y + e.C.z * e.C.z) < ContactLayout.StickThreshold;
            }

            c.ident.z = stick ? c.ident.z | (uint)ConstraintFlags.Stick : c.ident.z & ~(uint)ConstraintFlags.Stick;
            c.penaltyLin = new float4(penalty, c.penaltyLin.w);
            c.lambdaLin = new float4(e.F, c.lambdaLin.w);
        }
    }
}
