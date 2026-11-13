using System;
using System.Collections.Generic;
using Unity.Mathematics;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>A reference-solver scene converted to what the GPU stages and their oracles consume.</summary>
    internal sealed class SceneArrays
    {
        public CpuBodyArrays bodies;
        public List<BodyDescriptor> descriptors = new List<BodyDescriptor>();
        public ConstraintRecord[] constraints;
        public int constraintCount;
        public int bodyCount;

        /// <summary>Bodies the primal stage solves: not free, not kinematic, mass above zero.</summary>
        public bool[] Solved()
        {
            var solved = new bool[bodyCount];
            for (int s = 0; s < bodyCount; s++)
            {
                uint flags = bodies.Meta[s].ident.x;
                solved[s] = (flags & ((uint)BodyFlags.Free | (uint)BodyFlags.Kinematic)) == 0u && bodies.Mass[s].momentMass.w > 0f;
            }
            return solved;
        }

        /// <summary>Builds the lists and the coloring of the converted scene and stores the ranges in the oracle's body arrays.</summary>
        public ConstraintTopology BuildTopology()
        {
            var topology = new ConstraintTopology();
            topology.Build(constraints, constraintCount, Solved(), bodyCount);
            Array.Copy(topology.Ranges, bodies.Range, bodyCount);
            return topology;
        }
    }

    internal static class ConstraintTestScenes
    {
        /// <summary>
        /// Converts the bodies, joints and springs of a reference solver, in creation order (body index = slot, constraint index = order
        /// among joints and springs), with the joints' current penalties and multipliers. Call it before the solver steps.
        /// </summary>
        public static SceneArrays FromReference(ReferenceSolver solver, int capacity)
        {
            var scene = new SceneArrays
            {
                bodies = new CpuBodyArrays(capacity),
                constraints = new ConstraintRecord[solver.Forces.Count],
                bodyCount = solver.Bodies.Count,
            };

            var queue = new BodyEditQueue(capacity);
            var commands = new List<EditCommand>();
            for (int i = 0; i < solver.Bodies.Count; i++)
            {
                AvbdBody b = solver.Bodies[i];
                float volume = b.Size.x * b.Size.y * b.Size.z;
                BodyDescriptor d = BodyDescriptor.Box(b.Size, b.Mass > 0f ? b.Mass / volume : 0f, b.PositionLin);
                d.rotation = b.PositionAng;
                d.linearVelocity = b.VelocityLin;
                d.angularVelocity = b.VelocityAng;
                scene.descriptors.Add(d);

                BodyHandle h = queue.Spawn(d);
                if (h.Slot != i)
                    throw new InvalidOperationException("Slots are not allocated in order.");
            }
            queue.Drain(commands);
            scene.bodies.Apply(commands);

            foreach (AvbdForce force in solver.Forces)
            {
                uint slotA = force.BodyA != null ? (uint)force.BodyA.Index : AvbdConst.Invalid;
                uint slotB = (uint)force.BodyB.Index;
                if (force is AvbdJoint j)
                {
                    ConstraintRecord r = ConstraintRecordFactory.Joint(slotA, slotB, j.rA, j.rB, j.stiffnessLin, j.stiffnessAng, j.fracture, j.torqueArm);
                    r.penaltyLin = new float4(j.penaltyLin, 0f);
                    r.penaltyAng = new float4(j.penaltyAng, 0f);
                    r.lambdaLin = new float4(j.lambdaLin, 0f);
                    r.lambdaAng = new float4(j.lambdaAng, 0f);
                    scene.constraints[scene.constraintCount++] = r;
                }
                else if (force is AvbdSpring s)
                {
                    scene.constraints[scene.constraintCount++] = ConstraintRecordFactory.Spring(slotA, slotB, s.rA, s.rB, s.stiffness, s.rest);
                }
            }
            return scene;
        }

        /// <summary>Gives every joint of the solver random per-axis penalties and multipliers, so that the primal stage sees large stiffness.</summary>
        public static void RandomizeJointState(ReferenceSolver solver, System.Random rng)
        {
            foreach (AvbdForce force in solver.Forces)
            {
                if (!(force is AvbdJoint j))
                    continue;
                j.penaltyLin = LogUniform3(rng, 10f, 1e4f);
                j.penaltyAng = LogUniform3(rng, 10f, 1e4f);
                j.lambdaLin = Uniform3(rng, 100f);
                j.lambdaAng = Uniform3(rng, 100f);
            }
        }

        /// <summary>
        /// Boxes on a 3 x 3 x 2 lattice (spacing 3, so that no two bodies are close enough for the reference solver to create a
        /// manifold), some of them static, joined by random joints and springs between lattice neighbors and pinned to the world.
        /// Random poses and velocities. Z-up, like the upstream scenes.
        /// </summary>
        public static void RandomLattice(ReferenceSolver s, int seed)
        {
            var rng = new System.Random(seed);
            s.Clear();

            var bodies = new List<AvbdBody>();
            var cell = new List<int3>();
            for (int z = 0; z < 2; z++)
                for (int y = 0; y < 3; y++)
                    for (int x = 0; x < 3; x++)
                    {
                        int index = bodies.Count;
                        float3 size = new float3(F(rng, 0.5f, 1.2f), F(rng, 0.5f, 1.2f), F(rng, 0.5f, 1.2f));
                        float density = index % 5 == 0 ? 0f : F(rng, 0.5f, 2f);
                        float3 position = new float3(x * 3f, y * 3f, z * 3f) + Uniform3(rng, 0.2f);
                        AvbdBody b = s.AddBody(size, density, 0.5f, position, Uniform3(rng, 2f));
                        b.PositionAng = quaternion.EulerXYZ(F(rng, 0f, 6f), F(rng, 0f, 6f), F(rng, 0f, 6f));
                        b.VelocityAng = Uniform3(rng, 2f);
                        bodies.Add(b);
                        cell.Add(new int3(x, y, z));
                    }

            for (int i = 0; i < bodies.Count; i++)
                for (int j = i + 1; j < bodies.Count; j++)
                {
                    int3 d = math.abs(cell[i] - cell[j]);
                    if (d.x + d.y + d.z != 1 || rng.NextDouble() > 0.6)
                        continue;

                    float3 rA = Uniform3(rng, 0.5f);
                    float3 rB = Uniform3(rng, 0.5f);
                    switch (rng.Next(4))
                    {
                        case 0: s.AddJoint(bodies[i], bodies[j], rA, rB); break;
                        case 1: s.AddJoint(bodies[i], bodies[j], rA, rB, float.PositiveInfinity, float.PositiveInfinity); break;
                        case 2: s.AddJoint(bodies[i], bodies[j], rA, rB, 5000f, 2000f); break;
                        default: s.AddSpring(bodies[i], bodies[j], rA, rB, F(rng, 50f, 500f), F(rng, 1f, 3f)); break;
                    }
                }

            // Joints pinned to a world point near the body.
            for (int i = 0; i < bodies.Count; i++)
                if (rng.NextDouble() < 0.3)
                {
                    float3 rB = Uniform3(rng, 0.5f);
                    float3 world = bodies[i].PositionLin + Uniform3(rng, 0.6f);
                    if (rng.Next(2) == 0)
                        s.AddJoint(null, bodies[i], world, rB);
                    else
                        s.AddJoint(null, bodies[i], world, rB, float.PositiveInfinity, float.PositiveInfinity);
                }

            RandomizeJointState(s, rng);
        }

        static float F(System.Random rng, float min, float max) => min + (float)rng.NextDouble() * (max - min);

        static float3 Uniform3(System.Random rng, float range) => new float3(F(rng, -range, range), F(rng, -range, range), F(rng, -range, range));

        static float3 LogUniform3(System.Random rng, float min, float max)
        {
            float lo = math.log(min);
            float hi = math.log(max);
            return new float3(math.exp(F(rng, lo, hi)), math.exp(F(rng, lo, hi)), math.exp(F(rng, lo, hi)));
        }
    }
}
