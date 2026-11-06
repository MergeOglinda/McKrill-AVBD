using System.Collections.Generic;
using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU reference of the AVBD rigid-body solver: a port of upstream avbd-demo3d's Solver. It is the
    /// oracle the GPU kernels are tested against, so it favors fidelity and readability over speed.
    /// It is managed, allocating code, not Burst code.
    ///
    /// One <see cref="Step"/> does: broadphase (O(n^2) pairs), force initialization and warm start,
    /// body initialization and the adaptive warm start, then <c>iterations</c> rounds of primal sweep
    /// (every dynamic body solves its 6x6 system once) followed by the dual update of every force, and
    /// finally the BDF1 velocity update.
    ///
    /// <see cref="GaussSeidelMode.Sequential"/> reproduces upstream exactly (bodies visited in list
    /// order, newest first, each seeing its predecessors' updates). <see cref="GaussSeidelMode.Colored"/>
    /// sweeps graph colors instead, which is what a GPU can execute in parallel.
    /// </summary>
    public sealed class ReferenceSolver
    {
        readonly List<AvbdBody> bodies = new List<AvbdBody>();
        readonly List<AvbdForce> forces = new List<AvbdForce>();
        readonly List<List<AvbdBody>> colorGroups = new List<List<AvbdBody>>();

        public AvbdSolverParams Params { get; } = new AvbdSolverParams();

        /// <summary>Counters of the last step.</summary>
        public AvbdStepStats Stats { get; private set; }

        /// <summary>All bodies in creation order.</summary>
        public IReadOnlyList<AvbdBody> Bodies => bodies;

        /// <summary>All forces in creation order. Manifolds are created and removed by the solver.</summary>
        public IReadOnlyList<AvbdForce> Forces => forces;

        /// <summary>Body groups of the last Colored step (empty in Sequential mode), color 0 first.</summary>
        public IReadOnlyList<List<AvbdBody>> ColorGroups => colorGroups;

        // ---- scene building --------------------------------------------------------------------------

        /// <summary>Adds a box. Density 0 makes it static.</summary>
        public AvbdBody AddBody(float3 size, float density, float friction, float3 position, float3 velocity = default)
        {
            var body = new AvbdBody(size, density, friction, position, velocity) { Index = bodies.Count };
            bodies.Add(body);
            return body;
        }

        /// <summary>Adds a ball-socket joint. Pass a null bodyA to pin bodyB to the world point rA.</summary>
        public AvbdJoint AddJoint(
            AvbdBody bodyA, AvbdBody bodyB, float3 rA, float3 rB,
            float stiffnessLin = float.PositiveInfinity, float stiffnessAng = 0f, float fracture = float.PositiveInfinity)
        {
            var joint = new AvbdJoint(this, bodyA, bodyB, rA, rB, stiffnessLin, stiffnessAng, fracture);
            AddForce(joint);
            return joint;
        }

        public AvbdSpring AddSpring(AvbdBody bodyA, AvbdBody bodyB, float3 rA, float3 rB, float stiffness, float rest = -1f)
        {
            var spring = new AvbdSpring(this, bodyA, bodyB, rA, rB, stiffness, rest);
            AddForce(spring);
            return spring;
        }

        public AvbdIgnoreCollision AddIgnoreCollision(AvbdBody bodyA, AvbdBody bodyB)
        {
            var ignore = new AvbdIgnoreCollision(this, bodyA, bodyB);
            AddForce(ignore);
            return ignore;
        }

        /// <summary>Registers a force with the solver and with both of its bodies.</summary>
        public void AddForce(AvbdForce force)
        {
            forces.Add(force);
            force.BodyA?.Forces.Add(force);
            force.BodyB?.Forces.Add(force);
        }

        public void RemoveForce(AvbdForce force)
        {
            if (forces.Remove(force))
                DetachFromBodies(force);
        }

        public void Clear()
        {
            forces.Clear();
            bodies.Clear();
            colorGroups.Clear();
            Stats = default;
        }

        static void DetachFromBodies(AvbdForce force)
        {
            force.BodyA?.Forces.Remove(force);
            force.BodyB?.Forces.Remove(force);
        }

        /// <summary>
        /// Rotation error a - b. Upstream's formula unless
        /// <see cref="AvbdSolverParams.canonicalizeQuatDifference"/> is set.
        /// </summary>
        public float3 QuatDifference(quaternion a, quaternion b)
        {
            return Params.canonicalizeQuatDifference ? AvbdQuat.Difference(a, b) : AvbdQuat.DifferenceRaw(a, b);
        }

        // ---- stepping ---------------------------------------------------------------------------------

        public void Step()
        {
            AvbdSolverParams p = Params;
            float dt = p.dt;
            float3 gravity = p.gravity;
            var stats = new AvbdStepStats();

            // Broadphase: naive O(n^2) over all pairs, newest body first like upstream's list walk.
            // A pair gets a manifold if the bounding spheres overlap and no force already joins it
            // (which includes an existing manifold, so manifolds persist across steps).
            for (int i = bodies.Count - 1; i >= 0; i--)
            {
                AvbdBody bodyA = bodies[i];
                for (int j = i - 1; j >= 0; j--)
                {
                    AvbdBody bodyB = bodies[j];
                    float3 dp = bodyA.PositionLin - bodyB.PositionLin;
                    float r = bodyA.Radius + bodyB.Radius;
                    if (math.dot(dp, dp) <= r * r && !bodyA.ConstrainedTo(bodyB))
                        AddForce(new AvbdManifold(this, bodyA, bodyB));
                }
            }

            // Initialize and warm-start forces; forces that report inactive are removed.
            for (int i = forces.Count - 1; i >= 0; i--)
            {
                AvbdForce force = forces[i];
                if (!force.Initialize())
                {
                    DetachFromBodies(force);
                    forces.RemoveAt(i);
                }
                else if (force is AvbdManifold manifold)
                {
                    stats.manifolds++;
                    stats.contacts += manifold.NumContacts;
                }
            }

            // Initialize and warm-start bodies (the primal variables).
            for (int i = bodies.Count - 1; i >= 0; i--)
            {
                AvbdBody body = bodies[i];

                // Inertial position (Eq. 2).
                body.InertialLin = body.PositionLin + body.VelocityLin * dt;
                if (body.IsDynamic)
                    body.InertialLin += gravity * (dt * dt);
                body.InertialAng = AvbdQuat.IntegrateAngular(body.PositionAng, body.VelocityAng * dt);

                // Adaptive warm start (original VBD paper): the fraction of gravity the body actually
                // accelerated by last step. Upstream's accel.z * sign(g) / |g| for gravity along -z.
                float3 accel = (body.VelocityLin - body.PrevVelocityLin) / dt;
                float gravityLength = math.length(gravity);
                float accelWeight = gravityLength > 0f
                    ? math.clamp(math.dot(accel, gravity) / (gravityLength * gravityLength), 0f, 1f)
                    : 0f;
                if (!math.isfinite(accelWeight))
                    accelWeight = 0f;

                // Save the initial pose (x-) and compute the warm-started pose.
                body.InitialLin = body.PositionLin;
                body.InitialAng = body.PositionAng;
                if (body.IsDynamic)
                {
                    body.PositionLin = body.PositionLin + body.VelocityLin * dt + gravity * (accelWeight * dt * dt);
                    body.PositionAng = AvbdQuat.IntegrateAngular(body.PositionAng, body.VelocityAng * dt);
                }
            }

            // Order of the primal sweep.
            colorGroups.Clear();
            if (p.mode == GaussSeidelMode.Colored)
                stats.colors = GraphColoring.Color(bodies, colorGroups);

            // Main solver loop.
            for (int it = 0; it < p.iterations; it++)
            {
                // Primal update.
                if (p.mode == GaussSeidelMode.Colored)
                {
                    for (int c = 0; c < colorGroups.Count; c++)
                    {
                        List<AvbdBody> group = colorGroups[c];
                        for (int k = 0; k < group.Count; k++)
                            PrimalUpdate(group[k], ref stats);
                    }
                }
                else
                {
                    for (int i = bodies.Count - 1; i >= 0; i--)
                        PrimalUpdate(bodies[i], ref stats);
                }

                // Dual update.
                for (int i = forces.Count - 1; i >= 0; i--)
                    forces[i].UpdateDual(p.alpha);
            }

            // Velocities (BDF1) after the final iteration.
            for (int i = bodies.Count - 1; i >= 0; i--)
            {
                AvbdBody body = bodies[i];
                body.PrevVelocityLin = body.VelocityLin;
                if (body.IsDynamic)
                {
                    body.VelocityLin = (body.PositionLin - body.InitialLin) / dt;
                    body.VelocityAng = QuatDifference(body.PositionAng, body.InitialAng) / dt;
                }
            }

            Stats = stats;
        }

        // One body's primal update: assemble the 6x6 system from the inertia term and every force
        // acting on the body, solve it with LDL^T, and apply the Newton step (Eqs. 4-6).
        void PrimalUpdate(AvbdBody body, ref AvbdStepStats stats)
        {
            if (!body.IsDynamic)
                return;

            AvbdSolverParams p = Params;
            float dtSq = p.dt * p.dt;

            var system = new AvbdPrimalSystem
            {
                lhsLin = AvbdMat3.Diagonal(body.Mass / dtSq, body.Mass / dtSq, body.Mass / dtSq),
                lhsAng = AvbdMat3.Diagonal(body.Moment.x / dtSq, body.Moment.y / dtSq, body.Moment.z / dtSq),
                lhsCross = float3x3.zero,
                rhsLin = (body.Mass / dtSq) * (body.PositionLin - body.InertialLin),
                rhsAng = new float3(body.Moment.x, body.Moment.y, body.Moment.z) / dtSq
                         * QuatDifference(body.PositionAng, body.InertialAng),
            };

            // Every force on the body stamps its terms into the system (newest force first, like upstream).
            for (int i = body.Forces.Count - 1; i >= 0; i--)
                body.Forces[i].UpdatePrimal(body, p.alpha, ref system);

            // Solve the symmetric positive definite system (only the lower triangle is read) and apply
            // the update (Eq. 4).
            var a = new Mat6
            {
                ll = system.lhsLin,
                aa = system.lhsAng,
                al = system.lhsCross,
                la = math.transpose(system.lhsCross),
            };

            if (a.TrySolveLDLT(new Vec6(-system.rhsLin, -system.rhsAng), out Vec6 dx))
            {
                body.PositionLin = body.PositionLin + dx.lin;
                body.PositionAng = AvbdQuat.IntegrateAngular(body.PositionAng, dx.ang);
            }
            else
            {
                stats.solveFailures++;
            }
        }
    }
}
