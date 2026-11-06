using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// The 6x6 linear system of one body's primal update, kept as the blocks upstream uses:
    /// <c>lhsLin</c> (linear-linear), <c>lhsAng</c> (angular-angular), <c>lhsCross</c> (angular rows,
    /// linear columns; only the lower triangle of the full matrix is ever read) and the two
    /// right-hand-side halves. All float3x3 values are mathematical matrices (see <see cref="AvbdMat3"/>).
    /// </summary>
    public struct AvbdPrimalSystem
    {
        public float3x3 lhsLin;
        public float3x3 lhsAng;
        public float3x3 lhsCross;
        public float3 rhsLin;
        public float3 rhsAng;
    }

    /// <summary>
    /// Common interface of every constraint or force acting on one or two bodies, a port of upstream's
    /// Force. BodyA may be null for a joint anchored to the world; BodyB is always set.
    /// </summary>
    public abstract class AvbdForce
    {
        public ReferenceSolver Solver { get; }
        public AvbdBody BodyA { get; }
        public AvbdBody BodyB { get; }

        protected AvbdForce(ReferenceSolver solver, AvbdBody bodyA, AvbdBody bodyB)
        {
            Solver = solver;
            BodyA = bodyA;
            BodyB = bodyB;
        }

        /// <summary>
        /// False for forces with no physical effect (IgnoreCollision). Such forces never stamp into a
        /// body's system, so they create no edge in the coloring graph.
        /// </summary>
        public virtual bool AffectsDynamics => true;

        /// <summary>Caches anything constant over the step and warm-starts. Returns false to remove the force.</summary>
        public abstract bool Initialize();

        /// <summary>Stamps this force's Hessian and gradient terms for <paramref name="body"/> into the system.</summary>
        public abstract void UpdatePrimal(AvbdBody body, float alpha, ref AvbdPrimalSystem system);

        /// <summary>Updates the multiplier and penalty after a primal sweep.</summary>
        public abstract void UpdateDual(float alpha);
    }
}
