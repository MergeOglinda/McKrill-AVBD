namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// A force with no physical effect. Its presence between two bodies stops the broadphase from
    /// creating a manifold for them (see <see cref="AvbdBody.ConstrainedTo"/>).
    /// </summary>
    public sealed class AvbdIgnoreCollision : AvbdForce
    {
        public AvbdIgnoreCollision(ReferenceSolver solver, AvbdBody bodyA, AvbdBody bodyB)
            : base(solver, bodyA, bodyB)
        {
        }

        public override bool AffectsDynamics => false;

        public override bool Initialize() => true;

        public override void UpdatePrimal(AvbdBody body, float alpha, ref AvbdPrimalSystem system)
        {
        }

        public override void UpdateDual(float alpha)
        {
        }
    }
}
