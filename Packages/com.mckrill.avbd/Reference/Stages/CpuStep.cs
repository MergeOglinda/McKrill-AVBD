namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle of a whole solver step, in the order <see cref="AvbdSolver.Step"/> runs the stages: predict (P), the constraint
    /// warm start (D), then per iteration a primal sweep (V) and the dual update (D), then the velocity update (U).
    /// The constraint lists and the coloring are built once by the caller; they do not change while no body or constraint is added.
    /// </summary>
    internal static class CpuStep
    {
        /// <summary>
        /// Runs one step with contacts: after predict, the narrow phase over all pairs and the topology rebuild with the contacts it found
        /// (the lists and the coloring change every step), then the same iterations, with the contact dual update after each joint dual update.
        /// <paramref name="constraints"/> holds the joints and springs first, then the contact region. Returns the number of failed primal solves.
        /// </summary>
        public static int Run(CpuBodyArrays bodies, ConstraintRecord[] constraints, int jointCount, CpuContacts contacts,
            int bodyCount, in AvbdSolverSettings settings, PlasticRecord[] plastic = null)
        {
            CpuPredict.Run(bodies, settings, bodyCount);
            contacts.Narrowphase(bodies, constraints, settings, bodyCount);
            ConstraintTopology topology = contacts.BuildTopology(bodies, constraints, jointCount, bodyCount);
            CpuConstraintInit.Run(bodies, constraints, jointCount, settings, plastic);

            int failures = 0;
            for (int it = 0; it < settings.iterations; it++)
            {
                failures += CpuPrimal.Sweep(bodies, constraints, topology, settings, plastic);
                CpuDual.Run(bodies, constraints, jointCount, settings, plastic);
                contacts.Dual(bodies, constraints, settings, bodyCount);
            }

            CpuVelocity.Run(bodies, settings, bodyCount);
            return failures;
        }

        /// <summary>Runs one step. Returns the number of failed primal solves.</summary>
        public static int Run(CpuBodyArrays bodies, ConstraintRecord[] constraints, int constraintCount, ConstraintTopology topology,
            int bodyCount, in AvbdSolverSettings settings, PlasticRecord[] plastic = null)
        {
            CpuPredict.Run(bodies, settings, bodyCount);
            CpuConstraintInit.Run(bodies, constraints, constraintCount, settings, plastic);

            int failures = 0;
            for (int it = 0; it < settings.iterations; it++)
            {
                failures += CpuPrimal.Sweep(bodies, constraints, topology, settings, plastic);
                CpuDual.Run(bodies, constraints, constraintCount, settings, plastic);
            }

            CpuVelocity.Run(bodies, settings, bodyCount);
            return failures;
        }
    }
}
