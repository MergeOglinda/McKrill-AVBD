namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle of a whole solver step, in the order <see cref="AvbdSolver.Step"/> runs the stages: predict (P), the constraint
    /// warm start (D), then per iteration a primal sweep (V) and the dual update (D), then the velocity update (U).
    /// The constraint lists and the coloring are built once by the caller; they do not change while no body or constraint is added.
    /// </summary>
    internal static class CpuStep
    {
        /// <summary>Runs one step. Returns the number of failed primal solves.</summary>
        public static int Run(CpuBodyArrays bodies, ConstraintRecord[] constraints, int constraintCount, ConstraintTopology topology,
            int bodyCount, in AvbdSolverSettings settings)
        {
            CpuPredict.Run(bodies, settings, bodyCount);
            CpuConstraintInit.Run(bodies, constraints, constraintCount, settings);

            int failures = 0;
            for (int it = 0; it < settings.iterations; it++)
            {
                failures += CpuPrimal.Sweep(bodies, constraints, topology, settings);
                CpuDual.Run(bodies, constraints, constraintCount, settings);
            }

            CpuVelocity.Run(bodies, settings, bodyCount);
            return failures;
        }
    }
}
