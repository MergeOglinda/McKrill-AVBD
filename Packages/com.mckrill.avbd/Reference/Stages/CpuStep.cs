namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle of a whole solver step, the same sequence of stages as <c>AvbdSolver.Step</c> without collisions: predict (P),
    /// constraint warm start (D), per iteration a primal sweep (V) over the colors and the dual update (D), then the velocity update (U).
    /// The warm start follows the predict stage, because that stage writes the start-of-step pose the warm start reads.
    /// </summary>
    internal static class CpuStep
    {
        /// <summary>Runs one step. Returns the number of failed primal solves.</summary>
        public static int Run(CpuBodyArrays bodies, ConstraintRecord[] constraints, int constraintCount, ConstraintTopology topology, int bodyCount, in AvbdSolverSettings settings)
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
