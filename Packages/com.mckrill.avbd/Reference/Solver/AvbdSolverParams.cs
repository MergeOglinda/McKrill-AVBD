using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>How the primal (per-body) update of one solver iteration is ordered.</summary>
    public enum GaussSeidelMode
    {
        /// <summary>
        /// Faithful port of upstream: bodies are visited one after another in list order (newest body
        /// first), and each body sees the already-updated positions of the bodies visited before it.
        /// </summary>
        Sequential = 0,

        /// <summary>
        /// Bodies are greedily graph-colored so that no two bodies joined by a constraint share a color.
        /// One iteration sweeps the colors in order; bodies inside one color are independent, so on the
        /// GPU a color is one dispatch. This is the ordering the GPU solver will run.
        /// </summary>
        Colored = 1,
    }

    /// <summary>
    /// Solver parameters of the CPU reference. Defaults and constants are those of upstream
    /// avbd-demo3d (solver.cpp defaultParams(), solver.h defines). The one structural change is that
    /// gravity is a vector instead of a scalar along +z.
    /// </summary>
    public sealed class AvbdSolverParams
    {
        public const float PenaltyMin = 1f;                // PENALTY_MIN
        public const float PenaltyMax = 10000000000f;      // PENALTY_MAX
        public const float CollisionMargin = 0.01f;        // COLLISION_MARGIN
        public const float StickThreshold = 0.00001f;      // STICK_THRESH

        /// <summary>Timestep.</summary>
        public float dt = 1f / 60f;

        /// <summary>
        /// Gravity acceleration vector. Upstream uses scalar -10 along +z (a z-up frame), which is
        /// (0, 0, -10) here. Use (0, -10, 0) for Unity's y-up frame.
        /// </summary>
        public float3 gravity = new float3(0f, 0f, -10f);

        /// <summary>Solver iterations per step.</summary>
        public int iterations = 10;

        /// <summary>Stabilization: higher values correct constraint error more slowly and smoothly.</summary>
        public float alpha = 0.99f;

        /// <summary>Penalty ramping for linear constraints.</summary>
        public float betaLin = 10000f;

        /// <summary>Penalty ramping for angular constraints.</summary>
        public float betaAng = 100f;

        /// <summary>Warm-start decay of penalties and multipliers. Must be below 1.</summary>
        public float gamma = 0.999f;

        /// <summary>Ordering of the primal sweep.</summary>
        public GaussSeidelMode mode = GaussSeidelMode.Sequential;

        /// <summary>
        /// When true, rotation errors use <see cref="AvbdQuat.Difference"/> (sign-canonicalized) instead
        /// of upstream's <see cref="AvbdQuat.DifferenceRaw"/>. Off by default so the port matches upstream.
        /// Only matters for relative rotations beyond 180 degrees.
        /// </summary>
        public bool canonicalizeQuatDifference = false;
    }

    /// <summary>Counters of the last <see cref="ReferenceSolver.Step"/>.</summary>
    public struct AvbdStepStats
    {
        /// <summary>Manifolds that survived initialization (at least one contact).</summary>
        public int manifolds;

        /// <summary>Total contact points over those manifolds.</summary>
        public int contacts;

        /// <summary>Number of graph colors used (0 in Sequential mode).</summary>
        public int colors;

        /// <summary>
        /// Primal solves skipped because the 6x6 system was not numerically positive definite.
        /// Upstream has no such check and would write NaNs; the reference skips the update instead.
        /// </summary>
        public int solveFailures;
    }
}
