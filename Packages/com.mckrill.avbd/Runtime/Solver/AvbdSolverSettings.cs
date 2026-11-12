using System;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>
    /// Global solver constants (design doc section 1: one small set of constants, not per-record fields).
    /// The defaults match the upstream avbd-demo3d solver, except for the gravity direction (see <see cref="Default"/>).
    /// </summary>
    public struct AvbdSolverSettings
    {
        /// <summary>Time step in seconds.</summary>
        public float dt;

        /// <summary>Gravity acceleration vector. Applied to bodies with mass above zero.</summary>
        public float3 gravity;

        /// <summary>Solver iterations per step. Each iteration is one primal and one dual pass.</summary>
        public int iterations;

        /// <summary>Stabilization parameter. 1 is no stabilization, 0 is full.</summary>
        public float alpha;

        /// <summary>Penalty ramping rate of linear constraints.</summary>
        public float betaLin;

        /// <summary>Penalty ramping rate of angular constraints.</summary>
        public float betaAng;

        /// <summary>Warm-starting decay of multipliers and penalties.</summary>
        public float gamma;

        /// <summary>Lowest penalty a constraint row can have.</summary>
        public float penaltyMin;

        /// <summary>Highest penalty a constraint row can have.</summary>
        public float penaltyMax;

        /// <summary>
        /// 60 Hz, 10 iterations, upstream's stabilization constants, and gravity (0, -10, 0) for Unity's y-up world.
        /// Upstream's demo scenes are z-up and need gravity (0, 0, -10).
        /// </summary>
        public static AvbdSolverSettings Default => new AvbdSolverSettings
        {
            dt = 1f / 60f,
            gravity = new float3(0f, -10f, 0f),
            iterations = 10,
            alpha = 0.99f,
            betaLin = 1e4f,
            betaAng = 100f,
            gamma = 0.999f,
            penaltyMin = 1f,
            penaltyMax = 1e10f,
        };

        /// <summary>Throws ArgumentException if a value is out of range.</summary>
        public void Validate()
        {
            if (!(dt > 0f) || !math.isfinite(dt))
                throw new ArgumentException("dt must be finite and positive.", nameof(dt));
            if (!math.all(math.isfinite(gravity)))
                throw new ArgumentException("gravity must be finite.", nameof(gravity));
            if (iterations < 0)
                throw new ArgumentException("iterations must not be negative.", nameof(iterations));
            if (!(alpha >= 0f && alpha <= 1f))
                throw new ArgumentException("alpha must be between 0 and 1.", nameof(alpha));
            if (!(betaLin >= 0f) || !math.isfinite(betaLin))
                throw new ArgumentException("betaLin must be finite and not negative.", nameof(betaLin));
            if (!(betaAng >= 0f) || !math.isfinite(betaAng))
                throw new ArgumentException("betaAng must be finite and not negative.", nameof(betaAng));
            if (!(gamma >= 0f && gamma <= 1f))
                throw new ArgumentException("gamma must be between 0 and 1.", nameof(gamma));
            if (!(penaltyMin > 0f) || !math.isfinite(penaltyMin))
                throw new ArgumentException("penaltyMin must be finite and positive.", nameof(penaltyMin));
            if (!(penaltyMax >= penaltyMin) || !math.isfinite(penaltyMax))
                throw new ArgumentException("penaltyMax must be finite and at least penaltyMin.", nameof(penaltyMax));
        }
    }
}
