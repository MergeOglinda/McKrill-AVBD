using UnityEngine;

namespace McKrill.Avbd
{
    /// <summary>
    /// Runs the solver stages on the GPU for the bodies of one <see cref="AvbdBodyBuffers"/>.
    /// The solver does not own the buffers; dispose them separately.
    /// Stages are added one at a time; <see cref="Step"/> runs the ones that exist.
    /// Joints and springs come from an optional <see cref="AvbdConstraintBuffers"/>.
    /// </summary>
    public sealed class AvbdSolver
    {
        readonly AvbdBodyBuffers bodies;
        readonly AvbdShaderSet shaders;
        readonly AvbdConstraintBuffers constraints;
        readonly int predictKernel;
        readonly int primalKernel;
        readonly int initKernel;
        readonly int dualKernel;
        AvbdSolverSettings settings;

        static readonly int RecordsId = Shader.PropertyToID("_Constraints");
        static readonly int ListId = Shader.PropertyToID("_ConstraintList");
        static readonly int ColorBodiesId = Shader.PropertyToID("_ColorBodies");
        static readonly int StatusId = Shader.PropertyToID("_Status");

        /// <param name="constraints">Joints and springs, or null for a solver of free bodies only.</param>
        public AvbdSolver(AvbdBodyBuffers bodies, AvbdSolverSettings settings, AvbdShaderSet shaderSet = null, AvbdConstraintBuffers constraints = null)
        {
            this.bodies = bodies ?? throw new System.ArgumentNullException(nameof(bodies));
            this.constraints = constraints;
            settings.Validate();
            this.settings = settings;

            shaders = shaderSet != null ? shaderSet : AvbdShaderSet.Load();
            if (shaders == null)
                throw new System.InvalidOperationException("The AvbdShaderSet asset was not found.");
            if (shaders.Predict == null)
                throw new System.InvalidOperationException("The AvbdShaderSet has no predict shader.");

            predictKernel = shaders.Predict.FindKernel("Predict");

            if (constraints != null)
            {
                if (shaders.Primal == null)
                    throw new System.InvalidOperationException("The AvbdShaderSet has no primal shader.");
                primalKernel = shaders.Primal.FindKernel("Primal");

                if (shaders.Dual == null)
                    throw new System.InvalidOperationException("The AvbdShaderSet has no dual shader.");
                initKernel = shaders.Dual.FindKernel("ConstraintInit");
                dualKernel = shaders.Dual.FindKernel("Dual");
            }
        }

        /// <summary>Solver constants. Takes effect at the next step. Throws ArgumentException for out-of-range values.</summary>
        public AvbdSolverSettings Settings
        {
            get => settings;
            set
            {
                value.Validate();
                settings = value;
            }
        }

        /// <summary>
        /// Applies pending body edits (stage E) and constraint changes, then runs the solver stages that exist:
        /// predict (P), the constraint warm start (D), then per iteration a primal sweep (V) and the dual update (D).
        /// The velocity stage (U) is not implemented yet.
        /// </summary>
        public void Step()
        {
            bodies.ApplyEdits();
            if (constraints != null && constraints.HasPendingChanges)
                constraints.Rebuild();

            DispatchPredict();
            if (constraints != null)
            {
                DispatchConstraintInit();
                for (int it = 0; it < settings.iterations; it++)
                {
                    DispatchPrimalSweep();
                    DispatchDual();
                }
            }
        }

        // Stage P over every slot that has ever been used.
        internal void DispatchPredict()
        {
            ComputeShader shader = shaders.Predict;
            SolverUniforms.Bind(shader, settings);
            bodies.BindTo(shader, predictKernel);
            AvbdDispatch.Linear(shader, predictKernel, bodies.HighWaterMark);
        }

        // Stage V: one primal sweep. One dispatch per color; the dispatches of one sweep run in order, because a body reads the
        // poses that the earlier colors just wrote. Does nothing without constraints.
        internal void DispatchPrimalSweep()
        {
            if (constraints == null)
                return;

            ComputeShader shader = shaders.Primal;
            SolverUniforms.Bind(shader, settings);
            bodies.BindTo(shader, primalKernel);
            shader.SetBuffer(primalKernel, RecordsId, constraints.Records);
            shader.SetBuffer(primalKernel, ListId, constraints.List);
            shader.SetBuffer(primalKernel, ColorBodiesId, constraints.ColorBodies);
            shader.SetBuffer(primalKernel, StatusId, constraints.Status);

            ConstraintTopology topology = constraints.Topology;
            for (int color = 0; color < topology.ColorCount; color++)
                AvbdDispatch.Range(shader, primalKernel, topology.ColorOffsets[color], topology.ColorSize(color));
        }

        // Stage D, once per step after stage P: evaluates every joint at the start-of-step pose and warm-starts its variables.
        internal void DispatchConstraintInit()
        {
            DispatchConstraintKernel(initKernel);
        }

        // Stage D, after every primal sweep: multiplier, penalty and fracture update of every joint.
        internal void DispatchDual()
        {
            DispatchConstraintKernel(dualKernel);
        }

        // One thread per constraint. Constraints are independent, so there is no coloring and one dispatch covers them all.
        void DispatchConstraintKernel(int kernel)
        {
            if (constraints == null || constraints.RecordCount == 0)
                return;

            ComputeShader shader = shaders.Dual;
            SolverUniforms.Bind(shader, settings);
            bodies.BindTo(shader, kernel);
            shader.SetBuffer(kernel, RecordsId, constraints.Records);
            AvbdDispatch.Linear(shader, kernel, constraints.RecordCount);
        }
    }
}
