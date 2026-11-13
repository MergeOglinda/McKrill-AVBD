using UnityEngine;

namespace McKrill.Avbd
{
    /// <summary>
    /// Runs the solver stages on the GPU for the bodies of one <see cref="AvbdBodyBuffers"/>.
    /// The solver does not own the body or constraint buffers you pass in; dispose them separately.
    /// Joints and springs come from an optional <see cref="AvbdConstraintBuffers"/>. Without one, <see cref="Step"/> creates a private empty
    /// set (which only schedules the bodies), so dispose the solver when you step it without passing constraints.
    /// </summary>
    public sealed class AvbdSolver : System.IDisposable
    {
        readonly AvbdBodyBuffers bodies;
        readonly AvbdShaderSet shaders;
        AvbdConstraintBuffers constraints;
        AvbdConstraintBuffers ownedConstraints; // created by Step when none was passed in
        readonly int predictKernel;
        readonly int velocityKernel;
        int primalKernel;
        int initKernel;
        int dualKernel;
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

            if (shaders.Velocity == null)
                throw new System.InvalidOperationException("The AvbdShaderSet has no velocity shader.");
            velocityKernel = shaders.Velocity.FindKernel("Velocity");

            if (constraints != null)
                FindConstraintKernels();
        }

        // The kernels of the constraint stages (V and D).
        void FindConstraintKernels()
        {
            if (shaders.Primal == null)
                throw new System.InvalidOperationException("The AvbdShaderSet has no primal shader.");
            primalKernel = shaders.Primal.FindKernel("Primal");

            if (shaders.Dual == null)
                throw new System.InvalidOperationException("The AvbdShaderSet has no dual shader.");
            initKernel = shaders.Dual.FindKernel("ConstraintInit");
            dualKernel = shaders.Dual.FindKernel("Dual");
        }

        /// <summary>Releases the private constraint buffers that <see cref="Step"/> created, if any. Safe to call more than once.</summary>
        public void Dispose()
        {
            ownedConstraints?.Dispose();
            ownedConstraints = null;
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
        /// One time step: applies pending body edits (stage E) and rebuilds the constraint schedule if the constraints or the set of solved
        /// bodies changed, then runs predict (P), the constraint warm start (D), per iteration a primal sweep (V) and the dual update (D),
        /// and finally the velocity update (U). Collisions are not implemented yet.
        /// </summary>
        public void Step()
        {
            bodies.ApplyEdits();

            if (constraints == null)
            {
                // Bodies without joints still need the primal update (inertia and gravity), which runs over the schedule.
                ownedConstraints = new AvbdConstraintBuffers(bodies, 1, shaders);
                constraints = ownedConstraints;
                FindConstraintKernels();
            }
            if (constraints.HasPendingChanges)
                constraints.Rebuild();

            DispatchPredict();
            DispatchConstraintInit();
            for (int it = 0; it < settings.iterations; it++)
            {
                DispatchPrimalSweep();
                DispatchDual();
            }
            DispatchVelocity();
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

            for (int color = 0; color < constraints.ColorCount; color++)
                AvbdDispatch.Range(shader, primalKernel, constraints.ColorOffset(color), constraints.ColorSize(color));
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

        // Stage U over every slot that has ever been used.
        internal void DispatchVelocity()
        {
            ComputeShader shader = shaders.Velocity;
            SolverUniforms.Bind(shader, settings);
            bodies.BindTo(shader, velocityKernel);
            AvbdDispatch.Linear(shader, velocityKernel, bodies.HighWaterMark);
        }
    }
}
