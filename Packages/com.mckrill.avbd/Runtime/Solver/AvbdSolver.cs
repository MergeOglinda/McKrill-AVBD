using UnityEngine;

namespace McKrill.Avbd
{
    /// <summary>
    /// Runs the solver stages on the GPU for the bodies of one <see cref="AvbdBodyBuffers"/>.
    /// The solver does not own the buffers; dispose them separately.
    /// Stages are added one at a time; <see cref="Step"/> runs the ones that exist.
    /// </summary>
    public sealed class AvbdSolver
    {
        readonly AvbdBodyBuffers bodies;
        readonly AvbdShaderSet shaders;
        readonly int predictKernel;
        AvbdSolverSettings settings;

        public AvbdSolver(AvbdBodyBuffers bodies, AvbdSolverSettings settings, AvbdShaderSet shaderSet = null)
        {
            this.bodies = bodies ?? throw new System.ArgumentNullException(nameof(bodies));
            settings.Validate();
            this.settings = settings;

            shaders = shaderSet != null ? shaderSet : AvbdShaderSet.Load();
            if (shaders == null)
                throw new System.InvalidOperationException("The AvbdShaderSet asset was not found.");
            if (shaders.Predict == null)
                throw new System.InvalidOperationException("The AvbdShaderSet has no predict shader.");

            predictKernel = shaders.Predict.FindKernel("Predict");
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

        /// <summary>Applies pending body edits (stage E), then runs the solver stages that exist.</summary>
        public void Step()
        {
            bodies.ApplyEdits();
            DispatchPredict();
        }

        // Stage P over every slot that has ever been used.
        internal void DispatchPredict()
        {
            ComputeShader shader = shaders.Predict;
            SolverUniforms.Bind(shader, settings);
            bodies.BindTo(shader, predictKernel);
            AvbdDispatch.Linear(shader, predictKernel, bodies.HighWaterMark);
        }
    }
}
