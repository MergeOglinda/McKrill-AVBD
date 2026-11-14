using UnityEngine;

namespace McKrill.Avbd
{
    /// <summary>
    /// Runs the solver stages on the GPU for the bodies of one <see cref="AvbdBodyBuffers"/>.
    /// The solver does not own the body or constraint buffers; dispose them separately. Without constraint buffers it creates an
    /// empty set of its own at the first step (so that free bodies still get their primal solve), and <see cref="Dispose"/> releases that one.
    /// <see cref="Step"/> runs a whole step: edits, constraint rebuild, predict, the iterations and the velocity update.
    /// Joints and springs come from an optional <see cref="AvbdConstraintBuffers"/>, which also owns the contact region if it was created with
    /// contact bodies: then every step runs the narrow phase over all pairs of those bodies (no broadphase).
    /// </summary>
    public sealed class AvbdSolver : System.IDisposable
    {
        readonly AvbdBodyBuffers bodies;
        readonly AvbdShaderSet shaders;
        AvbdConstraintBuffers constraints;
        bool ownsConstraints;
        AvbdBroadphase broadphase;
        readonly int predictKernel;
        readonly int primalKernel;
        readonly int initKernel;
        readonly int dualKernel;
        readonly int velocityKernel;
        readonly int narrowphaseKernel = -1;
        readonly int dualContactKernel = -1;
        AvbdSolverSettings settings;

        static readonly int RecordsId = Shader.PropertyToID("_Constraints");
        static readonly int ListId = Shader.PropertyToID("_ConstraintList");
        static readonly int ColorBodiesId = Shader.PropertyToID("_ColorBodies");
        static readonly int StatusId = Shader.PropertyToID("_Status");
        static readonly int PairCountsId = Shader.PropertyToID("_PairCounts");
        static readonly int PairExcludedId = Shader.PropertyToID("_PairExcluded");
        static readonly int DirtySlotsId = Shader.PropertyToID("_DirtySlots");
        static readonly int ContactBaseId = Shader.PropertyToID("_ContactBase");

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

            if (shaders.Primal == null)
                throw new System.InvalidOperationException("The AvbdShaderSet has no primal shader.");
            primalKernel = shaders.Primal.FindKernel("Primal");

            if (shaders.Dual == null)
                throw new System.InvalidOperationException("The AvbdShaderSet has no dual shader.");
            initKernel = shaders.Dual.FindKernel("ConstraintInit");
            dualKernel = shaders.Dual.FindKernel("Dual");

            if (shaders.Velocity == null)
                throw new System.InvalidOperationException("The AvbdShaderSet has no velocity shader.");
            velocityKernel = shaders.Velocity.FindKernel("Velocity");

            // The contact kernels are only needed with contacts, but an asset that has the dual shader has the kernel.
            dualContactKernel = shaders.Dual.FindKernel("DualContact");
            if (shaders.Narrowphase != null)
                narrowphaseKernel = shaders.Narrowphase.FindKernel("Narrowphase");
            else if (constraints != null && constraints.HasContacts)
                throw new System.InvalidOperationException("The AvbdShaderSet has no narrowphase shader.");
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
        /// Runs one whole step. Applies pending body edits (stage E) and constraint changes, then runs predict (P), the constraint
        /// warm start (D), per iteration a primal sweep (V) and the dual update (D), and finally the velocity update (U).
        /// The constraint lists are built on the CPU and the graph is colored on the GPU (stage C). With contacts, the narrow phase (N) runs after
        /// predict, and the contact counts are read back to rebuild the lists and the coloring every step (a stall, until stage G runs on the
        /// GPU). Stage B (the broadphase) only runs when enabled and its pairs are not used yet (the narrow phase tests all pairs). Stages S, G, Z and R do not exist yet; the constraints come from <see cref="AvbdConstraintBuffers"/>.
        /// Must be called from the main thread. The previous step's results are the next step's input, so call it once per time step.
        /// </summary>
        public void Step()
        {
            bodies.ApplyEdits();
            if (constraints == null)
            {
                constraints = new AvbdConstraintBuffers(bodies, 1, shaders);
                ownsConstraints = true;
            }
            if (constraints.NeedsRebuild)
                constraints.Rebuild();

            // Stage B, on the poses of the end of the previous step, like the reference solver's broadphase.
            broadphase?.Run(bodies, constraints);

            DispatchPredict();
            if (constraints.HasContacts)
            {
                constraints.PrepareContacts();
                DispatchNarrowphase();
                constraints.RefreshContactTopology();
            }
            DispatchConstraintInit();
            for (int it = 0; it < settings.iterations; it++)
            {
                DispatchPrimalSweep();
                DispatchDual();
                DispatchDualContacts();
            }
            DispatchVelocity();
        }

        /// <summary>True if <see cref="Step"/> runs the broadphase (stage B).</summary>
        public bool BroadphaseEnabled => broadphase != null;

        /// <summary>
        /// Makes <see cref="Step"/> run the broadphase (stage B) on the poses of the end of the previous step. The pairs stay in a GPU
        /// buffer. The narrowphase tests all pairs by itself and does not read them yet, so nothing changes in the simulation. Calling it again replaces the
        /// broadphase (and its pair capacities). Needs the body buffers' capacity to stay as it is, which it does.
        /// </summary>
        /// <param name="dynamicPairCapacity">Pairs of two solved bodies that can be stored; more are counted and dropped.</param>
        /// <param name="staticPairCapacity">Pairs of a solved body and a static or kinematic one that can be stored.</param>
        public void EnableBroadphase(int dynamicPairCapacity, int staticPairCapacity = 1024)
        {
            if (shaders.Broadphase == null)
                throw new System.InvalidOperationException("The AvbdShaderSet has no broadphase shader.");
            broadphase?.Dispose();
            broadphase = new AvbdBroadphase(shaders.Broadphase, bodies.Capacity, dynamicPairCapacity, staticPairCapacity);
        }

        /// <summary>Stops running the broadphase and releases its buffers.</summary>
        public void DisableBroadphase()
        {
            broadphase?.Dispose();
            broadphase = null;
        }

        /// <summary>The broadphase, or null if <see cref="EnableBroadphase"/> has not been called.</summary>
        internal AvbdBroadphase Broadphase => broadphase;

        /// <summary>Releases the constraint buffers the solver created for itself, if any, and the broadphase. Safe to call more than once.</summary>
        public void Dispose()
        {
            DisableBroadphase();
            if (ownsConstraints)
                constraints.Dispose();
            constraints = null;
            ownsConstraints = false;
        }

        // Stage P over every slot that has ever been used.
        internal void DispatchPredict()
        {
            ComputeShader shader = shaders.Predict;
            SolverUniforms.Bind(shader, settings);
            bodies.BindTo(shader, predictKernel);
            AvbdDispatch.Linear(shader, predictKernel, bodies.HighWaterMark);
        }

        // Stage U over every slot that has ever been used. Reads the pose and the start-of-step pose, writes the Motion record.
        internal void DispatchVelocity()
        {
            ComputeShader shader = shaders.Velocity;
            SolverUniforms.Bind(shader, settings);
            bodies.BindTo(shader, velocityKernel);
            AvbdDispatch.Linear(shader, velocityKernel, bodies.HighWaterMark);
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

        // Stage N over every pair of the used contact bodies, after stage P (it reads the start-of-step poses).
        internal void DispatchNarrowphase()
        {
            if (constraints == null || !constraints.HasContacts)
                return;
            int pairs = constraints.ContactPairsInUse;
            if (pairs == 0)
                return;

            ComputeShader shader = shaders.Narrowphase;
            SolverUniforms.Bind(shader, settings);
            bodies.BindTo(shader, narrowphaseKernel);
            shader.SetBuffer(narrowphaseKernel, RecordsId, constraints.Records);
            shader.SetBuffer(narrowphaseKernel, PairCountsId, constraints.PairCountsBuffer);
            shader.SetBuffer(narrowphaseKernel, PairExcludedId, constraints.PairExcludedBuffer);
            shader.SetBuffer(narrowphaseKernel, DirtySlotsId, constraints.DirtySlotsBuffer);
            shader.SetInt(ContactBaseId, constraints.ContactBase);
            AvbdDispatch.Linear(shader, narrowphaseKernel, pairs);
        }

        // Stage D for the contacts, after every primal sweep: one thread per contact record of the used pairs.
        internal void DispatchDualContacts()
        {
            if (constraints == null || !constraints.HasContacts)
                return;
            int pairs = constraints.ContactPairsInUse;
            if (pairs == 0)
                return;

            ComputeShader shader = shaders.Dual;
            SolverUniforms.Bind(shader, settings);
            bodies.BindTo(shader, dualContactKernel);
            shader.SetBuffer(dualContactKernel, RecordsId, constraints.Records);
            shader.SetBuffer(dualContactKernel, PairCountsId, constraints.PairCountsBuffer);
            shader.SetInt(ContactBaseId, constraints.ContactBase);
            AvbdDispatch.Range(shader, dualContactKernel, constraints.ContactBase, pairs * ContactLayout.MaxContactsPerPair);
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
