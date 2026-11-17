using UnityEngine;

namespace McKrill.Avbd
{
    /// <summary>
    /// Runs the solver stages on the GPU for the bodies of one <see cref="AvbdBodyBuffers"/>.
    /// The solver does not own the body or constraint buffers; dispose them separately. Without constraint buffers it creates an
    /// empty set of its own at the first step (so that free bodies still get their primal solve), and <see cref="Dispose"/> releases that one.
    /// <see cref="Step"/> runs a whole step: edits, constraint rebuild, predict, the iterations and the velocity update.
    /// Joints and springs come from an optional <see cref="AvbdConstraintBuffers"/>, which also owns the manifold pool if it was created with
    /// a contact capacity: then every step runs the broadphase (stage B) and the narrow phase (stage N) on the pairs it finds.
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
        readonly int npClearKernel = -1;
        readonly int npPlanKernel = -1;
        readonly int npRetireKernel = -1;
        readonly int npAllocateKernel = -1;
        readonly int dualContactKernel = -1;
        AvbdSolverSettings settings;

        static readonly int RecordsId = Shader.PropertyToID("_Constraints");
        static readonly int ListId = Shader.PropertyToID("_ConstraintList");
        static readonly int ColorBodiesId = Shader.PropertyToID("_ColorBodies");
        static readonly int ColorWorkId = Shader.PropertyToID("_ColorWork");
        static readonly int ColorId = Shader.PropertyToID("_Color");
        static readonly int StatusId = Shader.PropertyToID("_Status");
        static readonly int ContactPairsId = Shader.PropertyToID("_ContactPairs");
        static readonly int ContactPointsId = Shader.PropertyToID("_ContactPoints");
        static readonly int PairTableId = Shader.PropertyToID("_PairTable");
        static readonly int FreeListId = Shader.PropertyToID("_FreeList");
        static readonly int NpControlId = Shader.PropertyToID("_NpControl");
        static readonly int PairsId = Shader.PropertyToID("_Pairs");
        static readonly int BpControlId = Shader.PropertyToID("_BpControl");
        static readonly int NpArgsId = Shader.PropertyToID("_NpArgs");
        static readonly int BinId = Shader.PropertyToID("_Bin");
        static readonly int SlotCapacityId = Shader.PropertyToID("_SlotCapacity");
        static readonly int TableMaskId = Shader.PropertyToID("_TableMask");
        static readonly int DynamicPairCapacityId = Shader.PropertyToID("_DynamicPairCapacity");
        static readonly int StaticPairCapacityId = Shader.PropertyToID("_StaticPairCapacity");
        static readonly int DirtySlotsId = Shader.PropertyToID("_DirtySlots");

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
            {
                npClearKernel = shaders.Narrowphase.FindKernel("NpClearTable");
                npPlanKernel = shaders.Narrowphase.FindKernel("NpPlan");
                npRetireKernel = shaders.Narrowphase.FindKernel("NpRetire");
                npAllocateKernel = shaders.Narrowphase.FindKernel("NpAllocate");
                narrowphaseKernel = shaders.Narrowphase.FindKernel("Narrowphase");
            }
            if (constraints != null && constraints.HasContacts)
            {
                if (shaders.Narrowphase == null)
                    throw new System.InvalidOperationException("The AvbdShaderSet has no narrowphase shader.");
                if (shaders.Broadphase == null)
                    throw new System.InvalidOperationException("The AvbdShaderSet has no broadphase shader.");
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
        /// Runs one whole step. Applies pending body edits (stage E) and constraint changes, then runs the broadphase (B) and predict (P), with
        /// contacts the narrow phase (N), the constraint warm start (D), per iteration a primal sweep (V) and the dual update (D), and finally the
        /// velocity update (U).
        /// The constraint lists (stage G) are built and the graph is colored (stage C) on the GPU. With contacts, both are rebuilt every step after the
        /// narrow phase; neither waits for the GPU: the lists need no readback, and the coloring keeps the colors of the step before and is tuned from statistics that
        /// arrive a few frames late (<see cref="AvbdConstraintBuffers.PersistentColoring"/>). A rebuild (constraints or bodies added) still colors from scratch and waits.
        /// Stages S, Z and R do not exist yet; the constraints come from <see cref="AvbdConstraintBuffers"/>.
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
            DispatchBroadphase();

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
        /// Makes <see cref="Step"/> run the broadphase (stage B) on the poses of the end of the previous step. The pairs stay in a GPU buffer.
        /// With contacts the solver runs the broadphase anyway (the narrow phase reads its pairs), with capacities derived from the contact
        /// capacity; this replaces it with one of the given capacities. Without contacts nothing reads the pairs, so this only costs time and is
        /// for tools and tests. Calling it again replaces the broadphase.
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

        /// <summary>Stops running the broadphase and releases its buffers. Not possible with contacts: the narrow phase needs its pairs.</summary>
        public void DisableBroadphase()
        {
            if (constraints != null && constraints.HasContacts)
                throw new System.InvalidOperationException("The narrow phase reads the pairs of the broadphase; it cannot be disabled with contacts.");
            broadphase?.Dispose();
            broadphase = null;
        }

        /// <summary>The broadphase, or null if there are no contacts and <see cref="EnableBroadphase"/> has not been called.</summary>
        internal AvbdBroadphase Broadphase => broadphase;

        /// <summary>Releases the constraint buffers the solver created for itself, if any, and the broadphase. Safe to call more than once.</summary>
        public void Dispose()
        {
            broadphase?.Dispose();
            broadphase = null;
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
        // poses that the earlier colors just wrote. Each dispatch is indirect: the size of a color is only known on the GPU, where stage C
        // wrote the arguments, and the host only knows how many colors there are. Does nothing without constraints.
        internal void DispatchPrimalSweep()
        {
            if (constraints == null)
                return;

            ComputeShader shader = shaders.Primal;
            SolverUniforms.Bind(shader, settings);
            bodies.BindTo(shader, primalKernel);
            shader.SetBuffer(primalKernel, RecordsId, constraints.Records);
            shader.SetBuffer(primalKernel, ContactPairsId, constraints.ContactPairsBuffer);
            shader.SetBuffer(primalKernel, ContactPointsId, constraints.ContactPointsBuffer);
            shader.SetBuffer(primalKernel, ListId, constraints.List);
            shader.SetBuffer(primalKernel, ColorBodiesId, constraints.ColorBodies);
            shader.SetBuffer(primalKernel, ColorWorkId, constraints.ColorWork);
            shader.SetBuffer(primalKernel, StatusId, constraints.Status);

            for (int color = 0; color < constraints.ColorCount; color++)
            {
                shader.SetInt(ColorId, color);
                shader.DispatchIndirect(primalKernel, constraints.ColorArgs, (uint)(color * AvbdGraphColoring.ArgsStride));
            }
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

        // Stage B: with contacts it is created at the first step, with capacities that go with the contact capacity. Runs on the poses of the end
        // of the previous step, so call it before predict.
        internal void DispatchBroadphase()
        {
            if (broadphase == null && constraints != null && constraints.HasContacts)
                broadphase = new AvbdBroadphase(shaders.Broadphase, bodies.Capacity, constraints.CandidateCapacity, constraints.StaticCandidateCapacity);
            broadphase?.Run(bodies, constraints);
        }

        // Stage N, after stage B and stage P (it reads the start-of-step poses): frees the slots of recycled bodies, lets the new pairs of the
        // broadphase take slots, and collides every used slot. Every kernel but the first is dispatched indirectly, with sizes that NpPlan computed
        // on the GPU from the pool's high-water mark and the pair counts of the broadphase: the host reads no count, and the work follows the number
        // of manifolds and pairs, not the capacities. NpPlan runs twice: for the first kernels, and again for the sizes that depend on the slots
        // NpAllocate took (Narrowphase, and the dual update of the contacts after every primal sweep).
        internal void DispatchNarrowphase()
        {
            if (constraints == null || !constraints.HasContacts)
                return;
            if (broadphase == null)
                throw new System.InvalidOperationException("The narrow phase needs the pairs of the broadphase: run DispatchBroadphase first.");

            ComputeShader shader = shaders.Narrowphase;
            int slots = constraints.ContactCapacity;

            foreach (int kernel in new[] { npClearKernel, npPlanKernel, npRetireKernel, npAllocateKernel, narrowphaseKernel })
            {
                bodies.BindTo(shader, kernel);
                shader.SetBuffer(kernel, ContactPairsId, constraints.ContactPairsBuffer);
                shader.SetBuffer(kernel, ContactPointsId, constraints.ContactPointsBuffer);
                shader.SetBuffer(kernel, PairTableId, constraints.PairTableBuffer);
                shader.SetBuffer(kernel, FreeListId, constraints.FreeListBuffer);
                shader.SetBuffer(kernel, NpControlId, constraints.PoolControlBuffer);
                shader.SetBuffer(kernel, DirtySlotsId, constraints.DirtySlotsBuffer);
                shader.SetBuffer(kernel, PairsId, broadphase.Pairs);
                shader.SetBuffer(kernel, BpControlId, broadphase.Control);
                shader.SetBuffer(kernel, NpArgsId, constraints.NpArgsBuffer);
            }
            SolverUniforms.Bind(shader, settings);
            shader.SetInt(SlotCapacityId, slots);
            shader.SetInt(TableMaskId, constraints.PairTableSize - 1);
            shader.SetInt(DynamicPairCapacityId, broadphase.DynamicPairCapacity);
            shader.SetInt(StaticPairCapacityId, broadphase.StaticPairCapacity);

            AvbdDispatch.Linear(shader, npClearKernel, constraints.PairTableSize);
            shader.Dispatch(npPlanKernel, 1, 1, 1);
            shader.DispatchIndirect(npRetireKernel, constraints.NpArgsBuffer, NpArgsOffset(ContactLayout.PoolArgsRetire));

            shader.SetInt(BinId, 0);
            shader.DispatchIndirect(npAllocateKernel, constraints.NpArgsBuffer, NpArgsOffset(ContactLayout.PoolArgsAllocateDynamic));
            shader.SetInt(BinId, 1);
            shader.DispatchIndirect(npAllocateKernel, constraints.NpArgsBuffer, NpArgsOffset(ContactLayout.PoolArgsAllocateStatic));

            shader.Dispatch(npPlanKernel, 1, 1, 1); // the high-water mark may have grown
            shader.DispatchIndirect(narrowphaseKernel, constraints.NpArgsBuffer, NpArgsOffset(ContactLayout.PoolArgsNarrowphase));
        }

        static uint NpArgsOffset(int entry) => (uint)(entry * AvbdIndirect.ArgsStride);

        // Stage D for the contacts, after every primal sweep: one thread per manifold slot below the pool's high-water mark (a free slot skips), which
        // updates the points of its pair. Dispatched indirectly with the arguments the narrow phase of this step planned.
        internal void DispatchDualContacts()
        {
            if (constraints == null || !constraints.HasContacts)
                return;

            ComputeShader shader = shaders.Dual;
            SolverUniforms.Bind(shader, settings);
            bodies.BindTo(shader, dualContactKernel);
            shader.SetBuffer(dualContactKernel, ContactPairsId, constraints.ContactPairsBuffer);
            shader.SetBuffer(dualContactKernel, ContactPointsId, constraints.ContactPointsBuffer);
            shader.SetBuffer(dualContactKernel, NpControlId, constraints.PoolControlBuffer);
            shader.SetInt(SlotCapacityId, constraints.ContactCapacity);
            shader.DispatchIndirect(dualContactKernel, constraints.NpArgsBuffer, NpArgsOffset(ContactLayout.PoolArgsDual));
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
