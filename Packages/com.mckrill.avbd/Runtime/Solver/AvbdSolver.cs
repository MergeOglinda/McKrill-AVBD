using Unity.Mathematics;
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
        readonly int predictExplosionsKernel;
        readonly int resolveExplosionsKernel;
        readonly int predictForcesKernel;
        readonly int predictForcesExplosionsKernel;
        readonly int forceFieldClearKernel;
        readonly int forceFieldAccumulateKernel;
        readonly int solveHeaderKernel;
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
        uint dualSweepCounter; // ids of the sweeps that run a fused dual update: 1, 2, ..., never 0 (see DispatchPrimalSweep)
        uint lastDualSweep;    // the id of the last of them in this step, 0 if the step has none

        /// <summary>
        /// Run the dual update of the joints inside the primal sweeps (default true), not in a dispatch of its own after every sweep. The update after sweep k
        /// then happens at the start of sweep k + 1, in the thread of the first body of the joint that is solved (the one of the lower color, so that
        /// neither body has moved yet and the joint sees the poses of the end of sweep k), and one dual dispatch after the last sweep does the rest: with
        /// 30 iterations that is 1 dual dispatch instead of 30, and 29 fewer reads of every joint record and of both of its poses. The result is the same as
        /// without, to rounding (a joint that no solved body touches, such as one between two static bodies, gets all its updates in the last dispatch, one
        /// after the other). It adds the update to the register-bound primal kernel, so it is a switch: compare both in a GPU trace. Takes effect at the
        /// next step. Contacts are not affected (<see cref="DispatchDualContacts"/> still runs after every sweep).
        /// </summary>
        public bool FuseDualUpdate { get; set; } = true;

        // Keyword of the variants of the primal and dual shaders that read the plastic array (AvbdPlasticRecords.hlsl).
        const string PlasticKeyword = "AVBD_PLASTIC_JOINTS";
        // Keyword of the variant of the primal shader that runs the dual update of the joints itself (see FuseDualUpdate).
        const string FusedDualKeyword = "AVBD_FUSED_DUAL";
        static readonly int PlasticId = Shader.PropertyToID("_Plastic");
        static readonly int RecordsId = Shader.PropertyToID("_Constraints");
        static readonly int FractureId = Shader.PropertyToID("_Fracture");
        static readonly int ListId = Shader.PropertyToID("_ConstraintList");
        static readonly int ColorBodiesId = Shader.PropertyToID("_ColorBodies");
        static readonly int ColorWorkId = Shader.PropertyToID("_ColorWork");
        static readonly int ColorId = Shader.PropertyToID("_Color");
        static readonly int DualSweepId = Shader.PropertyToID("_DualSweep");
        static readonly int ClearMarksId = Shader.PropertyToID("_ClearMarks");
        static readonly int LastSweepId = Shader.PropertyToID("_LastSweep");
        static readonly int FinalUpdatesId = Shader.PropertyToID("_FinalUpdates");
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
        static readonly int ExplosionsId = Shader.PropertyToID("_Explosions");
        static readonly int ExplosionCountId = Shader.PropertyToID("_ExplosionCount");
        static readonly int ForceFieldsId = Shader.PropertyToID("_ForceFields");
        static readonly int ForceFieldCountId = Shader.PropertyToID("_ForceFieldCount");
        static readonly int ForceFieldAccumId = Shader.PropertyToID("_ForceFieldAccum");

        /// <summary>Most explosions one <see cref="Step"/> applies (<see cref="Explode"/> returns false for more). Mirrors AVBD_MAX_EXPLOSIONS in AvbdExplosions.hlsl.</summary>
        public const int MaxExplosionsPerStep = 16;

        // Explosions queued for the next step, and the buffer they are uploaded to (made at the first explosion).
        readonly ExplosionRecord[] pendingExplosions = new ExplosionRecord[MaxExplosionsPerStep];
        int pendingExplosionCount;
        int pendingAnchoredCount; // of those, the ones whose center is relative to a body
        uint explosionSeed;
        GraphicsBuffer explosionBuffer;

        /// <summary>Most force fields a solver holds at once (<see cref="AddForceField"/> returns an invalid handle for more). Mirrors AVBD_MAX_FORCE_FIELDS in AvbdForceFields.hlsl.</summary>
        public const int MaxForceFields = 16;

        // The force fields: a small table, kept here and uploaded when it changes (made at the first use; 16 records of 64 B). An empty slot has kind 0.
        readonly ForceFieldRecord[] forceFieldRecords = new ForceFieldRecord[MaxForceFields];
        readonly uint[] forceFieldGenerations = new uint[MaxForceFields];
        readonly bool[] forceFieldUsed = new bool[MaxForceFields];
        int forceFieldCount;     // fields in use
        int forceFieldSlots;     // one past the last slot in use: what the shader loops over
        bool forceFieldsDirty;
        GraphicsBuffer forceFieldBuffer;
        GraphicsBuffer forceFieldAccumBuffer; // 4 words per field: the centroid sums of the spring fields (made with the first force field)
        int springCount;                      // spring fields in use

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
            predictExplosionsKernel = shaders.Predict.FindKernel("PredictExplosions");
            resolveExplosionsKernel = shaders.Predict.FindKernel("ResolveExplosions");
            predictForcesKernel = shaders.Predict.FindKernel("PredictForces");
            predictForcesExplosionsKernel = shaders.Predict.FindKernel("PredictForcesExplosions");
            forceFieldClearKernel = shaders.Predict.FindKernel("ForceFieldClear");
            forceFieldAccumulateKernel = shaders.Predict.FindKernel("ForceFieldAccumulate");
            solveHeaderKernel = shaders.Predict.FindKernel("BuildSolveHeader");

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
            DispatchSolveHeader();
            bool fused = FuseDualUpdate && constraints.RecordCount > 0;
            lastDualSweep = 0u;
            for (int it = 0; it < settings.iterations; it++)
            {
                DispatchPrimalSweep(fused, fused && it > 0); // the first sweep has no dual update before it
                if (!fused)
                    DispatchDual();
                DispatchDualContacts();
            }
            if (fused && settings.iterations > 0)
                DispatchFinalDual(); // the update of the last sweep, which no sweep follows, and the marks of the sweeps
            DispatchVelocity();
        }

        /// <summary>
        /// Queues an explosion for the next <see cref="Step"/>: a one-shot radial impulse. In that step's predict stage every solved body (not
        /// static, not kinematic) whose center of mass is closer than <paramref name="radius"/> to <paramref name="center"/> has its velocity changed by
        /// <c>impulse * (1 - distance / radius)^2 / mass</c> along the line from the center, so a light body flies faster than a heavy one and
        /// nothing is moved at the edge. The impulse acts a little off the center of mass (a quarter of the body's radius, in a direction hashed from the
        /// slot and the explosion), so a body is also given a small spin. There is no upward bias and no shock wave: every body in range is hit in the same step.
        /// The constraints and contacts solve the step like any other, so a jointed structure is loaded through its joints.
        /// </summary>
        /// <param name="center">World position of the explosion.</param>
        /// <param name="radius">Reach in meters; above zero.</param>
        /// <param name="impulse">Impulse in N s that a body at the center would receive. May be negative (an implosion).</param>
        /// <returns>False if <see cref="MaxExplosionsPerStep"/> explosions are already queued for the next step; the explosion is dropped.</returns>
        /// <exception cref="System.ArgumentException">A value is not finite, or the radius is not above zero.</exception>
        public bool Explode(float3 center, float radius, float impulse)
        {
            if (!math.all(math.isfinite(center)) || !float.IsFinite(radius) || !float.IsFinite(impulse))
                throw new System.ArgumentException("An explosion needs finite values.");
            if (!(radius > 0f))
                throw new System.ArgumentException("The radius of an explosion must be above zero.", nameof(radius));
            if (pendingExplosionCount >= MaxExplosionsPerStep)
                return false;

            pendingExplosions[pendingExplosionCount++] = new ExplosionRecord
            {
                centerRadius = new float4(center, radius),
                impulse = impulse,
                seed = explosionSeed++,
            };
            return true;
        }

        /// <summary>
        /// As <see cref="Explode(float3, float, float)"/>, with the center given relative to a body: <paramref name="localCenter"/> is in the frame of
        /// <paramref name="body"/> (origin at its center of mass, along its axes). At the start of the step that applies it, the GPU turns it into the world
        /// position of that point of the body from the pose the last step left, so an explosion that was asked for a few frames ago (a ray cast answers late)
        /// goes off where that point of the body is now, not where it was. A body that was removed before the step applies it gives no explosion.
        /// </summary>
        /// <param name="body">The body the center is relative to.</param>
        /// <param name="localCenter">The center in the frame of the body, in meters.</param>
        /// <returns>False if the body is not alive, or if <see cref="MaxExplosionsPerStep"/> explosions are already queued for the next step; the explosion is dropped.</returns>
        /// <exception cref="System.ArgumentException">A value is not finite, or the radius is not above zero.</exception>
        public bool Explode(BodyHandle body, float3 localCenter, float radius, float impulse)
        {
            if (!math.all(math.isfinite(localCenter)) || !float.IsFinite(radius) || !float.IsFinite(impulse))
                throw new System.ArgumentException("An explosion needs finite values.");
            if (!(radius > 0f))
                throw new System.ArgumentException("The radius of an explosion must be above zero.", nameof(radius));
            if (!bodies.IsAlive(body))
                return false;
            if (pendingExplosionCount >= MaxExplosionsPerStep)
                return false;

            pendingExplosions[pendingExplosionCount++] = new ExplosionRecord
            {
                centerRadius = new float4(float3.zero, radius), // the center is resolved on the GPU
                impulse = impulse,
                seed = explosionSeed++,
                anchorSlot = (uint)body.Slot + 1u,
                anchorLocal = new float4(localCenter, 0f),
            };
            pendingAnchoredCount++;
            return true;
        }

        /// <summary>
        /// Adds a force that acts on a group of bodies in every <see cref="Step"/> from now on, until it is removed: stage P adds its acceleration to the
        /// inertial target (the same place as gravity), so contacts and joints resist it like gravity. The group is the bodies whose group mask has a bit
        /// in common with the mask of the field; the label is the group mask that every body already has, so no per-body data is added. A body in
        /// several groups gets the sum of their fields. Examples: <c>AddForceField(ForceFieldDescriptor.Wing(wings, up, area))</c> gives lift to the blocks
        /// labelled <c>wings</c> only; <c>ForceFieldDescriptor.GravityScale(light, 0.3f)</c> makes the bodies labelled <c>light</c> fall slowly.
        /// Static and kinematic bodies are never moved. A step without fields runs the plain predict kernel.
        /// </summary>
        /// <returns>The handle of the field, or <see cref="ForceFieldHandle.Invalid"/> if <see cref="MaxForceFields"/> fields exist already.</returns>
        /// <exception cref="System.ArgumentException">A value is not valid for the kind of the field (see <see cref="ForceFieldDescriptor"/>).</exception>
        public ForceFieldHandle AddForceField(in ForceFieldDescriptor field)
        {
            field.Validate();
            for (int slot = 0; slot < MaxForceFields; slot++)
            {
                if (forceFieldUsed[slot])
                    continue;

                forceFieldUsed[slot] = true;
                forceFieldGenerations[slot]++;
                forceFieldRecords[slot] = field.ToRecord();
                forceFieldCount++;
                if (field.kind == ForceKind.Spring)
                    springCount++;
                forceFieldSlots = System.Math.Max(forceFieldSlots, slot + 1);
                forceFieldsDirty = true;
                return new ForceFieldHandle(slot, forceFieldGenerations[slot]);
            }
            return ForceFieldHandle.Invalid;
        }

        /// <summary>Replaces what a field does (its kind may change too), for example to change a thrust or a gravity scale while it runs. The new values apply from the next step.</summary>
        /// <returns>False if the handle is stale or invalid.</returns>
        /// <exception cref="System.ArgumentException">A value is not valid for the kind of the field.</exception>
        public bool UpdateForceField(ForceFieldHandle handle, in ForceFieldDescriptor field)
        {
            field.Validate();
            if (!IsForceFieldAlive(handle))
                return false;

            forceFieldRecords[handle.Slot] = field.ToRecord();
            forceFieldsDirty = true;
            return true;
        }

        /// <summary>Removes a force field. Its slot can be given to a new field; the old handle stays stale.</summary>
        /// <returns>False if the handle is stale or invalid.</returns>
        public bool RemoveForceField(ForceFieldHandle handle)
        {
            if (!IsForceFieldAlive(handle))
                return false;

            int slot = handle.Slot;
            if (forceFieldRecords[slot].ident.x == (uint)ForceKind.Spring)
                springCount--;
            forceFieldUsed[slot] = false;
            forceFieldRecords[slot] = default;
            forceFieldCount--;
            while (forceFieldSlots > 0 && !forceFieldUsed[forceFieldSlots - 1])
                forceFieldSlots--;
            forceFieldsDirty = true;
            return true;
        }

        /// <summary>True if the handle is of a force field that has not been removed.</summary>
        public bool IsForceFieldAlive(ForceFieldHandle handle)
        {
            return handle.IsValid && handle.Slot < MaxForceFields && forceFieldUsed[handle.Slot] && forceFieldGenerations[handle.Slot] == handle.Generation;
        }

        /// <summary>Number of force fields in use.</summary>
        public int ForceFieldCount => forceFieldCount;

        /// <summary>Number of explosions queued for the next <see cref="Step"/>.</summary>
        public int PendingExplosionCount => pendingExplosionCount;

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
            explosionBuffer?.Release();
            explosionBuffer = null;
            pendingExplosionCount = 0;
            pendingAnchoredCount = 0;
            forceFieldBuffer?.Release();
            forceFieldBuffer = null;
            forceFieldAccumBuffer?.Release();
            forceFieldAccumBuffer = null;
            if (ownsConstraints)
                constraints.Dispose();
            constraints = null;
            ownsConstraints = false;
        }

        // Stage P over every slot that has ever been used. With explosions queued it runs the variant that applies them first, and the queue is emptied. With
        // force fields it runs the variant that adds their acceleration.
        internal void DispatchPredict()
        {
            ComputeShader shader = shaders.Predict;
            SolverUniforms.Bind(shader, settings);

            bool forces = forceFieldCount > 0;
            bool blasts = pendingExplosionCount > 0;
            int kernel = forces ? (blasts ? predictForcesExplosionsKernel : predictForcesKernel) : (blasts ? predictExplosionsKernel : predictKernel);

            if (forces)
            {
                if (forceFieldBuffer == null)
                {
                    forceFieldBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, MaxForceFields, Unity.Collections.LowLevel.Unsafe.UnsafeUtility.SizeOf<ForceFieldRecord>());
                    forceFieldsDirty = true;
                }
                if (forceFieldsDirty)
                {
                    forceFieldBuffer.SetData(forceFieldRecords);
                    forceFieldsDirty = false;
                }
                shader.SetBuffer(kernel, ForceFieldsId, forceFieldBuffer);
                shader.SetInt(ForceFieldCountId, forceFieldSlots);

                // The kernel declares the sums of the springs even when no spring exists, so the buffer is always bound (256 bytes).
                if (forceFieldAccumBuffer == null)
                    forceFieldAccumBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, MaxForceFields * 4, sizeof(uint));
                shader.SetBuffer(kernel, ForceFieldAccumId, forceFieldAccumBuffer);

                if (springCount > 0)
                {
                    // The spring fields pull the centroid of their group: sum it first, over the poses at the start of the step.
                    shader.SetBuffer(forceFieldClearKernel, ForceFieldAccumId, forceFieldAccumBuffer);
                    shader.Dispatch(forceFieldClearKernel, 1, 1, 1);

                    shader.SetBuffer(forceFieldAccumulateKernel, ForceFieldsId, forceFieldBuffer);
                    shader.SetInt(ForceFieldCountId, forceFieldSlots);
                    shader.SetBuffer(forceFieldAccumulateKernel, ForceFieldAccumId, forceFieldAccumBuffer);
                    bodies.BindTo(shader, forceFieldAccumulateKernel);
                    AvbdDispatch.Linear(shader, forceFieldAccumulateKernel, bodies.HighWaterMark);
                }
            }

            if (blasts)
            {
                if (explosionBuffer == null)
                    explosionBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, MaxExplosionsPerStep, Unity.Collections.LowLevel.Unsafe.UnsafeUtility.SizeOf<ExplosionRecord>());
                explosionBuffer.SetData(pendingExplosions, 0, 0, pendingExplosionCount);
                shader.SetInt(ExplosionCountId, pendingExplosionCount);
                if (pendingAnchoredCount > 0)
                {
                    // Centers relative to a body become world positions first, from the poses before this predict moves anything.
                    shader.SetBuffer(resolveExplosionsKernel, ExplosionsId, explosionBuffer);
                    bodies.BindTo(shader, resolveExplosionsKernel);
                    shader.Dispatch(resolveExplosionsKernel, 1, 1, 1);
                }
                shader.SetBuffer(kernel, ExplosionsId, explosionBuffer);
                pendingExplosionCount = 0;
                pendingAnchoredCount = 0;
            }

            bodies.BindTo(shader, kernel);
            AvbdDispatch.Linear(shader, kernel, bodies.HighWaterMark);
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
        //
        // `fusedDual` picks the variant of the shader that can run the dual update of the joints itself (AVBD_FUSED_DUAL), and `dualOfPreviousSweep` makes this
        // sweep do it: the sweep then updates every joint that a solved body touches, with the poses of the end of the sweep before, so the first sweep of a
        // step does not, and every step that fuses ends with DispatchDual(true). Without fusedDual it is the plain sweep.
        internal void DispatchPrimalSweep(bool fusedDual = false, bool dualOfPreviousSweep = false)
        {
            if (constraints == null)
                return;

            ComputeShader shader = shaders.Primal;
            if (fusedDual)
                shader.EnableKeyword(FusedDualKeyword);
            else
                shader.DisableKeyword(FusedDualKeyword);
            uint sweepId = 0u;
            if (fusedDual && dualOfPreviousSweep)
            {
                // A joint that a sweep has updated holds this id, so the next sweep does not take it for updated. The marks are cleared at the end of every
                // step, so a wrap-around cannot meet an old one; ids stay below 2^31 (the mark is a float word, and a NaN is better avoided).
                dualSweepCounter = dualSweepCounter >= 0x7FFFFFFFu ? 1u : dualSweepCounter + 1u;
                sweepId = dualSweepCounter;
                lastDualSweep = sweepId;
            }
            shader.SetInt(DualSweepId, (int)sweepId);
            SolverUniforms.Bind(shader, settings);
            bodies.BindTo(shader, primalKernel);
            shader.SetBuffer(primalKernel, RecordsId, constraints.Records);
            if (fusedDual)
                shader.SetBuffer(primalKernel, FractureId, constraints.Fractures); // only the variant that runs the dual update reads it
            BindPlastic(shader, primalKernel);
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

        // Stage V, once per step before the first sweep: gathers the flags, the moments and mass and the range in the constraint list of every body into
        // the 32 B record (BodySolveRecord) that the sweeps read, in place of those three arrays. Run it after the edits, the lists, the colors and the
        // predict of the step, and again after anything that changes a body's flags, mass or list before a sweep that is dispatched alone (the tests do).
        internal void DispatchSolveHeader()
        {
            if (constraints == null)
                return;

            ComputeShader shader = shaders.Predict;
            bodies.BindTo(shader, solveHeaderKernel);
            AvbdDispatch.Linear(shader, solveHeaderKernel, bodies.HighWaterMark);
        }

        // Stage D, once per step after stage P: evaluates every joint at the start-of-step pose and warm-starts its variables.
        internal void DispatchConstraintInit()
        {
            DispatchConstraintKernel(initKernel);
        }

        // Stage D, after every primal sweep: multiplier, penalty and fracture update of every joint.
        internal void DispatchDual()
        {
            shaders.Dual.SetInt(ClearMarksId, 0);
            DispatchConstraintKernel(dualKernel);
        }

        // Stage D, after the last sweep of a step that fused the update into the sweeps: the update of that last sweep, the clearing of the marks the sweeps
        // left in the records, and for a joint that no sweep touched (no solved body is on it) the updates of all the iterations.
        internal void DispatchFinalDual()
        {
            ComputeShader shader = shaders.Dual;
            shader.SetInt(ClearMarksId, 1);
            shader.SetInt(LastSweepId, (int)lastDualSweep);
            shader.SetInt(FinalUpdatesId, settings.iterations);
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
            if (kernel == dualKernel)
                shader.SetBuffer(kernel, FractureId, constraints.Fractures); // the fracture thresholds: only the dual update reads them
            BindPlastic(shader, kernel);
            AvbdDispatch.Linear(shader, kernel, constraints.RecordCount);
        }

        // Picks the variant of the shader the constraint buffers need: with a plastic array the kernels that read it, otherwise the kernels without any
        // plastic code (and no extra buffer). The keyword belongs to the shader asset, so it is set before every dispatch, not once.
        void BindPlastic(ComputeShader shader, int kernel)
        {
            if (constraints.HasPlasticJoints)
            {
                shader.EnableKeyword(PlasticKeyword);
                shader.SetBuffer(kernel, PlasticId, constraints.PlasticBuffer);
            }
            else
            {
                shader.DisableKeyword(PlasticKeyword);
            }
        }
    }
}
