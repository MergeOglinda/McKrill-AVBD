using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd
{
    /// <summary>
    /// Owns the GPU arrays of joints and springs and the per-step schedule the primal stage needs: each solved body's constraint
    /// list and the bodies bucketed by graph color.
    ///
    /// AddJoint and AddSpring only record the constraint on the CPU. <see cref="Rebuild"/> (which <see cref="AvbdSolver.Step"/>
    /// calls when constraints are pending) applies the pending body edits, completes and uploads the records, builds the per-body lists
    /// (stage G) and colors the graph and buckets the bodies (stage C), both on the GPU.
    ///
    /// Contacts: with a nonzero <c>contactCapacity</c> the object also owns the manifold pool of stage N (see <see cref="ContactLayout"/>):
    /// <c>contactCapacity</c> slots, each holding the manifold of one pair of bodies as a 32 B pair record and up to 8 contact points of 64 B. The pairs come
    /// from the broadphase (stage B), which <see cref="AvbdSolver"/> runs when there are contacts. After the narrow phase,
    /// <see cref="RefreshContactTopology"/> rebuilds the lists (stage G) and the coloring (stage C), because they change every step. The lists
    /// are built from the pool and the records where they are, with no readback. The coloring keeps the colors of the step before and recolors
    /// only the bodies that now clash (<see cref="PersistentColoring"/>), so the host never waits for it: the number of colors the sweep dispatches
    /// is a cap that follows statistics read back a few frames late, and the size of every color stays on the GPU (see <see cref="AvbdGraphColoring"/>).
    /// <see cref="Rebuild"/> colors from scratch and waits, which gives the exact number of colors.
    ///
    /// Limits of this version: constraints cannot be removed, and removing a body that has constraints is not supported.
    /// The arrays have a fixed capacity. Methods must be called from the main thread.
    /// </summary>
    public sealed class AvbdConstraintBuffers : IDisposable
    {
        /// <summary>Largest constraint count: keeps constraint ids below the tag bit (rule P6) and the buffer below 2 GB.</summary>
        public const int MaxConstraints = 10_000_000;

        /// <summary>
        /// Most manifold slots. Each takes a pair record of 32 B and 8 contact points of 64 B, so this many take 272 MB, and a scene needs a few
        /// slots per touching pair of bodies: a body in a stack or pyramid touches about three others.
        /// </summary>
        public const int MaxContactManifolds = 500_000;

        /// <summary>Candidate pairs the broadphase can hold per manifold slot (a body's bounding sphere is larger than the body, so more pairs are candidates than touch).</summary>
        internal const int CandidatePairsPerSlot = 8;

        const int StatusWords = 4;

        struct Spec
        {
            public ConstraintKind kind;
            public uint slotA;
            public uint slotB;
            public float3 rA;
            public float3 rB;
            public float stiffnessLin; // joint: linear stiffness. spring: stiffness
            public float stiffnessAng;
            public float fractureOrRest; // joint: fracture threshold. spring: rest length, negative = current distance
        }

        readonly AvbdBodyBuffers bodies;
        readonly ConstraintRecord[] records;
        readonly List<Spec> pending = new List<Spec>();
        readonly AvbdGraphColoring coloring;
        readonly AvbdConstraintLists lists;
        int colorCount;
        ColorSchedule schedule;   // read back on demand, for tests and tools: the sweep itself never needs it on the CPU
        bool scheduleValid;
        GraphColoringOrder coloringOrder = GraphColoringOrder.Hashed;

        GraphicsBuffer recordBuffer;
        GraphicsBuffer listBuffer;
        GraphicsBuffer colorBuffer;
        GraphicsBuffer statusBuffer;
        GraphicsBuffer contactPairBuffer;  // per manifold slot: the pair of bodies (AVBD_INVALID for a free slot), the contact count, the stick bits, the normal and the friction
        GraphicsBuffer contactPointBuffer; // ContactLayout.MaxContactsPerPair contact points per manifold slot
        GraphicsBuffer pairTableBuffer;    // hash table from a pair to its slot (scratch of stage N)
        GraphicsBuffer freeListBuffer;     // stack of free manifold slots
        GraphicsBuffer poolControlBuffer;  // AVBD_NP_*
        GraphicsBuffer npArgsBuffer;       // indirect dispatch arguments of stage N and of the contacts' dual update, AVBD_NP_ARGS_*
        GraphicsBuffer dirtyBuffer;        // bit per body slot: spawned or removed since the last narrow phase

        readonly int contactSlots;
        readonly int tableSize;
        readonly uint[] dirtyBits;
        readonly HashSet<int> dirtySlots = new HashSet<int>();
        bool dirtyUploaded;

        int recordCount;   // records in the CPU array
        int uploadedCount; // records already on the GPU
        bool topologyDirty;
        int builtSolvedSetVersion;
        int brokenCountVersion = -1; // list build that BrokenCount was last read for
        int brokenCount;
        bool disposed;

        /// <param name="bodies">The body arrays the constraints refer to.</param>
        /// <param name="capacity">Number of joints and springs. Fixed for the lifetime of the object.</param>
        /// <param name="shaderSet">The compute shaders; null loads the package's shader set.</param>
        /// <param name="contactCapacity">Number of manifold slots (pairs of touching bodies that can have contacts at once), or 0 for no
        /// contacts. At most <see cref="MaxContactManifolds"/>. Pairs that find no free slot get no contacts until one is free
        /// (<see cref="ReadContacts"/> reports it).</param>
        public AvbdConstraintBuffers(AvbdBodyBuffers bodies, int capacity, AvbdShaderSet shaderSet = null, int contactCapacity = 0)
        {
            this.bodies = bodies ?? throw new ArgumentNullException(nameof(bodies));
            if (capacity < 1 || capacity > MaxConstraints)
                throw new ArgumentOutOfRangeException(nameof(capacity), "Constraint capacity must be between 1 and " + MaxConstraints + ".");
            if (contactCapacity < 0 || contactCapacity > MaxContactManifolds)
                throw new ArgumentOutOfRangeException(nameof(contactCapacity), "contactCapacity must be between 0 and " + MaxContactManifolds + ".");
            if (!SystemInfo.supportsComputeShaders)
                throw new InvalidOperationException("Compute shaders are not supported on this device.");

            contactSlots = contactCapacity;
            int tableEntries = 64;
            while (tableEntries < contactSlots * 2)
                tableEntries <<= 1;
            tableSize = tableEntries;
            int slotCount = Math.Max(1, contactSlots);
            var freePairs = new ContactPairRecord[slotCount];
            for (int s = 0; s < freePairs.Length; s++)
                freePairs[s].ident = new uint4(AvbdConst.Invalid, AvbdConst.Invalid, 0u, 0u);
            dirtyBits = new uint[Math.Max(1, (bodies.Capacity + 31) / 32)];
            records = new ConstraintRecord[capacity];

            AvbdShaderSet shaders = shaderSet != null ? shaderSet : AvbdShaderSet.Load();
            if (shaders == null)
                throw new InvalidOperationException("The AvbdShaderSet asset was not found.");
            coloring = new AvbdGraphColoring(shaders.Coloring);
            lists = new AvbdConstraintLists(shaders.Lists, bodies.Capacity, (capacity + contactSlots) * 2); // a joint or a manifold is listed for at most two bodies

            try
            {
                recordBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, capacity, ConstraintRecordStrides.Constraint);
                listBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (capacity + contactSlots) * 2, sizeof(uint));
                contactPairBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, slotCount, ContactRecordStrides.Pair);
                contactPointBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, slotCount * ContactLayout.MaxContactsPerPair, ContactRecordStrides.Point);
                pairTableBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, tableSize, sizeof(uint));
                freeListBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, slotCount, sizeof(uint));
                poolControlBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ContactLayout.PoolControlWords, sizeof(uint));
                // Written by a compute shader (a raw byte-address buffer) and read by DispatchIndirect.
                npArgsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments | GraphicsBuffer.Target.Raw,
                    ContactLayout.PoolArgsCount * AvbdIndirect.ArgsStride / sizeof(uint), sizeof(uint));
                dirtyBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, dirtyBits.Length, sizeof(uint));
                colorBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bodies.Capacity, sizeof(uint));
                statusBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, StatusWords, sizeof(uint));
            }
            catch
            {
                Dispose();
                throw;
            }

            // Rule P9: nothing is assumed to be zeroed. The contact points are the exception: a point is read only if its pair's count says it
            // exists, and the pair records (all free) are initialized here.
            recordBuffer.SetData(records, 0, 0, capacity);
            listBuffer.SetData(new uint[(capacity + contactSlots) * 2]);
            contactPairBuffer.SetData(freePairs);
            pairTableBuffer.SetData(new uint[tableSize]);
            var freeList = new uint[slotCount];
            for (int i = 0; i < contactSlots; i++)
                freeList[i] = (uint)(contactSlots - 1 - i); // the stack pops slot 0 first
            freeListBuffer.SetData(freeList);
            poolControlBuffer.SetData(new uint[] { (uint)contactSlots, 0u, 0u }); // all slots free, nothing dropped, no slot taken yet
            var emptyArgs = new uint[ContactLayout.PoolArgsCount * 3];
            for (int i = 0; i < ContactLayout.PoolArgsCount; i++)
            {
                emptyArgs[i * 3 + 1] = 1u; // 0 x 1 x 1 groups: nothing, until stage N has planned
                emptyArgs[i * 3 + 2] = 1u;
            }
            npArgsBuffer.SetData(emptyArgs);
            dirtyBuffer.SetData(dirtyBits);
            colorBuffer.SetData(new uint[bodies.Capacity]);
            ClearStatus();
        }

        // ---- state ------------------------------------------------------------------------------------

        public int Capacity => records.Length;

        /// <summary>Constraints added, counting those that are not uploaded yet.</summary>
        public int Count => recordCount + pending.Count;

        /// <summary>Add calls that failed because every slot was in use.</summary>
        public int OverflowCount { get; private set; }

        /// <summary>True if constraints were added since the last <see cref="Rebuild"/>.</summary>
        public bool HasPendingChanges => topologyDirty;

        /// <summary>
        /// True if the lists or the color buckets are out of date: constraints were added, or bodies were spawned, removed or
        /// changed in a way that decides whether the primal stage solves them (see <see cref="AvbdBodyBuffers.SolvedSetVersion"/>).
        /// </summary>
        internal bool NeedsRebuild => topologyDirty || builtSolvedSetVersion != bodies.SolvedSetVersion;

        /// <summary>True if the object owns a contact region (it was created with a contactCapacity above 0).</summary>
        public bool HasContacts => contactSlots > 0;

        /// <summary>Number of manifold slots: the pairs of bodies that can have contacts at once.</summary>
        public int ContactCapacity => contactSlots;

        /// <summary>
        /// Number of joints and springs that have fractured, as of the last list build (stage G): every step with contacts, otherwise
        /// <see cref="Rebuild"/>. Reading it waits for the GPU the first time after a build, so do not read it every frame in a hot loop.
        /// </summary>
        public int BrokenCount
        {
            get
            {
                ThrowIfDisposed();
                if (brokenCountVersion != lists.Version)
                {
                    brokenCount = lists.ReadControl().Broken;
                    brokenCountVersion = lists.Version;
                }
                return brokenCount;
            }
        }

        /// <summary>What the last list build counted: entries, fractured joints left out, and records it could not list. Waits for the GPU.</summary>
        internal ConstraintListControl ReadListControl()
        {
            ThrowIfDisposed();
            return lists.ReadControl();
        }

        /// <summary>Size of the pair hash table of stage N: a power of two, at least twice the number of slots.</summary>
        internal int PairTableSize => tableSize;

        /// <summary>Capacity of the broadphase's dynamic-dynamic bin that goes with this contact capacity.</summary>
        internal int CandidateCapacity => contactSlots * CandidatePairsPerSlot + 1024;

        /// <summary>Capacity of the broadphase's dynamic-static bin that goes with this contact capacity.</summary>
        internal int StaticCandidateCapacity => contactSlots + 1024;

        /// <summary>
        /// Number of graph colors the primal stage dispatches per iteration (each dispatch is indirect). After <see cref="Rebuild"/> it is the exact
        /// number of colors; after a step with contacts and <see cref="PersistentColoring"/> it is the cap, which is a few above the colors in use.
        /// </summary>
        public int ColorCount => colorCount;

        /// <summary>
        /// With contacts, keep the colors of the step before and recolor only the bodies that clash with a neighbor, without waiting for the GPU
        /// (the default). Off: color from scratch every step and wait for it, which is the coloring of <see cref="CpuGraphColoring"/>. Applies to
        /// <see cref="GraphColoringOrder.Hashed"/> only: <see cref="GraphColoringOrder.SlotOrder"/> promises the reference solver's coloring.
        /// </summary>
        public bool PersistentColoring { get; set; } = true;

        /// <summary>Steps with persistent colors between two colorings from scratch (see <see cref="AvbdGraphColoring.FreshInterval"/>).</summary>
        public int ColoringFreshInterval
        {
            get => coloring.FreshInterval;
            set => coloring.FreshInterval = Math.Max(0, value);
        }

        /// <summary>The statistics of the newest persistent coloring step that has come back from the GPU.</summary>
        internal ColoringSample LastColoringSample => coloring.LastSample;
        internal int ColoringSamplesSeen => coloring.SamplesSeen;
        internal int ColoringRounds => coloring.Tuner.RoundsFor(false);

        /// <summary>Takes in the coloring statistics that are in flight, waiting for them (for tests and tools).</summary>
        internal void FlushColoringStatistics() => coloring.PollStatistics(true);

        /// <summary>
        /// Which body colors first (see <see cref="GraphColoringOrder"/>). The default is <see cref="GraphColoringOrder.Hashed"/> (few rounds);
        /// <see cref="GraphColoringOrder.SlotOrder"/> gives the reference solver's coloring. Changing it recolors at the next <see cref="Rebuild"/>.
        /// </summary>
        public GraphColoringOrder ColoringOrder
        {
            get => coloringOrder;
            set
            {
                if (value != coloringOrder)
                {
                    coloringOrder = value;
                    topologyDirty = true;
                }
            }
        }

        /// <summary>Rounds the last coloring needed. Fewer is faster; see <see cref="GraphColoringOrder"/>.</summary>
        public int LastColoringRounds => coloring.LastRounds;

        // ---- buffers for the solver stages -----------------------------------------------------------------

        /// <summary>Constraints on the GPU: those added before the last <see cref="Rebuild"/>.</summary>
        internal int RecordCount => uploadedCount;

        /// <summary>The joints and springs (the contacts are in <see cref="ContactPairsBuffer"/> and <see cref="ContactPointsBuffer"/>).</summary>
        internal GraphicsBuffer Records => recordBuffer;
        internal GraphicsBuffer List => listBuffer;
        internal GraphicsBuffer ColorBodies => colorBuffer;

        /// <summary>The coloring's work buffer, which holds the start and size of every color on the GPU (AVBD_COLOR_COUNTS_OFFSET and _STARTS_OFFSET).</summary>
        /// <summary>For each entry of <see cref="List"/>, the body on the other side of the constraint (AVBD_INVALID for a world pin).</summary>
        internal GraphicsBuffer ListOther => lists.Other;

        internal GraphicsBuffer ColorWork => coloring.Work;

        /// <summary>Indirect dispatch arguments of the primal sweep, <see cref="AvbdGraphColoring.ArgsStride"/> bytes per color.</summary>
        internal GraphicsBuffer ColorArgs => coloring.Args;
        internal GraphicsBuffer Status => statusBuffer;

        /// <summary>The manifold pool's pair records, one per slot (<see cref="ContactPairRecord"/>). Never null: without contacts it holds one free slot.</summary>
        internal GraphicsBuffer ContactPairsBuffer => contactPairBuffer;

        /// <summary>The manifold pool's contact points, <see cref="ContactLayout.MaxContactsPerPair"/> per slot (<see cref="ContactPointRecord"/>).</summary>
        internal GraphicsBuffer ContactPointsBuffer => contactPointBuffer;
        internal GraphicsBuffer PairTableBuffer => pairTableBuffer;
        internal GraphicsBuffer FreeListBuffer => freeListBuffer;
        internal GraphicsBuffer PoolControlBuffer => poolControlBuffer;

        /// <summary>Indirect dispatch arguments of stage N and of the dual update of the contacts, <see cref="AvbdIndirect.ArgsStride"/> bytes per entry (<c>ContactLayout.PoolArgs*</c>).</summary>
        internal GraphicsBuffer NpArgsBuffer => npArgsBuffer;
        internal GraphicsBuffer DirtySlotsBuffer => dirtyBuffer;

        /// <summary>
        /// The start and size of every color and the indirect arguments of its sweep, read back from the GPU (this waits for the GPU; the result is
        /// kept until the next coloring). For tests and tools: the solver never reads them.
        /// </summary>
        internal ColorSchedule ReadColorSchedule()
        {
            ThrowIfDisposed();
            if (!scheduleValid)
            {
                schedule = coloring.ReadSchedule(colorCount);
                scheduleValid = true;
            }
            return schedule;
        }

        /// <summary>Start of a color in the bucket array (<see cref="ColorBodies"/>). Reads the GPU, see <see cref="ReadColorSchedule"/>.</summary>
        internal int ColorOffset(int color) => ReadColorSchedule().Starts[color];

        /// <summary>Number of bodies of a color. Reads the GPU, see <see cref="ReadColorSchedule"/>.</summary>
        internal int ColorSize(int color) => ReadColorSchedule().Counts[color];

        /// <summary>The bucket array read back: the solved body slots grouped by color. Order inside a color is unspecified.</summary>
        internal uint[] ReadColorBodies()
        {
            ThrowIfDisposed();
            var all = new uint[colorBuffer.count];
            colorBuffer.GetData(all);
            var used = new uint[ReadColorSchedule().Starts[colorCount]];
            Array.Copy(all, used, used.Length);
            return used;
        }

        // ---- adding constraints ----------------------------------------------------------------------------

        /// <summary>
        /// Adds a ball-socket joint with an optional angle lock, like upstream's Joint. Pass an invalid <paramref name="bodyA"/> to pin
        /// bodyB to the world-space point rA. Stiffness is a number above or equal to zero, or +infinity for a hard constraint.
        /// A stiffnessAng of 0 disables the angle lock. fracture is the multiplier norm above which the joint breaks (+infinity: never).
        /// Returns the constraint index, or -1 if the buffers are full.
        /// </summary>
        public int AddJoint(
            BodyHandle bodyA, BodyHandle bodyB, float3 rA, float3 rB,
            float stiffnessLin = float.PositiveInfinity, float stiffnessAng = 0f, float fracture = float.PositiveInfinity)
        {
            ThrowIfDisposed();
            if (bodyA.IsValid && !bodies.IsAlive(bodyA)) throw new ArgumentException("bodyA is stale.", nameof(bodyA));
            if (!bodies.IsAlive(bodyB)) throw new ArgumentException("bodyB is invalid or stale.", nameof(bodyB));
            if (bodyA == bodyB) throw new ArgumentException("A joint needs two different bodies.");
            RequireFinite(rA, nameof(rA));
            RequireFinite(rB, nameof(rB));
            RequireStiffness(stiffnessLin, nameof(stiffnessLin));
            RequireStiffness(stiffnessAng, nameof(stiffnessAng));
            if (!(fracture >= 0f)) throw new ArgumentException("fracture must be zero or more, or +infinity.", nameof(fracture));

            return Enqueue(new Spec
            {
                kind = ConstraintKind.Joint,
                slotA = bodyA.IsValid ? (uint)bodyA.Slot : AvbdConst.Invalid,
                slotB = (uint)bodyB.Slot,
                rA = rA,
                rB = rB,
                stiffnessLin = stiffnessLin,
                stiffnessAng = stiffnessAng,
                fractureOrRest = fracture,
            });
        }

        /// <summary>
        /// Adds a spring between a point on bodyA and a point on bodyB (both in body space), like upstream's Spring.
        /// A negative rest length means the distance between the two points at the time of <see cref="Rebuild"/>.
        /// Returns the constraint index, or -1 if the buffers are full.
        /// </summary>
        public int AddSpring(BodyHandle bodyA, BodyHandle bodyB, float3 rA, float3 rB, float stiffness, float rest = -1f)
        {
            ThrowIfDisposed();
            if (!bodies.IsAlive(bodyA)) throw new ArgumentException("bodyA is invalid or stale.", nameof(bodyA));
            if (!bodies.IsAlive(bodyB)) throw new ArgumentException("bodyB is invalid or stale.", nameof(bodyB));
            if (bodyA == bodyB) throw new ArgumentException("A spring needs two different bodies.");
            RequireFinite(rA, nameof(rA));
            RequireFinite(rB, nameof(rB));
            if (!(stiffness >= 0f) || !math.isfinite(stiffness)) throw new ArgumentException("stiffness must be finite and not negative.", nameof(stiffness));
            if (!math.isfinite(rest)) throw new ArgumentException("rest must be finite.", nameof(rest));

            return Enqueue(new Spec
            {
                kind = ConstraintKind.Spring,
                slotA = (uint)bodyA.Slot,
                slotB = (uint)bodyB.Slot,
                rA = rA,
                rB = rB,
                stiffnessLin = stiffness,
                fractureOrRest = rest,
            });
        }

        // Adds a finished record (the warm-start state included). For tests, which build records with the CPU oracle.
        internal int AddRecord(in ConstraintRecord record)
        {
            ThrowIfDisposed();
            if (pending.Count > 0)
                throw new InvalidOperationException("Rebuild before adding raw records.");
            if (recordCount >= records.Length)
            {
                OverflowCount++;
                return -1;
            }
            records[recordCount] = record;
            topologyDirty = true;
            return recordCount++;
        }

        int Enqueue(in Spec spec)
        {
            if (Count >= Capacity)
            {
                OverflowCount++;
                return -1;
            }
            pending.Add(spec);
            topologyDirty = true;
            return Count - 1;
        }

        // ---- rebuild ---------------------------------------------------------------------------------------

        /// <summary>
        /// Applies the pending body edits, completes and uploads the new constraints, and rebuilds the per-body lists, the coloring
        /// and the color buckets. It reads back body data, so call it when the constraint set changes, not every frame.
        /// </summary>
        public void Rebuild()
        {
            ThrowIfDisposed();
            bodies.ApplyEdits();

            int bodyCount = bodies.HighWaterMark;
            int capacity = bodies.Capacity;

            if (pending.Count > 0)
            {
                var shape = new BodyShapeRecord[capacity];
                var state = new BodyStateRecord[capacity];
                bodies.Shape.GetData(shape);
                bodies.State.GetData(state);

                for (int i = 0; i < pending.Count; i++)
                {
                    Spec s = pending[i];
                    records[recordCount] = s.kind == ConstraintKind.Joint
                        ? ConstraintRecordFactory.Joint(
                            s.slotA, s.slotB, s.rA, s.rB, s.stiffnessLin, s.stiffnessAng, s.fractureOrRest,
                            TorqueArm(shape, s.slotA, s.slotB))
                        : ConstraintRecordFactory.Spring(
                            s.slotA, s.slotB, s.rA, s.rB, s.stiffnessLin,
                            s.fractureOrRest >= 0f ? s.fractureOrRest : CurrentDistance(state, s));
                    recordCount++;
                }
                pending.Clear();
            }

            if (recordCount > uploadedCount)
            {
                recordBuffer.SetData(records, uploadedCount, uploadedCount, recordCount - uploadedCount);
                uploadedCount = recordCount;
            }

            // Stage G: the per-body lists, then stage C: color the graph and bucket the bodies. Both on the GPU.
            BuildLists(bodyCount);
            Recolor(bodyCount);

            topologyDirty = false;
            builtSolvedSetVersion = bodies.SolvedSetVersion;
        }

        /// <summary>
        /// Before the narrow phase: tells it which body slots were spawned or removed since the last step, so that a recycled slot does
        /// not inherit the contacts of the body that had it. Does nothing without contacts.
        /// </summary>
        internal void PrepareContacts()
        {
            ThrowIfDisposed();
            if (contactSlots == 0)
                return;

            dirtySlots.Clear();
            bodies.TakeStructuralSlots(dirtySlots);
            if (dirtySlots.Count == 0 && !dirtyUploaded)
                return;

            Array.Clear(dirtyBits, 0, dirtyBits.Length);
            foreach (int slot in dirtySlots)
                if ((uint)slot < (uint)bodies.Capacity)
                    dirtyBits[slot >> 5] |= 1u << (slot & 31);
            dirtyBuffer.SetData(dirtyBits);
            dirtyUploaded = dirtySlots.Count > 0;
        }

        /// <summary>
        /// After the narrow phase: rebuilds the lists (stage G), the coloring and the color buckets (stage C) with the contacts in them. The lists
        /// read the manifold pool and the records on the GPU, so nothing is read back for them. A fractured joint is left out here, which is how the
        /// broadphase of the next step stops excluding its pair (the reference solver removes a broken joint one step after it breaks). Call
        /// <see cref="Rebuild"/> first if constraints were added or the set of solved bodies changed.
        /// </summary>
        internal void RefreshContactTopology()
        {
            ThrowIfDisposed();
            if (contactSlots == 0)
                return;

            int bodyCount = bodies.HighWaterMark;
            BuildLists(bodyCount);
            ColorStep(bodyCount);
        }

        // Stage C of a step with contacts: persistent when it may be, from scratch (and waiting) otherwise.
        void ColorStep(int bodyCount)
        {
            if (PersistentColoring && coloringOrder == GraphColoringOrder.Hashed && coloring.HasBaseline && bodyCount <= AvbdGraphColoring.MaxPersistentBodies)
            {
                colorCount = coloring.RunPersistent(bodies, bodyCount, recordBuffer, listBuffer, lists.Other, colorBuffer, coloringOrder);
                scheduleValid = false;
            }
            else
                Recolor(bodyCount);
        }

        /// <summary>For tests: the coloring of a step with contacts, on the lists the last <see cref="Rebuild"/> built.</summary>
        internal void ColorStepForTests()
        {
            ThrowIfDisposed();
            ColorStep(bodies.HighWaterMark);
        }

        /// <summary>For tests: fixes the cap and the number of rounds of the persistent coloring (zero: tuned).</summary>
        internal void PinColoring(int cap, int rounds)
        {
            coloring.Tuner.CapOverride = cap;
            coloring.Tuner.RoundsOverride = rounds;
        }

        // Stage C: colors the graph from scratch and buckets the bodies. The number of colors comes back (this waits for the GPU); the size of each color stays on the GPU.
        void Recolor(int bodyCount)
        {
            colorCount = coloring.Run(bodies, bodyCount, recordBuffer, listBuffer, lists.Other, colorBuffer, coloringOrder);
            scheduleValid = false;
        }

        // Stage G: the lists of the per-body constraints, from the joints and springs on the GPU and the manifold pool.
        void BuildLists(int bodyCount)
        {
            lists.Run(bodies, bodyCount, recordBuffer, uploadedCount, listBuffer, contactPairBuffer, contactSlots, poolControlBuffer, npArgsBuffer);
        }

        /// <summary>Reads the manifold pool back from the GPU (this waits for the GPU): the pairs, their slots and contact counts, the free slots and the overflow count.</summary>
        internal ContactSnapshot ReadContacts()
        {
            ThrowIfDisposed();
            var pairs = new ContactPairRecord[contactPairBuffer.count];
            var control = new uint[ContactLayout.PoolControlWords];
            contactPairBuffer.GetData(pairs);
            poolControlBuffer.GetData(control);
            var counts = new uint[pairs.Length];
            var bodiesOfSlot = new uint4[pairs.Length];
            for (int s = 0; s < pairs.Length; s++)
            {
                counts[s] = pairs[s].ident.z;
                bodiesOfSlot[s] = new uint4(pairs[s].ident.x, pairs[s].ident.y, 0u, 0u);
            }
            return new ContactSnapshot(counts, bodiesOfSlot, (int)control[ContactLayout.PoolFreeTopWord], (int)control[ContactLayout.PoolOverflowWord],
                (int)control[ContactLayout.PoolHighWaterWord]);
        }

        /// <summary>
        /// Every contact of the pool in the form the CPU oracle stores (one <see cref="ConstraintRecord"/> per point), at index
        /// <c>slot * ContactLayout.MaxContactsPerPair + k</c>; the entries beyond a slot's contact count are zero. Waits for the GPU. For tests and tools.
        /// </summary>
        internal ConstraintRecord[] ReadContactRecords()
        {
            ThrowIfDisposed();
            var pairs = new ContactPairRecord[contactPairBuffer.count];
            var points = new ContactPointRecord[contactPointBuffer.count];
            contactPairBuffer.GetData(pairs);
            contactPointBuffer.GetData(points);

            var records = new ConstraintRecord[points.Length];
            for (int s = 0; s < pairs.Length; s++)
            {
                if (pairs[s].ident.x == AvbdConst.Invalid)
                    continue;
                int count = (int)Math.Min(pairs[s].ident.z, (uint)ContactLayout.MaxContactsPerPair);
                for (int k = 0; k < count; k++)
                    records[s * ContactLayout.MaxContactsPerPair + k] = ContactCodec.ToRecord(pairs[s], points[s * ContactLayout.MaxContactsPerPair + k], k);
            }
            return records;
        }

        /// <summary>One contact of <see cref="ReadContactRecords"/>. Waits for the GPU. For tests.</summary>
        internal ConstraintRecord ReadContactRecord(int slot, int k)
        {
            ThrowIfDisposed();
            var pair = new ContactPairRecord[1];
            var point = new ContactPointRecord[1];
            contactPairBuffer.GetData(pair, 0, slot, 1);
            contactPointBuffer.GetData(point, 0, slot * ContactLayout.MaxContactsPerPair + k, 1);
            return ContactCodec.ToRecord(pair[0], point[0], k);
        }

        /// <summary>
        /// Overwrites the values of one contact point and its stick bit with those of a CPU-oracle contact record (anchors, feature key, C0, penalty, multiplier,
        /// stick flag). The bodies, the count, the normal and the friction of the pair stay. For tests.
        /// </summary>
        internal void WriteContactRecord(int slot, int k, in ConstraintRecord record)
        {
            ThrowIfDisposed();
            var pair = new ContactPairRecord[1];
            contactPairBuffer.GetData(pair, 0, slot, 1);
            uint bit = 1u << k;
            pair[0].ident.w = (record.ident.z & (uint)ConstraintFlags.Stick) != 0u ? pair[0].ident.w | bit : pair[0].ident.w & ~bit;
            contactPairBuffer.SetData(pair, 0, slot, 1);
            contactPointBuffer.SetData(new[] { ContactCodec.FromRecord(record) }, 0, slot * ContactLayout.MaxContactsPerPair + k, 1);
        }

        /// <summary>
        /// The groups (x, y, z) of every indirect dispatch the last narrow phase planned, in the order of <c>ContactLayout.PoolArgs*</c>: three
        /// words per entry. Waits for the GPU. For tests and tools.
        /// </summary>
        internal uint[] ReadNpArgs()
        {
            ThrowIfDisposed();
            var words = new uint[ContactLayout.PoolArgsCount * 3];
            npArgsBuffer.GetData(words);
            return words;
        }

        static float TorqueArm(BodyShapeRecord[] shape, uint slotA, uint slotB)
        {
            float3 sizeA = slotA != AvbdConst.Invalid ? shape[slotA].sizeFriction.xyz : float3.zero;
            float3 sizeB = shape[slotB].sizeFriction.xyz;
            return math.lengthsq(sizeA + sizeB);
        }

        static float CurrentDistance(BodyStateRecord[] state, in Spec s)
        {
            BodyStateRecord a = state[s.slotA];
            BodyStateRecord b = state[s.slotB];
            float3 pA = math.rotate(new quaternion(a.rotation), s.rA) + a.position.xyz;
            float3 pB = math.rotate(new quaternion(b.rotation), s.rB) + b.position.xyz;
            return math.length(pA - pB);
        }

        // ---- status ----------------------------------------------------------------------------------------

        /// <summary>Primal solves skipped since the last <see cref="ClearStatus"/> because the 6x6 system was not positive definite.</summary>
        public int ReadSolveFailures()
        {
            ThrowIfDisposed();
            var data = new uint[StatusWords];
            statusBuffer.GetData(data);
            return (int)data[0];
        }

        public void ClearStatus()
        {
            ThrowIfDisposed();
            statusBuffer.SetData(new uint[StatusWords]);
        }

        // ---- helpers ---------------------------------------------------------------------------------------

        static void RequireFinite(float3 v, string name)
        {
            if (!math.all(math.isfinite(v)))
                throw new ArgumentException(name + " must be finite.", name);
        }

        static void RequireStiffness(float k, string name)
        {
            if (!(k >= 0f))
                throw new ArgumentException(name + " must be zero or more, or +infinity.", name);
        }

        void ThrowIfDisposed()
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(AvbdConstraintBuffers));
        }

        public void Dispose()
        {
            if (disposed)
                return;
            disposed = true;

            Release(ref recordBuffer);
            Release(ref listBuffer);
            Release(ref colorBuffer);
            Release(ref statusBuffer);
            Release(ref contactPairBuffer);
            Release(ref contactPointBuffer);
            Release(ref pairTableBuffer);
            Release(ref freeListBuffer);
            Release(ref poolControlBuffer);
            Release(ref npArgsBuffer);
            Release(ref dirtyBuffer);
            coloring?.Dispose();
            lists?.Dispose();
        }

        static void Release(ref GraphicsBuffer buffer)
        {
            buffer?.Release();
            buffer = null;
        }
    }
}
