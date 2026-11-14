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
    /// calls when constraints are pending) applies the pending body edits, completes the records, builds the per-body lists on the CPU
    /// and uploads them, then colors the graph and buckets the bodies on the GPU (stage C). The CPU list build is temporary: stage G
    /// replaces it on the GPU.
    ///
    /// Contacts without a broadphase: with a nonzero <c>contactBodies</c> the object also owns the contact region of the record array and the
    /// per-pair arrays of stage N (see <see cref="ContactLayout"/>). Every pair of the first <c>contactBodies</c> body slots is a candidate,
    /// so memory and narrow-phase work grow with the square of the number of bodies. After the narrow phase, <see cref="RefreshContactTopology"/>
    /// reads the contact count of every pair back and rebuilds the lists (on the CPU) and the coloring (stage C, on the GPU), because they change every step. That readback is
    /// temporary, like the CPU build of the lists: stages G and C and a broadphase replace it.
    ///
    /// Limits of this version: constraints cannot be removed, and removing a body that has constraints is not supported.
    /// The arrays have a fixed capacity. Methods must be called from the main thread.
    /// </summary>
    public sealed class AvbdConstraintBuffers : IDisposable
    {
        /// <summary>Largest constraint count: keeps constraint ids below the tag bit (rule P6) and the buffer below 2 GB.</summary>
        public const int MaxConstraints = 10_000_000;

        /// <summary>Most body slots that can collide. The contact region grows with the square of it: 512 slots take 167 MB.</summary>
        public const int MaxContactBodies = 512;

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
        readonly ConstraintTopology topology = new ConstraintTopology();
        readonly AvbdGraphColoring coloring;
        int[] colorOffsets = new int[1];
        int colorCount;
        GraphColoringOrder coloringOrder = GraphColoringOrder.SlotOrder;

        GraphicsBuffer recordBuffer;
        GraphicsBuffer listBuffer;
        GraphicsBuffer colorBuffer;
        GraphicsBuffer statusBuffer;
        GraphicsBuffer pairCountBuffer;    // contacts per pair, written by stage N
        GraphicsBuffer pairExcludedBuffer; // bit per pair: joined by a joint or a spring
        GraphicsBuffer dirtyBuffer;        // bit per body slot: spawned or removed since the last narrow phase

        readonly int contactBodies;
        readonly int contactPairs;
        readonly uint[] pairCounts;     // last read back
        readonly uint[] pairExcluded;
        readonly uint[] dirtyBits;
        readonly HashSet<int> dirtySlots = new HashSet<int>();
        bool dirtyUploaded;
        bool[] solved = new bool[0];

        int recordCount;   // records in the CPU array
        int uploadedCount; // records already on the GPU
        bool topologyDirty;
        int builtSolvedSetVersion;
        int excludedRecords; // joints and springs already in the exclusion bits
        bool disposed;

        /// <param name="bodies">The body arrays the constraints refer to.</param>
        /// <param name="capacity">Number of joints and springs. Fixed for the lifetime of the object.</param>
        /// <param name="shaderSet">The compute shaders; null loads the package's shader set.</param>
        /// <param name="contactBodies">Number of body slots (from slot 0) that collide with each other, or 0 for no contacts.
        /// At most <see cref="MaxContactBodies"/>. Bodies in higher slots do not collide.</param>
        public AvbdConstraintBuffers(AvbdBodyBuffers bodies, int capacity, AvbdShaderSet shaderSet = null, int contactBodies = 0)
        {
            this.bodies = bodies ?? throw new ArgumentNullException(nameof(bodies));
            if (capacity < 1 || capacity > MaxConstraints)
                throw new ArgumentOutOfRangeException(nameof(capacity), "Constraint capacity must be between 1 and " + MaxConstraints + ".");
            if (contactBodies != 0 && (contactBodies < 2 || contactBodies > MaxContactBodies || contactBodies > bodies.Capacity))
                throw new ArgumentOutOfRangeException(nameof(contactBodies), "contactBodies must be 0, or between 2 and " + MaxContactBodies + " and not more than the body capacity.");
            if (!SystemInfo.supportsComputeShaders)
                throw new InvalidOperationException("Compute shaders are not supported on this device.");

            this.contactBodies = contactBodies;
            contactPairs = (int)ContactLayout.PairCount(contactBodies);
            int contactRecords = contactPairs * ContactLayout.MaxContactsPerPair;
            pairCounts = new uint[Math.Max(1, contactPairs)];
            pairExcluded = new uint[Math.Max(1, (contactPairs + 31) / 32)];
            dirtyBits = new uint[Math.Max(1, (contactBodies + 31) / 32)];
            records = new ConstraintRecord[capacity];

            AvbdShaderSet shaders = shaderSet != null ? shaderSet : AvbdShaderSet.Load();
            if (shaders == null)
                throw new InvalidOperationException("The AvbdShaderSet asset was not found.");
            coloring = new AvbdGraphColoring(shaders.Coloring);

            try
            {
                recordBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, capacity + contactRecords, ConstraintRecordStrides.Constraint);
                listBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (capacity + contactRecords) * 2, sizeof(uint));
                pairCountBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, pairCounts.Length, sizeof(uint));
                pairExcludedBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, pairExcluded.Length, sizeof(uint));
                dirtyBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, dirtyBits.Length, sizeof(uint));
                colorBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bodies.Capacity, sizeof(uint));
                statusBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, StatusWords, sizeof(uint));
            }
            catch
            {
                Dispose();
                throw;
            }

            // Rule P9: nothing is assumed to be zeroed. The contact region of the records is the exception: a contact record is read only
            // if the pair's count says it exists, and the counts are initialized here.
            recordBuffer.SetData(records, 0, 0, capacity);
            listBuffer.SetData(new uint[(capacity + contactRecords) * 2]);
            pairCountBuffer.SetData(pairCounts);
            pairExcludedBuffer.SetData(pairExcluded);
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

        /// <summary>True if the object owns a contact region (it was created with contactBodies above 0).</summary>
        public bool HasContacts => contactBodies > 0;

        /// <summary>Number of body slots, from slot 0, that collide with each other.</summary>
        public int ContactBodies => contactBodies;

        /// <summary>Number of contacts found by the narrow phase in the last step (0 before the first step, or without contacts).</summary>
        public int ContactCount { get; private set; }

        /// <summary>Contacts per pair as of the last <see cref="RefreshContactTopology"/>, indexed like <see cref="ContactLayout.PairIndex"/>.</summary>
        internal uint[] PairCounts => pairCounts;

        /// <summary>Number of body slots the narrow phase covers now: the used slots, up to <see cref="ContactBodies"/>.</summary>
        internal int ContactBodiesInUse => Math.Min(bodies.HighWaterMark, contactBodies);

        /// <summary>Number of pairs the narrow phase covers now.</summary>
        internal int ContactPairsInUse => (int)ContactLayout.PairCount(ContactBodiesInUse);

        /// <summary>Index of the first contact record in <see cref="Records"/>: the capacity of the joints and springs.</summary>
        internal int ContactBase => records.Length;

        /// <summary>Number of graph colors of the last rebuild: the primal stage dispatches this many times per iteration.</summary>
        public int ColorCount => colorCount;

        /// <summary>
        /// Which body colors first (see <see cref="GraphColoringOrder"/>). The default, <see cref="GraphColoringOrder.SlotOrder"/>, gives the
        /// reference solver's coloring. Changing it recolors at the next <see cref="Rebuild"/>.
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

        internal GraphicsBuffer Records => recordBuffer;
        internal GraphicsBuffer List => listBuffer;
        internal GraphicsBuffer ColorBodies => colorBuffer;
        internal GraphicsBuffer Status => statusBuffer;
        internal GraphicsBuffer PairCountsBuffer => pairCountBuffer;
        internal GraphicsBuffer PairExcludedBuffer => pairExcludedBuffer;
        internal GraphicsBuffer DirtySlotsBuffer => dirtyBuffer;
        internal ConstraintTopology Topology => topology; // the lists; the coloring is on the GPU (see ColorCount, ColorOffset)

        /// <summary>Start of a color in the bucket array (<see cref="ColorBodies"/>).</summary>
        internal int ColorOffset(int color) => colorOffsets[color];

        /// <summary>Number of bodies of a color.</summary>
        internal int ColorSize(int color) => colorOffsets[color + 1] - colorOffsets[color];

        /// <summary>The bucket array read back: the solved body slots grouped by color. Order inside a color is unspecified.</summary>
        internal uint[] ReadColorBodies()
        {
            ThrowIfDisposed();
            var all = new uint[colorBuffer.count];
            colorBuffer.GetData(all);
            var used = new uint[colorOffsets[colorCount]];
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
                    records[recordCount++] = s.kind == ConstraintKind.Joint
                        ? ConstraintRecordFactory.Joint(
                            s.slotA, s.slotB, s.rA, s.rB, s.stiffnessLin, s.stiffnessAng, s.fractureOrRest,
                            TorqueArm(shape, s.slotA, s.slotB))
                        : ConstraintRecordFactory.Spring(
                            s.slotA, s.slotB, s.rA, s.rB, s.stiffnessLin,
                            s.fractureOrRest >= 0f ? s.fractureOrRest : CurrentDistance(state, s));
                }
                pending.Clear();
            }

            if (recordCount > uploadedCount)
            {
                recordBuffer.SetData(records, uploadedCount, uploadedCount, recordCount - uploadedCount);
                uploadedCount = recordCount;
            }

            // Which bodies the primal stage solves.
            var meta = new BodyMetaRecord[capacity];
            var mass = new BodyMassRecord[capacity];
            bodies.Meta.GetData(meta);
            bodies.Mass.GetData(mass);
            if (solved.Length < bodyCount)
                solved = new bool[bodyCount];
            const uint notSolved = (uint)BodyFlags.Free | (uint)BodyFlags.Kinematic;
            for (int s = 0; s < bodyCount; s++)
                solved[s] = (meta[s].ident.x & notSolved) == 0u && mass[s].momentMass.w > 0f;

            if (contactBodies > 0 && recordCount > excludedRecords)
                UploadExcludedPairs();

            topology.Build(records, recordCount, solved, bodyCount, false, contactBodies > 0 ? pairCounts : null, ContactBodiesInUse, ContactBase);

            if (bodyCount > 0)
                bodies.Range.SetData(topology.Ranges, 0, 0, bodyCount);
            if (topology.ListCount > 0)
                listBuffer.SetData(topology.List, 0, 0, topology.ListCount);

            // Stage C: color the graph and bucket the bodies, on the GPU.
            colorCount = coloring.Run(bodies, bodyCount, recordBuffer, listBuffer, colorBuffer, coloringOrder, out colorOffsets);

            topologyDirty = false;
            builtSolvedSetVersion = bodies.SolvedSetVersion;
        }

        // Sets the exclusion bit of the pair of every joint and spring: like the reference solver, no manifold is created between bodies
        // that a force already joins. (Ignore-collision pairs are not supported yet.)
        void UploadExcludedPairs()
        {
            Array.Clear(pairExcluded, 0, pairExcluded.Length);
            for (int i = 0; i < recordCount; i++)
            {
                uint a = records[i].ident.x;
                uint b = records[i].ident.y;
                if (a == AvbdConst.Invalid || a >= (uint)contactBodies || b >= (uint)contactBodies)
                    continue;
                long pair = ContactLayout.PairIndex((int)a, (int)b);
                pairExcluded[pair >> 5] |= 1u << (int)(pair & 31);
            }
            pairExcludedBuffer.SetData(pairExcluded);
            excludedRecords = recordCount;
        }

        /// <summary>
        /// Before the narrow phase: tells it which body slots were spawned or removed since the last step, so that a recycled slot does
        /// not inherit the contacts of the body that had it. Does nothing without contacts.
        /// </summary>
        internal void PrepareContacts()
        {
            ThrowIfDisposed();
            if (contactBodies == 0)
                return;

            dirtySlots.Clear();
            bodies.TakeStructuralSlots(dirtySlots);
            if (dirtySlots.Count == 0 && !dirtyUploaded)
                return;

            Array.Clear(dirtyBits, 0, dirtyBits.Length);
            foreach (int slot in dirtySlots)
                if (slot < contactBodies)
                    dirtyBits[slot >> 5] |= 1u << (slot & 31);
            dirtyBuffer.SetData(dirtyBits);
            dirtyUploaded = dirtySlots.Count > 0;
        }

        /// <summary>
        /// After the narrow phase: reads the contact count of every pair back and rebuilds the lists, the coloring and the color buckets
        /// with the contacts in them. Stalls until the GPU has finished the narrow phase. Call <see cref="Rebuild"/> first if constraints
        /// were added or the set of solved bodies changed.
        /// </summary>
        internal void RefreshContactTopology()
        {
            ThrowIfDisposed();
            if (contactBodies == 0)
                return;

            int bodyCount = bodies.HighWaterMark;
            int inUse = ContactBodiesInUse;
            int pairs = (int)ContactLayout.PairCount(inUse);
            if (pairs > 0)
                pairCountBuffer.GetData(pairCounts, 0, 0, pairs);

            int total = 0;
            for (int p = 0; p < pairs; p++)
                total += (int)pairCounts[p];
            ContactCount = total;

            topology.Build(records, recordCount, solved, bodyCount, false, pairCounts, inUse, ContactBase);

            if (bodyCount > 0)
                bodies.Range.SetData(topology.Ranges, 0, 0, bodyCount);
            if (topology.ListCount > listBuffer.count)
                throw new InvalidOperationException("The constraint list needs " + topology.ListCount + " entries; the buffer has " + listBuffer.count + ".");
            if (topology.ListCount > 0)
                listBuffer.SetData(topology.List, 0, 0, topology.ListCount);

            // Stage C on the GPU colors the bodies with the contacts in the graph.
            colorCount = coloring.Run(bodies, bodyCount, recordBuffer, listBuffer, colorBuffer, coloringOrder, out colorOffsets);
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
            Release(ref pairCountBuffer);
            Release(ref pairExcludedBuffer);
            Release(ref dirtyBuffer);
            coloring?.Dispose();
        }

        static void Release(ref GraphicsBuffer buffer)
        {
            buffer?.Release();
            buffer = null;
        }
    }
}
