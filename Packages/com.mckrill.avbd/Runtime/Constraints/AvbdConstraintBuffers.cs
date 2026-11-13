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
    /// calls when constraints are pending) applies the pending body edits, completes the records, builds the lists and the coloring
    /// on the CPU and uploads them. This CPU build is temporary: stages G (lists) and C (coloring) replace it on the GPU.
    ///
    /// Limits of this version: constraints cannot be removed, and removing a body that has constraints is not supported.
    /// The arrays have a fixed capacity. Methods must be called from the main thread.
    /// </summary>
    public sealed class AvbdConstraintBuffers : IDisposable
    {
        /// <summary>Largest constraint count: keeps constraint ids below the tag bit (rule P6) and the buffer below 2 GB.</summary>
        public const int MaxConstraints = 10_000_000;

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

        GraphicsBuffer recordBuffer;
        GraphicsBuffer listBuffer;
        GraphicsBuffer colorBuffer;
        GraphicsBuffer statusBuffer;

        int recordCount;   // records in the CPU array
        int uploadedCount; // records already on the GPU
        bool topologyDirty;
        int builtSolvedSetVersion;
        bool disposed;

        /// <param name="bodies">The body arrays the constraints refer to.</param>
        /// <param name="capacity">Number of constraints. Fixed for the lifetime of the object.</param>
        public AvbdConstraintBuffers(AvbdBodyBuffers bodies, int capacity)
        {
            this.bodies = bodies ?? throw new ArgumentNullException(nameof(bodies));
            if (capacity < 1 || capacity > MaxConstraints)
                throw new ArgumentOutOfRangeException(nameof(capacity), "Constraint capacity must be between 1 and " + MaxConstraints + ".");
            if (!SystemInfo.supportsComputeShaders)
                throw new InvalidOperationException("Compute shaders are not supported on this device.");

            records = new ConstraintRecord[capacity];

            try
            {
                recordBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, capacity, ConstraintRecordStrides.Constraint);
                listBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, capacity * 2, sizeof(uint));
                colorBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bodies.Capacity, sizeof(uint));
                statusBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, StatusWords, sizeof(uint));
            }
            catch
            {
                Dispose();
                throw;
            }

            // Rule P9: nothing is assumed to be zeroed.
            recordBuffer.SetData(records);
            listBuffer.SetData(new uint[capacity * 2]);
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

        /// <summary>Number of graph colors of the last rebuild: the primal stage dispatches this many times per iteration.</summary>
        public int ColorCount => topology.ColorCount;

        // ---- buffers for the solver stages -----------------------------------------------------------------

        /// <summary>Constraints on the GPU: those added before the last <see cref="Rebuild"/>.</summary>
        internal int RecordCount => uploadedCount;

        internal GraphicsBuffer Records => recordBuffer;
        internal GraphicsBuffer List => listBuffer;
        internal GraphicsBuffer ColorBodies => colorBuffer;
        internal GraphicsBuffer Status => statusBuffer;
        internal ConstraintTopology Topology => topology;

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
            var solved = new bool[bodyCount];
            const uint notSolved = (uint)BodyFlags.Free | (uint)BodyFlags.Kinematic;
            for (int s = 0; s < bodyCount; s++)
                solved[s] = (meta[s].ident.x & notSolved) == 0u && mass[s].momentMass.w > 0f;

            topology.Build(records, recordCount, solved, bodyCount);

            if (bodyCount > 0)
                bodies.Range.SetData(topology.Ranges, 0, 0, bodyCount);
            if (topology.ListCount > 0)
                listBuffer.SetData(topology.List, 0, 0, topology.ListCount);
            if (topology.ColorBodyCount > 0)
                colorBuffer.SetData(topology.ColorBodies, 0, 0, topology.ColorBodyCount);

            topologyDirty = false;
            builtSolvedSetVersion = bodies.SolvedSetVersion;
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
        }

        static void Release(ref GraphicsBuffer buffer)
        {
            buffer?.Release();
            buffer = null;
        }
    }
}
