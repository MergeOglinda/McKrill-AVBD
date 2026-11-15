using System;
using UnityEngine;

namespace McKrill.Avbd
{
    /// <summary>What the last list build counted (see <see cref="AvbdConstraintLists.ReadControl"/>).</summary>
    internal readonly struct ConstraintListControl
    {
        /// <summary>Entries in the list.</summary>
        public readonly int Total;

        /// <summary>Joints and springs left out because they are fractured.</summary>
        public readonly int Broken;

        /// <summary>Records or manifolds left out because they name a body slot that is not in use, or report too many contacts. 0 in a healthy scene.</summary>
        public readonly int Invalid;

        public ConstraintListControl(int total, int broken, int invalid)
        {
            Total = total;
            Broken = broken;
            Invalid = invalid;
        }
    }

    /// <summary>
    /// Stage G: builds the constraint list of every solved body on the GPU (<c>AvbdLists.compute</c>), with no readback: it counts the
    /// constraints of each body, scans the counts into offsets (written to the body Range array), fills the lists and sorts each one.
    /// The order matches <see cref="ConstraintTopology"/>, the CPU oracle: contacts first by pair and k, then joints and springs newest first.
    /// A fractured joint or spring is left out.
    /// </summary>
    internal sealed class AvbdConstraintLists : IDisposable
    {
        // Mirror AVBD_LIST_*.
        public const int ControlTotal = 0;
        public const int ControlBroken = 1;
        public const int ControlInvalid = 2;
        public const int ControlWords = 4;
        public const int ScanGroup = 1024;

        readonly ComputeShader shader;
        readonly int beginKernel, clearKernel, countRecordsKernel, countContactsKernel, scanKernel, fillRecordsKernel, fillContactsKernel, sortKernel;
        readonly int[] allKernels;
        readonly int bodyCapacity;
        GraphicsBuffer work;
        GraphicsBuffer control;
        GraphicsBuffer keys;   // one per list entry: the sort key
        GraphicsBuffer other;  // one per list entry: the body on the other side of the constraint
        bool disposed;

        static readonly int BodyMetaId = Shader.PropertyToID("_BodyMeta");
        static readonly int BodyMassId = Shader.PropertyToID("_BodyMass");
        static readonly int BodyRangeId = Shader.PropertyToID("_BodyRange");
        static readonly int RecordsId = Shader.PropertyToID("_Constraints");
        static readonly int ListId = Shader.PropertyToID("_ConstraintList");
        static readonly int KeysId = Shader.PropertyToID("_ListKey");
        static readonly int OtherId = Shader.PropertyToID("_ConstraintOther");
        static readonly int WorkId = Shader.PropertyToID("_ListWork");
        static readonly int ControlId = Shader.PropertyToID("_ListControl");
        static readonly int PairCountsId = Shader.PropertyToID("_PairCounts");
        static readonly int SlotBodiesId = Shader.PropertyToID("_SlotBodies");
        static readonly int BodyCountId = Shader.PropertyToID("_BodyCount");
        static readonly int ContactBaseId = Shader.PropertyToID("_ContactBase");
        static readonly int WorkStrideId = Shader.PropertyToID("_WorkStride");

        /// <summary>Number of builds so far. Anything cached from <see cref="ReadControl"/> is out of date when this changes.</summary>
        public int Version { get; private set; }

        /// <summary>For each list entry, the body on the other side of the constraint (AVBD_INVALID for a world pin); the same layout as the list.</summary>
        public GraphicsBuffer Other => other;

        /// <param name="listCapacity">Entries of the list buffer that <see cref="Run"/> will fill.</param>
        public AvbdConstraintLists(ComputeShader shader, int bodyCapacity, int listCapacity)
        {
            this.shader = shader != null ? shader : throw new InvalidOperationException("The AvbdShaderSet has no lists shader.");
            this.bodyCapacity = bodyCapacity;
            beginKernel = shader.FindKernel("ListBegin");
            clearKernel = shader.FindKernel("ListClear");
            countRecordsKernel = shader.FindKernel("ListCountRecords");
            countContactsKernel = shader.FindKernel("ListCountContacts");
            scanKernel = shader.FindKernel("ListScan");
            fillRecordsKernel = shader.FindKernel("ListFillRecords");
            fillContactsKernel = shader.FindKernel("ListFillContacts");
            sortKernel = shader.FindKernel("ListSort");
            allKernels = new[] { beginKernel, clearKernel, countRecordsKernel, countContactsKernel, scanKernel, fillRecordsKernel, fillContactsKernel, sortKernel };

            work = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Math.Max(1, bodyCapacity * 2), sizeof(uint));
            control = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ControlWords, sizeof(uint));
            keys = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Math.Max(1, listCapacity), 2 * sizeof(uint));
            other = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Math.Max(1, listCapacity), sizeof(uint));
            work.SetData(new uint[work.count]);
            control.SetData(new uint[ControlWords]);
        }

        /// <summary>
        /// Builds the lists of the solved bodies among slots [0, bodyCount) and writes their offsets and counts into the body Range array.
        /// </summary>
        /// <param name="records">Constraint records. The joints and springs are [0, recordCount); the contacts are at contactBase and above.</param>
        /// <param name="list">Receives the constraint ids. It must hold twice the number of records (joints, springs and contact records), and as many entries as the listCapacity of the constructor.</param>
        /// <param name="pairCounts">Contacts per manifold slot, or null without contacts.</param>
        /// <param name="slotBodies">The pair of bodies of each manifold slot (AVBD_INVALID = free).</param>
        /// <param name="slotCount">Number of manifold slots.</param>
        public void Run(AvbdBodyBuffers bodies, int bodyCount, GraphicsBuffer records, int recordCount, GraphicsBuffer list,
            GraphicsBuffer pairCounts, GraphicsBuffer slotBodies, int slotCount, int contactBase)
        {
            if (disposed) throw new ObjectDisposedException(nameof(AvbdConstraintLists));
            if (bodyCount > bodyCapacity) throw new ArgumentOutOfRangeException(nameof(bodyCount));
            Version++;
            if (bodyCount <= 0)
                return;

            bool contacts = slotCount > 0 && pairCounts != null && slotBodies != null;
            foreach (int kernel in allKernels)
            {
                shader.SetBuffer(kernel, BodyMetaId, bodies.Meta);
                shader.SetBuffer(kernel, BodyMassId, bodies.Mass);
                shader.SetBuffer(kernel, BodyRangeId, bodies.Range);
                shader.SetBuffer(kernel, RecordsId, records);
                shader.SetBuffer(kernel, ListId, list);
                shader.SetBuffer(kernel, KeysId, keys);
                shader.SetBuffer(kernel, OtherId, other);
                shader.SetBuffer(kernel, WorkId, work);
                shader.SetBuffer(kernel, ControlId, control);
                if (contacts)
                {
                    shader.SetBuffer(kernel, PairCountsId, pairCounts);
                    shader.SetBuffer(kernel, SlotBodiesId, slotBodies);
                }
            }
            shader.SetInt(BodyCountId, bodyCount);
            shader.SetInt(ContactBaseId, contactBase);
            shader.SetInt(WorkStrideId, bodyCapacity);

            shader.Dispatch(beginKernel, 1, 1, 1);
            AvbdDispatch.Linear(shader, clearKernel, bodyCount);
            AvbdDispatch.Linear(shader, countRecordsKernel, recordCount);
            if (contacts)
                AvbdDispatch.Linear(shader, countContactsKernel, slotCount);
            shader.Dispatch(scanKernel, 1, 1, 1);
            AvbdDispatch.Linear(shader, fillRecordsKernel, recordCount);
            if (contacts)
                AvbdDispatch.Linear(shader, fillContactsKernel, slotCount);
            AvbdDispatch.Linear(shader, sortKernel, bodyCount);
        }

        /// <summary>Reads the counters of the last build back (this waits for the GPU).</summary>
        public ConstraintListControl ReadControl()
        {
            if (disposed) throw new ObjectDisposedException(nameof(AvbdConstraintLists));
            var words = new uint[ControlWords];
            control.GetData(words);
            return new ConstraintListControl((int)words[ControlTotal], (int)words[ControlBroken], (int)words[ControlInvalid]);
        }

        public void Dispose()
        {
            if (disposed)
                return;
            disposed = true;
            work?.Release();
            work = null;
            control?.Release();
            control = null;
            keys?.Release();
            keys = null;
            other?.Release();
            other = null;
        }
    }
}
