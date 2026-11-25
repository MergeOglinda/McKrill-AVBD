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
    /// The contact kernels visit only the manifold slots below the pool's high-water mark (an indirect dispatch with the narrow phase's arguments).
    /// A manifold is one entry (the slot with the <c>AVBD_LIST_CONTACT_FLAG</c> bit), whatever its number of contact points.
    /// The order matches <see cref="ConstraintTopology"/>, the CPU oracle: contacts first by pair, then joints and springs newest first.
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

        /// <summary>Mirrors AVBD_LIST_CONTACT_FLAG: a list entry with this bit is a manifold slot, not a constraint id.</summary>
        public const uint ContactFlag = 0x80000000u;

        /// <summary>Mirrors AVBD_LIST_SORT_LANES: threads that sort the list of one body.</summary>
        public const int SortLanes = 8;

        readonly ComputeShader shader;
        readonly int clearKernel, countRecordsKernel, countContactsKernel, scanBlockKernel, scanTopKernel, scanApplyKernel, fillRecordsKernel, fillContactsKernel, sortKernel;
        readonly int[] allKernels;
        readonly int bodyCapacity;
        GraphicsBuffer work;
        GraphicsBuffer control;
        GraphicsBuffer keys;   // one per list entry, unsorted: the sort key
        GraphicsBuffer items;  // one per list entry, unsorted: the constraint id and the body on the other side
        GraphicsBuffer other;  // one per list entry, sorted: the body on the other side of the constraint
        bool disposed;

        static readonly int BodyFlagsId = Shader.PropertyToID("_BodyFlags");
        static readonly int BodyMassId = Shader.PropertyToID("_BodyMass");
        static readonly int BodyRangeId = Shader.PropertyToID("_BodyRange");
        static readonly int RecordsId = Shader.PropertyToID("_Constraints");
        static readonly int ListId = Shader.PropertyToID("_ConstraintList");
        static readonly int KeysId = Shader.PropertyToID("_ListKey");
        static readonly int ItemsId = Shader.PropertyToID("_ListItem");
        static readonly int NpControlId = Shader.PropertyToID("_NpControl");
        static readonly int SlotCapacityId = Shader.PropertyToID("_SlotCapacity");
        static readonly int OtherId = Shader.PropertyToID("_ConstraintOther");
        static readonly int WorkId = Shader.PropertyToID("_ListWork");
        static readonly int ControlId = Shader.PropertyToID("_ListControl");
        static readonly int ContactPairsId = Shader.PropertyToID("_ContactPairs");
        static readonly int BodyCountId = Shader.PropertyToID("_BodyCount");
        static readonly int WorkStrideId = Shader.PropertyToID("_WorkStride");

        /// <summary>Number of builds so far. Anything cached from <see cref="ReadControl"/> is out of date when this changes.</summary>
        public int Version { get; private set; }

        /// <summary>For each list entry, the body on the other side of the constraint (AVBD_INVALID for a world pin); the same layout as the list.</summary>
        public GraphicsBuffer Other => other;

        /// <param name="listCapacity">Entries of the list buffer that <see cref="Run"/> will fill.</param>
        public AvbdConstraintLists(ComputeShader shader, int bodyCapacity, int listCapacity)
        {
            this.shader = shader != null ? shader : throw new InvalidOperationException("The AvbdShaderSet has no lists shader.");
            if (bodyCapacity > ScanGroup * ScanGroup)
                throw new ArgumentOutOfRangeException(nameof(bodyCapacity), "The list scan handles at most " + ScanGroup * ScanGroup + " bodies.");
            this.bodyCapacity = bodyCapacity;
            clearKernel = shader.FindKernel("ListClear");
            countRecordsKernel = shader.FindKernel("ListCountRecords");
            countContactsKernel = shader.FindKernel("ListCountContacts");
            scanBlockKernel = shader.FindKernel("ListScanBlock");
            scanTopKernel = shader.FindKernel("ListScanTop");
            scanApplyKernel = shader.FindKernel("ListScanApply");
            fillRecordsKernel = shader.FindKernel("ListFillRecords");
            fillContactsKernel = shader.FindKernel("ListFillContacts");
            sortKernel = shader.FindKernel("ListSort");
            allKernels = new[]
            {
                clearKernel, countRecordsKernel, countContactsKernel, scanBlockKernel, scanTopKernel, scanApplyKernel,
                fillRecordsKernel, fillContactsKernel, sortKernel,
            };

            // Counts, cursors, contacts per body and the sums of the scan blocks (see AvbdLists.compute).
            int blocks = (bodyCapacity + ScanGroup - 1) / ScanGroup;
            work = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Math.Max(1, bodyCapacity * 3 + blocks), sizeof(uint));
            control = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ControlWords, sizeof(uint));
            keys = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Math.Max(1, listCapacity), 2 * sizeof(uint));
            items = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Math.Max(1, listCapacity), 2 * sizeof(uint));
            other = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Math.Max(1, listCapacity), sizeof(uint));
            work.SetData(new uint[work.count]);
            control.SetData(new uint[ControlWords]);
        }

        /// <summary>
        /// Builds the lists of the solved bodies among slots [0, bodyCount) and writes their offsets and counts into the body Range array.
        /// </summary>
        /// <param name="records">Joint and spring records, [0, recordCount).</param>
        /// <param name="list">Receives the entries: constraint ids, and manifold slots with <see cref="ContactFlag"/>. It must hold as many entries as the listCapacity of the constructor.</param>
        /// <param name="contactPairs">The pair records of the manifold slots (the bodies, AVBD_INVALID = free, and the contact count), or null without contacts.</param>
        /// <param name="slotCount">Number of manifold slots.</param>
        /// <param name="poolControl">The pool's control words (the high-water mark), or null without contacts.</param>
        /// <param name="poolArgs">The narrow phase's indirect arguments; its retire entry is one thread per slot below the high-water mark.</param>
        public void Run(AvbdBodyBuffers bodies, int bodyCount, GraphicsBuffer records, int recordCount, GraphicsBuffer list,
            GraphicsBuffer contactPairs, int slotCount, GraphicsBuffer poolControl, GraphicsBuffer poolArgs)
        {
            if (disposed) throw new ObjectDisposedException(nameof(AvbdConstraintLists));
            if (bodyCount > bodyCapacity) throw new ArgumentOutOfRangeException(nameof(bodyCount));
            Version++;
            if (bodyCount <= 0)
                return;

            bool contacts = slotCount > 0 && contactPairs != null && poolControl != null && poolArgs != null;
            foreach (int kernel in allKernels)
            {
                shader.SetBuffer(kernel, BodyFlagsId, bodies.Flags);
                shader.SetBuffer(kernel, BodyMassId, bodies.Mass);
                shader.SetBuffer(kernel, BodyRangeId, bodies.Range);
                shader.SetBuffer(kernel, RecordsId, records);
                shader.SetBuffer(kernel, ListId, list);
                shader.SetBuffer(kernel, KeysId, keys);
                shader.SetBuffer(kernel, ItemsId, items);
                shader.SetBuffer(kernel, OtherId, other);
                shader.SetBuffer(kernel, WorkId, work);
                shader.SetBuffer(kernel, ControlId, control);
                if (contacts)
                {
                    shader.SetBuffer(kernel, ContactPairsId, contactPairs);
                    shader.SetBuffer(kernel, NpControlId, poolControl);
                }
            }
            shader.SetInt(BodyCountId, bodyCount);
            shader.SetInt(WorkStrideId, bodyCapacity);
            shader.SetInt(SlotCapacityId, contacts ? slotCount : 0);

            uint retireArgs = (uint)(ContactLayout.PoolArgsRetire * AvbdIndirect.ArgsStride);

            AvbdDispatch.Linear(shader, clearKernel, bodyCount);
            AvbdDispatch.Linear(shader, countRecordsKernel, recordCount);
            if (contacts)
                shader.DispatchIndirect(countContactsKernel, poolArgs, retireArgs);

            // Scan of the counts: inside blocks of ScanGroup bodies, then over the blocks, then the offsets.
            shader.Dispatch(scanBlockKernel, (bodyCount + ScanGroup - 1) / ScanGroup, 1, 1);
            shader.Dispatch(scanTopKernel, 1, 1, 1);
            AvbdDispatch.Linear(shader, scanApplyKernel, bodyCount);

            AvbdDispatch.Linear(shader, fillRecordsKernel, recordCount);
            if (contacts)
                shader.DispatchIndirect(fillContactsKernel, poolArgs, retireArgs);
            AvbdDispatch.Linear(shader, sortKernel, bodyCount * SortLanes);
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
            items?.Release();
            items = null;
            other?.Release();
            other = null;
        }
    }
}
