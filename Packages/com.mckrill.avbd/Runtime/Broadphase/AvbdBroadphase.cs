using System;
using UnityEngine;

namespace McKrill.Avbd
{
    /// <summary>
    /// Stage B: finds the pairs of bodies whose bounding spheres overlap (<c>AvbdBroadphase.compute</c>), the candidates of the
    /// narrowphase. See the shader header for what a pair is and which pairs are left out.
    ///
    /// The grid is a counting sort of the bodies by hashed cell (count, scan, scatter into packed 32 B records), so the bodies of a cell are
    /// contiguous in memory and the pair kernel reads them as runs.
    ///
    /// Solved bodies and the static bodies that fit a grid cell share one grid, so the cost of a solved body does not grow with the total
    /// number of static bodies; static bodies larger than every solved body (a ground slab) are tested against every solved body.
    ///
    /// The pair list has two bins in one buffer: dynamic-dynamic pairs first, then dynamic-static pairs. Each bin has a capacity; pairs
    /// beyond it are counted but not stored, see <see cref="DynamicPairCount"/> and <see cref="Overflowed"/>.
    ///
    /// Everything stays on the GPU. The host only reads the pair counts back when asked (<see cref="ReadCounts"/>); the narrowphase reads them
    /// from the control buffer by itself and is dispatched indirectly.
    /// Pairs are recomputed from scratch on every <see cref="Run"/>; there is no temporal coherence, and the order of the pairs inside a
    /// bin is not deterministic.
    /// </summary>
    internal sealed class AvbdBroadphase : IDisposable
    {
        // Mirror AVBD_BP_*.
        public const int ControlWords = 4;
        public const int MaxRadiusWord = 0;
        public const int StaticCountWord = 1;
        public const int DynamicPairsWord = 2;
        public const int StaticPairsWord = 3;
        public const float CellMargin = 1.01f;

        /// <summary>Mirrors AVBD_BP_STATIC_FLAG: the bit of the grid cell word that marks a static body.</summary>
        public const uint StaticCellFlag = 0x80000000u;

        /// <summary>Mirrors AVBD_BP_SCAN_GROUP: cells per group of the scan of the cell counts.</summary>
        public const int ScanGroup = 1024;

        /// <summary>Bytes of one <see cref="PairRecord"/>.</summary>
        public const int PairStride = 16;

        const int MinGridCells = 64;

        // The scan of the cell counts handles at most ScanGroup * ScanGroup cells (a block scan, then one group over the block sums).
        const int MaxGridCells = ScanGroup * ScanGroup;

        // Bytes of one packed body of the grid (SortedBody in the shader).
        const int SortedBodyStride = 32;

        readonly ComputeShader shader;
        readonly int resetKernel, clearGridKernel, boundsKernel, cellKernel, scanBlockKernel, scanTopKernel, scanApplyKernel, scatterKernel, pairsDynamicKernel, pairsStaticKernel;

        readonly int bodyCapacity;
        readonly int gridCells;
        GraphicsBuffer control;
        GraphicsBuffer cellStart;
        GraphicsBuffer blockSums;
        GraphicsBuffer bodyRank;
        GraphicsBuffer sorted;
        GraphicsBuffer bodyCell;
        GraphicsBuffer staticList;
        GraphicsBuffer pairs;
        bool disposed;

        GraphicsBuffer emptyRecords; // bound in place of the constraint buffers when there are none; never read
        GraphicsBuffer emptyList;
        uint dynamicPairs, staticPairs;

        static readonly int BodyStateId = Shader.PropertyToID("_BodyState");
        static readonly int BodyMassId = Shader.PropertyToID("_BodyMass");
        static readonly int BodyShapeId = Shader.PropertyToID("_BodyShape");
        static readonly int BodyMetaId = Shader.PropertyToID("_BodyMeta");
        static readonly int BodyRangeId = Shader.PropertyToID("_BodyRange");
        static readonly int RecordsId = Shader.PropertyToID("_Constraints");
        static readonly int ListId = Shader.PropertyToID("_ConstraintList");
        static readonly int ControlId = Shader.PropertyToID("_BpControl");
        static readonly int CellStartId = Shader.PropertyToID("_CellStart");
        static readonly int BlockSumsId = Shader.PropertyToID("_BlockSums");
        static readonly int BodyRankId = Shader.PropertyToID("_BodyRank");
        static readonly int SortedId = Shader.PropertyToID("_Sorted");
        static readonly int BodyCellId = Shader.PropertyToID("_BodyCell");
        static readonly int StaticListId = Shader.PropertyToID("_StaticList");
        static readonly int PairsId = Shader.PropertyToID("_Pairs");
        static readonly int GridMaskId = Shader.PropertyToID("_GridMask");
        static readonly int DynamicCapacityId = Shader.PropertyToID("_DynamicPairCapacity");
        static readonly int StaticCapacityId = Shader.PropertyToID("_StaticPairCapacity");

        /// <param name="bodyCapacity">Slot capacity of the body buffers this broadphase serves.</param>
        /// <param name="dynamicPairCapacity">Pairs the dynamic-dynamic bin holds.</param>
        /// <param name="staticPairCapacity">Pairs the dynamic-static bin holds.</param>
        public AvbdBroadphase(ComputeShader shader, int bodyCapacity, int dynamicPairCapacity, int staticPairCapacity)
        {
            this.shader = shader != null ? shader : throw new InvalidOperationException("The AvbdShaderSet has no broadphase shader.");
            if (bodyCapacity < 1) throw new ArgumentOutOfRangeException(nameof(bodyCapacity));
            if (dynamicPairCapacity < 1) throw new ArgumentOutOfRangeException(nameof(dynamicPairCapacity));
            if (staticPairCapacity < 1) throw new ArgumentOutOfRangeException(nameof(staticPairCapacity));
            if ((long)dynamicPairCapacity + staticPairCapacity > 100_000_000)
                throw new ArgumentOutOfRangeException(nameof(dynamicPairCapacity), "The pair buffer would be larger than 1.6 GB.");

            resetKernel = shader.FindKernel("BpReset");
            clearGridKernel = shader.FindKernel("BpClearGrid");
            boundsKernel = shader.FindKernel("BpBounds");
            cellKernel = shader.FindKernel("BpCell");
            scanBlockKernel = shader.FindKernel("BpScanBlock");
            scanTopKernel = shader.FindKernel("BpScanTop");
            scanApplyKernel = shader.FindKernel("BpScanApply");
            scatterKernel = shader.FindKernel("BpScatter");
            pairsDynamicKernel = shader.FindKernel("BpPairsDynamic");
            pairsStaticKernel = shader.FindKernel("BpPairsStatic");

            this.bodyCapacity = bodyCapacity;
            DynamicPairCapacity = dynamicPairCapacity;
            StaticPairCapacity = staticPairCapacity;

            // A power of two, at least twice the number of bodies, so that a cell holds few bodies on average (and at most what the scan handles).
            int cells = MinGridCells;
            while (cells < bodyCapacity * 2 && cells < MaxGridCells)
                cells <<= 1;
            gridCells = cells;

            control = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ControlWords, sizeof(uint));
            cellStart = new GraphicsBuffer(GraphicsBuffer.Target.Structured, gridCells + 1, sizeof(uint));
            blockSums = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (gridCells + ScanGroup - 1) / ScanGroup + 1, sizeof(uint));
            bodyRank = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bodyCapacity, sizeof(uint));
            sorted = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bodyCapacity, SortedBodyStride);
            bodyCell = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bodyCapacity, 4 * sizeof(uint));
            staticList = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bodyCapacity, sizeof(uint));
            pairs = new GraphicsBuffer(GraphicsBuffer.Target.Structured, dynamicPairCapacity + staticPairCapacity, PairStride);
        }

        public int DynamicPairCapacity { get; }
        public int StaticPairCapacity { get; }

        /// <summary>The pair list: the dynamic-dynamic bin at [0, DynamicPairCapacity), then the dynamic-static bin.</summary>
        internal GraphicsBuffer Pairs => pairs;

        /// <summary>The control words (AVBD_BP_*), for kernels that read the pair counts on the GPU.</summary>
        internal GraphicsBuffer Control => control;

        /// <summary>Pairs found in the dynamic-dynamic bin by the last <see cref="Run"/>, stored or not. Valid after <see cref="ReadCounts"/>.</summary>
        public int DynamicPairCount => (int)Math.Min(dynamicPairs, int.MaxValue);

        /// <summary>Pairs found in the dynamic-static bin by the last <see cref="Run"/>, stored or not. Valid after <see cref="ReadCounts"/>.</summary>
        public int StaticPairCount => (int)Math.Min(staticPairs, int.MaxValue);

        /// <summary>True if the last <see cref="Run"/> found more pairs than a bin holds; the pairs beyond the capacity were dropped. Valid after <see cref="ReadCounts"/>.</summary>
        public bool Overflowed => dynamicPairs > DynamicPairCapacity || staticPairs > StaticPairCapacity;

        /// <summary>
        /// Runs the broadphase over the slots [0, bodies.HighWaterMark). Pending body edits are applied first. The constraints are the
        /// ones whose pairs are excluded; their lists must be built (<see cref="AvbdConstraintBuffers.Rebuild"/>). Pass null for none.
        /// </summary>
        public void Run(AvbdBodyBuffers bodies, AvbdConstraintBuffers constraints)
        {
            if (disposed) throw new ObjectDisposedException(nameof(AvbdBroadphase));
            if (bodies == null) throw new ArgumentNullException(nameof(bodies));
            if (bodies.Capacity != bodyCapacity)
                throw new ArgumentException("The body buffers have capacity " + bodies.Capacity + ", the broadphase was made for " + bodyCapacity + ".", nameof(bodies));

            bodies.ApplyEdits();
            int count = bodies.HighWaterMark;

            // Without constraint buffers nothing is excluded. The body ranges are then all empty, so these are never read.
            if (constraints == null && emptyRecords == null)
            {
                emptyRecords = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, ConstraintRecordStrides.Constraint);
                emptyList = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(uint));
            }
            GraphicsBuffer records = constraints != null ? constraints.Records : emptyRecords;
            GraphicsBuffer list = constraints != null ? constraints.List : emptyList;

            foreach (int kernel in new[]
            {
                resetKernel, clearGridKernel, boundsKernel, cellKernel, scanBlockKernel, scanTopKernel, scanApplyKernel, scatterKernel,
                pairsDynamicKernel, pairsStaticKernel,
            })
            {
                bodies.BindTo(shader, kernel);
                shader.SetBuffer(kernel, ControlId, control);
                shader.SetBuffer(kernel, CellStartId, cellStart);
                shader.SetBuffer(kernel, BlockSumsId, blockSums);
                shader.SetBuffer(kernel, BodyRankId, bodyRank);
                shader.SetBuffer(kernel, SortedId, sorted);
                shader.SetBuffer(kernel, BodyCellId, bodyCell);
                shader.SetBuffer(kernel, StaticListId, staticList);
                shader.SetBuffer(kernel, PairsId, pairs);
                shader.SetBuffer(kernel, RecordsId, records);
                shader.SetBuffer(kernel, ListId, list);
            }
            shader.SetInt(GridMaskId, gridCells - 1);
            shader.SetInt(DynamicCapacityId, DynamicPairCapacity);
            shader.SetInt(StaticCapacityId, StaticPairCapacity);

            shader.Dispatch(resetKernel, 1, 1, 1);
            if (count <= 0)
                return;

            AvbdDispatch.Linear(shader, clearGridKernel, gridCells);
            AvbdDispatch.Linear(shader, boundsKernel, count);
            AvbdDispatch.Linear(shader, cellKernel, count);

            // Scan of the cell counts into the start of every cell: blocks of ScanGroup cells, then the block sums, then the offsets (one extra item
            // for the total after the last cell).
            shader.Dispatch(scanBlockKernel, (gridCells + ScanGroup - 1) / ScanGroup, 1, 1);
            shader.Dispatch(scanTopKernel, 1, 1, 1);
            AvbdDispatch.Linear(shader, scanApplyKernel, gridCells + 1);

            AvbdDispatch.Linear(shader, scatterKernel, count);
            AvbdDispatch.Linear(shader, pairsDynamicKernel, count);
            AvbdDispatch.Linear(shader, pairsStaticKernel, count);
        }

        /// <summary>Reads the pair counts of the last <see cref="Run"/> back from the GPU (this waits for the GPU).</summary>
        public void ReadCounts()
        {
            if (disposed) throw new ObjectDisposedException(nameof(AvbdBroadphase));
            var words = new uint[ControlWords];
            control.GetData(words);
            dynamicPairs = words[DynamicPairsWord];
            staticPairs = words[StaticPairsWord];
        }

        /// <summary>Largest bounding-sphere radius among the solved bodies at the last <see cref="Run"/> (read back from the GPU).</summary>
        public float ReadMaxRadius()
        {
            var words = new uint[ControlWords];
            control.GetData(words);
            return BitConverter.ToSingle(BitConverter.GetBytes(words[MaxRadiusWord]), 0);
        }

        /// <summary>Reads the stored pairs of both bins back from the GPU: the dynamic-dynamic ones, then the dynamic-static ones. Calls <see cref="ReadCounts"/>.</summary>
        public PairRecord[] ReadPairs()
        {
            ReadCounts();
            int dynamicStored = (int)Math.Min(dynamicPairs, (uint)DynamicPairCapacity);
            int staticStored = (int)Math.Min(staticPairs, (uint)StaticPairCapacity);

            var result = new PairRecord[dynamicStored + staticStored];
            if (dynamicStored > 0)
            {
                var part = new PairRecord[dynamicStored];
                pairs.GetData(part, 0, 0, dynamicStored);
                Array.Copy(part, 0, result, 0, dynamicStored);
            }
            if (staticStored > 0)
            {
                var part = new PairRecord[staticStored];
                pairs.GetData(part, 0, DynamicPairCapacity, staticStored);
                Array.Copy(part, 0, result, dynamicStored, staticStored);
            }
            return result;
        }

        public void Dispose()
        {
            if (disposed)
                return;
            disposed = true;
            Release(ref control);
            Release(ref cellStart);
            Release(ref blockSums);
            Release(ref bodyRank);
            Release(ref sorted);
            Release(ref bodyCell);
            Release(ref staticList);
            Release(ref pairs);
            Release(ref emptyRecords);
            Release(ref emptyList);
        }

        static void Release(ref GraphicsBuffer buffer)
        {
            buffer?.Dispose();
            buffer = null;
        }
    }
}
