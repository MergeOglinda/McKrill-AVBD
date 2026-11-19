using System;
using UnityEngine;

namespace McKrill.Avbd
{
    /// <summary>What the last <see cref="AvbdBroadphase.Run"/> decided about the grid (see <see cref="AvbdBroadphase.ReadClassification"/>).</summary>
    internal struct BroadphaseClassification
    {
        /// <summary>Upper bound of the largest radius among the small bodies (the bodies of the grid); a bin edge, at most 4.4% above the radius.</summary>
        public float MaxSmallRadius;
        /// <summary>Bodies above every small body, tested against all (the large list).</summary>
        public int LargeCount;
        public bool TwoLevels;
        public float CoarseCell;
        public float FineCell;
        public int SmallBin;
        public int FineBin;
        /// <summary>Live bodies with a finite radius, and the solved ones among them.</summary>
        public int LiveCount;
        public int SolvedCount;
    }

    /// <summary>
    /// Stage B: finds the pairs of bodies whose bounding spheres overlap (<c>AvbdBroadphase.compute</c>), the candidates of the
    /// narrowphase. See the shader header for what a pair is and which pairs are left out.
    ///
    /// The grid is three-avbd's: a counting-sort grid (count per cell, scan, scatter) with a fine and a coarse level when a few bodies are much
    /// larger than the ones that move, and a list of large bodies (ground slabs) that are tested against everybody. Which bodies are large and
    /// how big the cells are is decided on the GPU from histograms of the radii, so nothing is read back.
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
        public const int MaxRadiusWord = 0;
        public const int LargeCountWord = 1;
        public const int DynamicPairsWord = 2;
        public const int StaticPairsWord = 3;
        public const int SmallBinWord = 4;
        public const int FineBinWord = 5;
        public const int TwoLevelsWord = 6;
        public const int CoarseCellWord = 7;
        public const int FineCellWord = 8;
        public const int LiveCountWord = 9;
        public const int SolvedCountWord = 10;
        public const int StatWords = 16;
        public const int BinsPerOctave = 16;
        public const int BinOctaves = 20;
        public const int RadiusBins = 2 * BinOctaves * BinsPerOctave;
        public const int HistAllWord = StatWords;
        public const int HistSolvedWord = StatWords + RadiusBins;
        public const int ControlWords = StatWords + 2 * RadiusBins;
        public const int LargeFactorBins = BinsPerOctave;
        public const int MaxLarge = 64;
        public const int ScanGroup = 1024;
        public const float CellMargin = 1.01f;

        /// <summary>Mirrors AVBD_BP_SEPARATION_PAD: meters of gap (world bounding boxes, face axes) above which a candidate pair is dropped.</summary>
        public const float SeparationPad = 0.001f;

        /// <summary>Mirrors AVBD_BP_STATIC_FLAG, AVBD_BP_LARGE_FLAG, AVBD_BP_COARSE_FLAG and AVBD_BP_SPHERE_FLAG: the bits of the grid word of a body (<c>_BodyCell.w</c>).</summary>
        public const uint StaticCellFlag = 0x80000000u;
        public const uint LargeCellFlag = 2u;
        public const uint CoarseCellFlag = 1u;
        public const uint SphereCellFlag = 4u;

        /// <summary>Mirrors AVBD_PAIR_SPHERE_A and AVBD_PAIR_SPHERE_B: the bits of <c>PairRecord.pair.w</c>, set when body A (B) of the pair is a sphere.</summary>
        public const uint PairSphereA = 1u;
        public const uint PairSphereB = 2u;

        /// <summary>Bytes of one <see cref="PairRecord"/>.</summary>
        public const int PairStride = 16;

        // The table is scanned in blocks of ScanGroup cells, and the sums of the blocks by one group: at most ScanGroup blocks.
        const int MinGridCells = ScanGroup;
        const int MaxGridCells = ScanGroup * ScanGroup;

        readonly ComputeShader shader;
        readonly int resetKernel, clearGridKernel, histogramKernel, classifyKernel, countKernel, scanBlockKernel, scanTopKernel, scanApplyKernel, scatterKernel, pairsKernel, pairsFixedKernel;

        readonly int bodyCapacity;
        readonly int gridCells;
        GraphicsBuffer control;
        GraphicsBuffer gridStart;
        GraphicsBuffer gridCursor;
        GraphicsBuffer gridSorted;
        GraphicsBuffer gridBlocks;
        GraphicsBuffer bodyCell;
        GraphicsBuffer largeList;
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
        static readonly int GridStartId = Shader.PropertyToID("_GridStart");
        static readonly int GridCursorId = Shader.PropertyToID("_GridCursor");
        static readonly int GridSortedId = Shader.PropertyToID("_GridSorted");
        static readonly int GridBlocksId = Shader.PropertyToID("_GridBlocks");
        static readonly int BodyCellId = Shader.PropertyToID("_BodyCell");
        static readonly int LargeListId = Shader.PropertyToID("_LargeList");
        static readonly int PairsId = Shader.PropertyToID("_Pairs");
        static readonly int TableMaskId = Shader.PropertyToID("_TableMask");
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
            histogramKernel = shader.FindKernel("BpHistogram");
            classifyKernel = shader.FindKernel("BpClassify");
            countKernel = shader.FindKernel("BpCount");
            scanBlockKernel = shader.FindKernel("BpScanBlock");
            scanTopKernel = shader.FindKernel("BpScanTop");
            scanApplyKernel = shader.FindKernel("BpScanApply");
            scatterKernel = shader.FindKernel("BpScatter");
            pairsKernel = shader.FindKernel("BpPairs");
            pairsFixedKernel = shader.FindKernel("BpPairsFixed");

            this.bodyCapacity = bodyCapacity;
            DynamicPairCapacity = dynamicPairCapacity;
            StaticPairCapacity = staticPairCapacity;

            // A power of two, at least twice the number of bodies, so that a cell holds few bodies on average (more bodies than half the
            // largest table share cells, which costs time and not correctness: a probe drops the bodies of other cells).
            int cells = MinGridCells;
            while (cells < bodyCapacity * 2L && cells < MaxGridCells)
                cells <<= 1;
            gridCells = cells;

            control = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ControlWords, sizeof(uint));
            gridStart = new GraphicsBuffer(GraphicsBuffer.Target.Structured, gridCells + 1, sizeof(uint));
            gridCursor = new GraphicsBuffer(GraphicsBuffer.Target.Structured, gridCells, sizeof(uint));
            gridSorted = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bodyCapacity, sizeof(uint));
            gridBlocks = new GraphicsBuffer(GraphicsBuffer.Target.Structured, gridCells / ScanGroup, sizeof(uint));
            bodyCell = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bodyCapacity, 4 * sizeof(uint));
            largeList = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bodyCapacity, sizeof(uint));
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

            foreach (int kernel in new[] { resetKernel, clearGridKernel, histogramKernel, classifyKernel, countKernel, scanBlockKernel, scanTopKernel, scanApplyKernel, scatterKernel, pairsKernel, pairsFixedKernel })
            {
                bodies.BindTo(shader, kernel);
                shader.SetBuffer(kernel, ControlId, control);
                shader.SetBuffer(kernel, GridStartId, gridStart);
                shader.SetBuffer(kernel, GridCursorId, gridCursor);
                shader.SetBuffer(kernel, GridSortedId, gridSorted);
                shader.SetBuffer(kernel, GridBlocksId, gridBlocks);
                shader.SetBuffer(kernel, BodyCellId, bodyCell);
                shader.SetBuffer(kernel, LargeListId, largeList);
                shader.SetBuffer(kernel, PairsId, pairs);
                shader.SetBuffer(kernel, RecordsId, records);
                shader.SetBuffer(kernel, ListId, list);
            }
            shader.SetInt(TableMaskId, gridCells - 1);
            shader.SetInt(DynamicCapacityId, DynamicPairCapacity);
            shader.SetInt(StaticCapacityId, StaticPairCapacity);

            AvbdDispatch.Linear(shader, resetKernel, ControlWords);
            if (count <= 0)
                return;

            AvbdDispatch.Linear(shader, clearGridKernel, gridCells + 1);
            AvbdDispatch.Linear(shader, histogramKernel, count);
            shader.Dispatch(classifyKernel, 1, 1, 1);
            AvbdDispatch.Linear(shader, countKernel, count);
            shader.Dispatch(scanBlockKernel, gridCells / ScanGroup, 1, 1);
            shader.Dispatch(scanTopKernel, 1, 1, 1);
            AvbdDispatch.Linear(shader, scanApplyKernel, gridCells);
            AvbdDispatch.Linear(shader, scatterKernel, count);
            AvbdDispatch.Linear(shader, pairsKernel, count);
            AvbdDispatch.Linear(shader, pairsFixedKernel, count);
        }

        /// <summary>Reads the pair counts of the last <see cref="Run"/> back from the GPU (this waits for the GPU).</summary>
        public void ReadCounts()
        {
            if (disposed) throw new ObjectDisposedException(nameof(AvbdBroadphase));
            uint[] words = ReadStatistics();
            dynamicPairs = words[DynamicPairsWord];
            staticPairs = words[StaticPairsWord];
        }

        /// <summary>
        /// Upper bound of the largest radius among the small bodies at the last <see cref="Run"/> (read back from the GPU): the radius that sizes the coarse cells.
        /// It is the upper edge of a radius bin, so between the radius and 1.05 times it.
        /// </summary>
        public float ReadMaxRadius() => ReadClassification().MaxSmallRadius;

        /// <summary>What the last <see cref="Run"/> decided about the grid (read back from the GPU).</summary>
        public BroadphaseClassification ReadClassification()
        {
            if (disposed) throw new ObjectDisposedException(nameof(AvbdBroadphase));
            uint[] words = ReadStatistics();
            return new BroadphaseClassification
            {
                MaxSmallRadius = AsFloat(words[MaxRadiusWord]),
                LargeCount = (int)words[LargeCountWord],
                TwoLevels = words[TwoLevelsWord] != 0u,
                CoarseCell = AsFloat(words[CoarseCellWord]),
                FineCell = AsFloat(words[FineCellWord]),
                SmallBin = (int)words[SmallBinWord],
                FineBin = (int)words[FineBinWord],
                LiveCount = (int)words[LiveCountWord],
                SolvedCount = (int)words[SolvedCountWord],
            };
        }

        uint[] ReadStatistics()
        {
            var words = new uint[StatWords];
            control.GetData(words, 0, 0, StatWords);
            return words;
        }

        static float AsFloat(uint bits) => BitConverter.ToSingle(BitConverter.GetBytes(bits), 0);

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
            Release(ref gridStart);
            Release(ref gridCursor);
            Release(ref gridSorted);
            Release(ref gridBlocks);
            Release(ref bodyCell);
            Release(ref largeList);
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
