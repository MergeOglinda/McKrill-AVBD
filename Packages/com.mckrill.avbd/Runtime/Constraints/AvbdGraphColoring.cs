using System;
using UnityEngine;

namespace McKrill.Avbd
{
    /// <summary>Which body colors first when two neighbors compete. Both orders give a valid coloring; they differ in speed and in the colors.</summary>
    public enum GraphColoringOrder
    {
        /// <summary>
        /// A higher slot colors first. The coloring is exactly the reference solver's sequential greedy coloring, so the colored sweep
        /// is the same sequence of updates (and the GPU solver can be compared with the reference). The number of rounds is the
        /// length of the longest chain of neighbors with falling slots, which is the whole chain for a rope built in slot order.
        /// </summary>
        SlotOrder = 0,

        /// <summary>A hash of the slot decides. The chains are short (a few rounds for any graph of bodies), the coloring differs from the reference's.</summary>
        Hashed = 1,
    }

    /// <summary>
    /// Stage C: colors the graph of solved bodies on the GPU (<c>AvbdColoring.compute</c>) and buckets the bodies by color.
    /// Reads the per-body constraint lists (stage G), writes the graph color into Meta.state.x and fills the color buckets.
    ///
    /// The host runs the rounds and reads back one small thing: after every batch of rounds whether bodies are still waiting. The same
    /// words hold the number of colors, which is how many times the primal sweep is dispatched. How big each of those dispatches is
    /// stays on the GPU: <c>ColorPlan</c> turns the body count of every color into its start in the bucket array and into the arguments
    /// of an indirect dispatch (<see cref="Args"/>), which the primal sweep reads through <see cref="Work"/>.
    /// Every color is recomputed from scratch each time <see cref="Run"/> is called; keeping colors between steps (temporal coherence)
    /// is not implemented.
    /// </summary>
    internal sealed class AvbdGraphColoring : IDisposable
    {
        /// <summary>Mirrors AVBD_COLOR_MAX.</summary>
        public const int MaxColors = 64;

        /// <summary>Mirrors AVBD_COLOR_PENDING.</summary>
        public const uint Pending = 0xFFFFFFFEu;

        // Mirror AVBD_COLOR_CONTROL_WORDS, _COUNT_WORD, _COUNTS_OFFSET, _CURSORS_OFFSET, _STARTS_OFFSET and _WORK_WORDS.
        public const int ControlWords = 4;
        public const int ColorCountWord = 2;
        public const int CountsOffset = 4;
        public const int CursorsOffset = 68;
        public const int StartsOffset = 132;
        public const int WorkWords = 196;

        /// <summary>Mirrors AVBD_COLOR_ARGS_STRIDE: bytes of indirect arguments per color (groups x, y and z).</summary>
        public const int ArgsStride = 12;

        /// <summary>Mirrors AVBD_COLOR_ROW_GROUPS: thread groups per row of a big color (the limit of one dimension of a dispatch).</summary>
        public const int RowGroups = 65535;

        /// <summary>Rounds dispatched between two readbacks.</summary>
        const int RoundsPerBatch = 8;

        readonly ComputeShader shader;
        readonly int resetKernel, beginKernel, initKernel, roundKernel, countKernel, planKernel, bucketKernel;
        GraphicsBuffer work;
        GraphicsBuffer args;
        bool disposed;

        static readonly int BodyMetaId = Shader.PropertyToID("_BodyMeta");
        static readonly int BodyMassId = Shader.PropertyToID("_BodyMass");
        static readonly int BodyRangeId = Shader.PropertyToID("_BodyRange");
        static readonly int RecordsId = Shader.PropertyToID("_Constraints");
        static readonly int ListId = Shader.PropertyToID("_ConstraintList");
        static readonly int ColorBodiesId = Shader.PropertyToID("_ColorBodies");
        static readonly int WorkId = Shader.PropertyToID("_ColorWork");
        static readonly int ArgsId = Shader.PropertyToID("_ColorArgs");
        static readonly int OrderId = Shader.PropertyToID("_Order");

        /// <summary>Rounds the last <see cref="Run"/> dispatched (a multiple of the batch size).</summary>
        public int LastRounds { get; private set; }

        /// <summary>The work buffer: the body count and the start of every color, as of the last <see cref="Run"/>. The primal sweep reads it.</summary>
        public GraphicsBuffer Work => work;

        /// <summary>Indirect dispatch arguments, <see cref="ArgsStride"/> bytes per color (up to <see cref="MaxColors"/>), written by the GPU in <see cref="Run"/>.</summary>
        public GraphicsBuffer Args => args;

        /// <summary>
        /// Thread groups of an indirect dispatch over <paramref name="count"/> bodies, as <c>ColorPlan</c> computes them: x is at most
        /// <see cref="RowGroups"/> and y the number of rows of that width. The mirror of the HLSL, which a test compares with the GPU's.
        /// </summary>
        public static (uint x, uint y) GroupsFor(long count)
        {
            long groups = (count + AvbdConst.GroupSize - 1) / AvbdConst.GroupSize;
            long rows = Math.Max(1L, (groups + RowGroups - 1) / RowGroups);
            return ((uint)Math.Min(groups, RowGroups), (uint)rows);
        }

        public AvbdGraphColoring(ComputeShader shader)
        {
            this.shader = shader != null ? shader : throw new InvalidOperationException("The AvbdShaderSet has no coloring shader.");
            resetKernel = shader.FindKernel("ColorReset");
            beginKernel = shader.FindKernel("ColorBegin");
            initKernel = shader.FindKernel("ColorInit");
            roundKernel = shader.FindKernel("ColorRound");
            countKernel = shader.FindKernel("ColorCount");
            planKernel = shader.FindKernel("ColorPlan");
            bucketKernel = shader.FindKernel("ColorBucket");
            work = new GraphicsBuffer(GraphicsBuffer.Target.Structured, WorkWords, sizeof(uint));
            work.SetData(new uint[WorkWords]);

            // The arguments are written by a compute shader (a raw byte-address buffer) and read by DispatchIndirect.
            var initialArgs = new uint[MaxColors * ArgsStride / sizeof(uint)];
            for (int c = 0; c < MaxColors; c++)
            {
                initialArgs[c * 3 + 1] = 1u; // an empty dispatch: 0 x 1 x 1 groups
                initialArgs[c * 3 + 2] = 1u;
            }
            args = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments | GraphicsBuffer.Target.Raw, initialArgs.Length, sizeof(uint));
            args.SetData(initialArgs);
        }

        /// <summary>
        /// Colors the solved bodies among slots [0, bodyCount) and fills <paramref name="colorBodies"/> with the slots grouped by color
        /// (order inside a color is unspecified; it does not change the result of a sweep). Returns the number of colors. The start and
        /// the size of every color are on the GPU (<see cref="Work"/>, <see cref="Args"/>); <see cref="ReadSchedule"/> reads them back for tests.
        /// </summary>
        /// <param name="records">Constraint records, read through the lists.</param>
        /// <param name="list">Per-body constraint lists; the offsets and counts are in the body Range array.</param>
        public int Run(AvbdBodyBuffers bodies, int bodyCount, GraphicsBuffer records, GraphicsBuffer list, GraphicsBuffer colorBodies,
            GraphColoringOrder order)
        {
            if (disposed) throw new ObjectDisposedException(nameof(AvbdGraphColoring));

            LastRounds = 0;
            if (bodyCount <= 0)
                return 0;

            foreach (int kernel in new[] { resetKernel, beginKernel, initKernel, roundKernel, countKernel, planKernel, bucketKernel })
                Bind(kernel, bodies, records, list, colorBodies);
            shader.SetInt(OrderId, (int)order);

            AvbdDispatch.Linear(shader, resetKernel, WorkWords);
            AvbdDispatch.Linear(shader, initKernel, bodyCount);

            // Rounds. Before the last round of a batch the counter of waiting bodies is cleared, so after the batch it holds the
            // number that were still waiting after that round.
            var control = new uint[ControlWords];
            int limit = bodyCount + RoundsPerBatch; // the highest-priority waiting body colors in every round, so this is never reached
            while (true)
            {
                for (int i = 0; i < RoundsPerBatch; i++)
                {
                    if (i == RoundsPerBatch - 1)
                        shader.Dispatch(beginKernel, 1, 1, 1);
                    AvbdDispatch.Linear(shader, roundKernel, bodyCount);
                }
                LastRounds += RoundsPerBatch;

                work.GetData(control, 0, 0, ControlWords);
                if (control[0] == 0u)
                    break;
                if (LastRounds > limit)
                    throw new InvalidOperationException("Graph coloring did not finish within " + LastRounds + " rounds.");
            }

            if (control[1] != 0u)
                throw new InvalidOperationException(
                    "The constraint graph needs more than " + MaxColors + " colors. Bodies connected to many mutually connected bodies are not supported.");

            int colors = (int)control[ColorCountWord];

            // The size of every color, then (on the GPU) the start of every color and the dispatch arguments, then the bucket fill.
            AvbdDispatch.Linear(shader, countKernel, bodyCount);
            shader.Dispatch(planKernel, 1, 1, 1);
            AvbdDispatch.Linear(shader, bucketKernel, bodyCount);
            return colors;
        }

        /// <summary>What <see cref="Run"/> left on the GPU for each of the first <paramref name="colors"/> colors. Waits for the GPU.</summary>
        public ColorSchedule ReadSchedule(int colors)
        {
            if (disposed) throw new ObjectDisposedException(nameof(AvbdGraphColoring));

            colors = Math.Max(0, Math.Min(colors, MaxColors));
            var words = new uint[WorkWords];
            work.GetData(words);
            var rawArgs = new uint[MaxColors * ArgsStride / sizeof(uint)];
            args.GetData(rawArgs);

            var schedule = new ColorSchedule
            {
                Starts = new int[colors + 1],
                Counts = new int[colors],
                GroupsX = new uint[colors],
                GroupsY = new uint[colors],
                GroupsZ = new uint[colors],
            };
            for (int c = 0; c < colors; c++)
            {
                schedule.Starts[c] = (int)words[StartsOffset + c];
                schedule.Counts[c] = (int)words[CountsOffset + c];
                schedule.GroupsX[c] = rawArgs[c * 3];
                schedule.GroupsY[c] = rawArgs[c * 3 + 1];
                schedule.GroupsZ[c] = rawArgs[c * 3 + 2];
            }
            schedule.Starts[colors] = colors > 0 ? schedule.Starts[colors - 1] + schedule.Counts[colors - 1] : 0;
            return schedule;
        }

        void Bind(int kernel, AvbdBodyBuffers bodies, GraphicsBuffer records, GraphicsBuffer list, GraphicsBuffer colorBodies)
        {
            shader.SetBuffer(kernel, BodyMetaId, bodies.Meta);
            shader.SetBuffer(kernel, BodyMassId, bodies.Mass);
            shader.SetBuffer(kernel, BodyRangeId, bodies.Range);
            shader.SetBuffer(kernel, RecordsId, records);
            shader.SetBuffer(kernel, ListId, list);
            shader.SetBuffer(kernel, ColorBodiesId, colorBodies);
            shader.SetBuffer(kernel, WorkId, work);
            shader.SetBuffer(kernel, ArgsId, args);
        }

        public void Dispose()
        {
            if (disposed)
                return;
            disposed = true;
            work?.Release();
            work = null;
            args?.Release();
            args = null;
        }
    }

    /// <summary>The start and size of each color in the bucket array, and the indirect dispatch arguments that go with them (a test and tool view).</summary>
    internal struct ColorSchedule
    {
        /// <summary>Start of each color in the bucket array; one extra entry at the end holds the total.</summary>
        public int[] Starts;
        public int[] Counts;
        public uint[] GroupsX;
        public uint[] GroupsY;
        public uint[] GroupsZ;
    }
}
