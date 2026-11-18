using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

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
    /// Reads the per-body constraint lists (stage G), writes the graph color into the body color array (<see cref="AvbdBodyBuffers.Color"/>) and fills the color buckets.
    ///
    /// The host runs the rounds and reads back one small thing: after every batch of rounds whether bodies are still waiting. The same
    /// words hold the number of colors, which is how many times the primal sweep is dispatched. How big each of those dispatches is
    /// stays on the GPU: <c>ColorPlan</c> turns the body count of every color into its start in the bucket array and into the arguments
    /// of an indirect dispatch (<see cref="Args"/>), which the primal sweep reads through <see cref="Work"/>.
    /// <see cref="Run"/> colors from scratch and waits for the GPU. <see cref="RunPersistent"/> keeps the colors of the step before, recolors only
    /// the bodies that now clash with a neighbor, and reads nothing back: it is what the solver runs every step with contacts, where the graph
    /// changes every step. A coloring from scratch is exact (the oracle's); a persistent one is a proper coloring that depends on the history.
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

        // Mirror AVBD_COLOR_STAT_RELEASED, _LAST_ROUND and _FINISHED: statistics of the persistent coloring.
        public const int StatReleased = 196;
        public const int StatLastRound = 197;
        public const int StatFinished = 198;
        public const int WorkWords = 200;

        /// <summary>
        /// Rounds after the ones the tuner chose, each dispatched indirectly with groups only if bodies are still waiting (ColorRoundPlan), so a step that
        /// needed more rounds than the tuner expected still ends in a proper coloring. What is left after them is colored by ColorForce.
        /// </summary>
        public const int ExtraRounds = 8;

        /// <summary>Most bodies the persistent coloring handles: an extra round is one indirect dispatch of at most 65535 groups.</summary>
        public const int MaxPersistentBodies = 65535 * AvbdConst.GroupSize / RoundLanes;

        /// <summary>Mirrors AVBD_COLOR_ARGS_STRIDE: bytes of indirect arguments per color (groups x, y and z).</summary>
        public const int ArgsStride = AvbdIndirect.ArgsStride;

        /// <summary>Mirrors AVBD_COLOR_ROW_GROUPS: thread groups per row of a big color (the limit of one dimension of a dispatch).</summary>
        public const int RowGroups = AvbdIndirect.RowGroups;

        /// <summary>Mirrors AVBD_PRIMAL_LANES_MAX: the most threads that solve one body in the primal sweep (colors of fewer than <see cref="PrimalLanesFrom4"/> bodies).</summary>
        public const int MaxPrimalLanes = 8;

        /// <summary>Mirrors AVBD_PRIMAL_LANES_FROM_4: the size of a color from which 4 threads solve a body.</summary>
        public const int PrimalLanesFrom4 = 4096;

        /// <summary>Mirrors AVBD_PRIMAL_LANES_FROM_2: the size of a color from which 2 threads solve a body.</summary>
        public const int PrimalLanesFrom2 = 32768;

        /// <summary>
        /// Threads that solve one body in the primal sweep over a color of <paramref name="bodies"/> bodies (AvbdPrimalLanes in the HLSL): 8, then 4 from
        /// <see cref="PrimalLanesFrom4"/> bodies and 2 from <see cref="PrimalLanesFrom2"/>. A small color is bound by the longest serial chain of loads of its
        /// busiest body, which more lanes shorten; a big color fills the GPU with fewer lanes and then pays less for the reduction. After three-avbd's rule.
        /// </summary>
        public static int PrimalLanesFor(long bodies) => bodies >= PrimalLanesFrom2 ? 2 : bodies >= PrimalLanesFrom4 ? 4 : MaxPrimalLanes;

        /// <summary>Mirrors AVBD_COLOR_LANES: threads that look at the neighbors of one body in a coloring round.</summary>
        public const int RoundLanes = 8;

        /// <summary>Mirrors AVBD_BUCKET_SPAN: the slots each thread of ColorBucket handles (a thread group handles GroupSize * BucketSpan of them).</summary>
        public const int BucketSpan = 4;

        // Rounds dispatched before the first readback, and between the later ones. A round after the coloring is done finds every body colored
        // and ends at once, so a long first batch costs little and, for the usual graph (a few rounds with the hashed order), saves the second
        // wait for the GPU.
        const int FirstBatchRounds = 16;
        const int RoundsPerBatch = 8;

        /// <summary>Steps with persistent colors between two colorings from scratch (colors drift upward while they are kept).</summary>
        public const int DefaultFreshInterval = 64;

        // Statistics requests in flight at most; a step issues no new one while this many wait (the readback takes a few frames).
        const int MaxStatsRequests = 3;

        readonly ComputeShader shader;
        readonly int resetKernel, beginKernel, initKernel, keepKernel, markKernel, releaseKernel, roundKernel, roundPlanKernel, forceKernel, countKernel, planKernel, bucketKernel;
        readonly ColoringTuner tuner = new ColoringTuner();
        readonly Queue<StatsRequest> statsRequests = new Queue<StatsRequest>();
        GraphicsBuffer work;
        GraphicsBuffer args;
        GraphicsBuffer scratch;     // one word per body slot: the conflict flags of the persistent coloring
        GraphicsBuffer finishArgs;  // indirect arguments of an extra round (ColorRoundPlan)
        int scratchCapacity;
        int stepsSincePerfect;      // steps since the last coloring from scratch
        bool baseline;              // the colors in the body records are a coloring of the last graph that was built
        bool disposed;

        struct StatsRequest
        {
            public AsyncGPUReadbackRequest request;
            public bool fresh;
            public int rounds;
            public int cap;
        }

        static readonly int BodyColorId = Shader.PropertyToID("_BodyColor");
        static readonly int BodyFlagsId = Shader.PropertyToID("_BodyFlags");
        static readonly int BodyMassId = Shader.PropertyToID("_BodyMass");
        static readonly int BodyRangeId = Shader.PropertyToID("_BodyRange");
        static readonly int RecordsId = Shader.PropertyToID("_Constraints");
        static readonly int ListId = Shader.PropertyToID("_ConstraintList");
        static readonly int OtherId = Shader.PropertyToID("_ConstraintOther");
        static readonly int ColorBodiesId = Shader.PropertyToID("_ColorBodies");
        static readonly int WorkId = Shader.PropertyToID("_ColorWork");
        static readonly int ArgsId = Shader.PropertyToID("_ColorArgs");
        static readonly int OrderId = Shader.PropertyToID("_Order");
        static readonly int CapId = Shader.PropertyToID("_ColorCap");
        static readonly int RoundId = Shader.PropertyToID("_Round");
        static readonly int ScratchId = Shader.PropertyToID("_ColorScratch");
        static readonly int FinishArgsId = Shader.PropertyToID("_ColorFinishArgs");
        static readonly int BaseId = Shader.PropertyToID("_Base");
        static readonly int CountId = Shader.PropertyToID("_Count");
        static readonly int BucketEndId = Shader.PropertyToID("_BucketEnd");

        /// <summary>Rounds the last <see cref="Run"/> or <see cref="RunPersistent"/> dispatched (from scratch: the first batch, then a multiple of the batch size).</summary>
        public int LastRounds { get; private set; }

        /// <summary>
        /// Steps (<see cref="RunPersistent"/> calls) that keep colors between two that color from scratch. The step that follows <see cref="Run"/>
        /// always colors from scratch. Zero colors from scratch every time.
        /// </summary>
        public int FreshInterval { get; set; } = DefaultFreshInterval;

        /// <summary>Colors the primal sweep dispatches after the last <see cref="RunPersistent"/>.</summary>
        public int Cap => tuner.Cap;

        /// <summary>True if the colors in the body records are a coloring of the last graph, so a persistent step can keep them. False until <see cref="Run"/> has run.</summary>
        public bool HasBaseline => baseline;

        /// <summary>The statistics of the newest persistent step that has arrived from the GPU (they are a few frames old), and how many have arrived.</summary>
        internal ColoringSample LastSample { get; private set; }
        public int SamplesSeen { get; private set; }

        /// <summary>The tuner, for tests.</summary>
        internal ColoringTuner Tuner => tuner;

        /// <summary>The work buffer: the body count and the start of every color, as of the last <see cref="Run"/>. The primal sweep reads it.</summary>
        public GraphicsBuffer Work => work;

        /// <summary>Indirect dispatch arguments, <see cref="ArgsStride"/> bytes per color (up to <see cref="MaxColors"/>), written by the GPU in <see cref="Run"/>.</summary>
        public GraphicsBuffer Args => args;

        /// <summary>
        /// Thread groups of the primal sweep over a color of <paramref name="count"/> bodies, as <c>ColorPlan</c> computes them:
        /// <see cref="PrimalLanesFor"/> threads per body, x at most <see cref="RowGroups"/> and y the number of rows of that width.
        /// The mirror of the HLSL, which a test compares with the GPU's.
        /// </summary>
        public static (uint x, uint y) GroupsFor(long count) => AvbdIndirect.GroupsFor(count * PrimalLanesFor(count));

        public AvbdGraphColoring(ComputeShader shader)
        {
            this.shader = shader != null ? shader : throw new InvalidOperationException("The AvbdShaderSet has no coloring shader.");
            resetKernel = shader.FindKernel("ColorReset");
            beginKernel = shader.FindKernel("ColorBegin");
            initKernel = shader.FindKernel("ColorInit");
            keepKernel = shader.FindKernel("ColorKeep");
            markKernel = shader.FindKernel("ColorMark");
            releaseKernel = shader.FindKernel("ColorRelease");
            roundKernel = shader.FindKernel("ColorRound");
            roundPlanKernel = shader.FindKernel("ColorRoundPlan");
            forceKernel = shader.FindKernel("ColorForce");
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

            finishArgs = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments | GraphicsBuffer.Target.Raw, 3, sizeof(uint));
            finishArgs.SetData(new uint[] { 0u, 1u, 1u });
        }

        /// <summary>
        /// Colors the solved bodies among slots [0, bodyCount) and fills <paramref name="colorBodies"/> with the slots grouped by color
        /// (slot order inside each chunk of GroupSize * BucketSpan slots, chunks in timing order: it does not change the result of a sweep, but a sweep reads memory more coherently). Returns the number of colors. The start and
        /// the size of every color are on the GPU (<see cref="Work"/>, <see cref="Args"/>); <see cref="ReadSchedule"/> reads them back for tests.
        /// </summary>
        /// <param name="records">Constraint records, read through the lists.</param>
        /// <param name="list">Per-body constraint lists; the offsets and counts are in the body Range array.</param>
        /// <param name="other">For each list entry, the body on the other side of the constraint (<see cref="AvbdConstraintLists.Other"/>).</param>
        public int Run(AvbdBodyBuffers bodies, int bodyCount, GraphicsBuffer records, GraphicsBuffer list, GraphicsBuffer other,
            GraphicsBuffer colorBodies, GraphColoringOrder order)
        {
            if (disposed) throw new ObjectDisposedException(nameof(AvbdGraphColoring));

            LastRounds = 0;
            PollStatistics(true); // the statistics of persistent steps before this one are about another graph: take them in before the tuner starts again
            if (bodyCount <= 0)
            {
                baseline = false;
                return 0;
            }

            BindAll(bodies, records, list, other, colorBodies);
            shader.SetInt(OrderId, (int)order);
            shader.SetInt(CapId, MaxColors);

            AvbdDispatch.Linear(shader, resetKernel, WorkWords);
            AvbdDispatch.Linear(shader, initKernel, bodyCount);

            // Rounds. Before the last round of a batch the counter of waiting bodies is cleared, so after the batch it holds the
            // number that were still waiting after that round.
            var control = new uint[ControlWords];
            int limit = bodyCount + FirstBatchRounds; // the highest-priority waiting body colors in every round, so this is never reached
            int batch = FirstBatchRounds;
            while (true)
            {
                for (int i = 0; i < batch; i++)
                {
                    if (i == batch - 1)
                        shader.Dispatch(beginKernel, 1, 1, 1);
                    shader.SetInt(RoundId, LastRounds + i);
                    AvbdDispatch.Linear(shader, roundKernel, bodyCount * RoundLanes);
                }
                LastRounds += batch;
                batch = RoundsPerBatch;

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

            FinishColoring(bodyCount);

            // The colors in the body records are now a coloring of this graph: the next persistent step may keep them, and starts from scratch
            // (the graph of the contacts is not this one), with the cap at its largest until the first statistics arrive.
            baseline = true;
            stepsSincePerfect = FreshInterval;
            tuner.Restart();
            return colors;
        }

        /// <summary>
        /// Colors the solved bodies like <see cref="Run"/>, but keeps the colors of the step before and reads nothing back, so the host never waits for
        /// the GPU. The bodies that now have the color of a neighbor that colors before them (and the new ones) are recolored by a fixed number of
        /// rounds, then up to <see cref="ExtraRounds"/> more that run only while bodies wait, and <c>ColorForce</c> for what is still left. Every <see cref="FreshInterval"/> steps, and in the step after <see cref="Run"/>,
        /// every body is recolored (colors drift upward while they are kept). Returns the number of colors the primal sweep must dispatch (the cap):
        /// at least the number of colors in use, in all but the steps that a lowered cap or a graph that suddenly needs more colors catch up with; a body
        /// without a free color below the cap shares the last color for a few steps. Only meaningful after <see cref="Run"/>.
        /// </summary>
        public int RunPersistent(AvbdBodyBuffers bodies, int bodyCount, GraphicsBuffer records, GraphicsBuffer list, GraphicsBuffer other,
            GraphicsBuffer colorBodies, GraphColoringOrder order)
        {
            if (disposed) throw new ObjectDisposedException(nameof(AvbdGraphColoring));
            if (!baseline)
                throw new InvalidOperationException("RunPersistent needs a coloring from scratch (Run) first.");

            LastRounds = 0;
            PollStatistics(false);
            if (bodyCount <= 0)
                return 0;

            EnsureScratch(bodies.Capacity);
            BindAll(bodies, records, list, other, colorBodies);

            bool fresh = FreshInterval <= 0 || stepsSincePerfect >= FreshInterval;
            int cap = tuner.Cap;
            int rounds = tuner.RoundsFor(fresh);
            shader.SetInt(OrderId, (int)order);
            shader.SetInt(CapId, cap);

            AvbdDispatch.Linear(shader, resetKernel, WorkWords);
            if (fresh)
            {
                AvbdDispatch.Linear(shader, initKernel, bodyCount);
                stepsSincePerfect = 0;
            }
            else
            {
                AvbdDispatch.Linear(shader, keepKernel, bodyCount);
                AvbdDispatch.Linear(shader, markKernel, bodyCount);
                AvbdDispatch.Linear(shader, releaseKernel, bodyCount);
                stepsSincePerfect++;
            }

            // The counter of waiting bodies is cleared before the last round, so after it the counter holds the bodies that are still waiting.
            for (int i = 0; i < rounds; i++)
            {
                if (i == rounds - 1)
                    shader.Dispatch(beginKernel, 1, 1, 1);
                shader.SetInt(RoundId, i);
                AvbdDispatch.Linear(shader, roundKernel, bodyCount * RoundLanes);
            }
            LastRounds = rounds;

            // The bodies that still wait get more rounds, each only if some are left (an indirect dispatch without groups otherwise); then ColorForce.
            shader.SetInt(BaseId, 0);
            shader.SetInt(CountId, bodyCount * RoundLanes);
            for (int i = 0; i < ExtraRounds; i++)
            {
                shader.Dispatch(roundPlanKernel, 1, 1, 1);
                shader.SetInt(RoundId, rounds + i);
                shader.DispatchIndirect(roundKernel, finishArgs, 0);
            }
            AvbdDispatch.Linear(shader, forceKernel, bodyCount);

            FinishColoring(bodyCount);
            RequestStatistics(fresh, rounds, cap);
            return cap;
        }

        // The size of every color, then (on the GPU) the start of every color, the number of colors and the dispatch arguments, then the bucket fill.
        void FinishColoring(int bodyCount)
        {
            AvbdDispatch.Linear(shader, countKernel, bodyCount);
            shader.Dispatch(planKernel, 1, 1, 1);
            shader.SetInt(BucketEndId, bodyCount);
            AvbdDispatch.Linear(shader, bucketKernel, (bodyCount + BucketSpan - 1) / BucketSpan);
        }

        void EnsureScratch(int capacity)
        {
            if (scratch != null && scratchCapacity >= capacity)
                return;
            scratch?.Release();
            scratchCapacity = Math.Max(1, capacity);
            scratch = new GraphicsBuffer(GraphicsBuffer.Target.Structured, scratchCapacity, sizeof(uint));
        }

        // ---- statistics ------------------------------------------------------------------------------------

        // Asks for the work buffer (the statistics words are in it) at the end of this step. The copy happens on the GPU timeline, so it holds this
        // step's numbers; the request completes a few frames later, and nothing waits for it.
        void RequestStatistics(bool fresh, int rounds, int cap)
        {
            if (statsRequests.Count >= MaxStatsRequests)
                return;
            statsRequests.Enqueue(new StatsRequest
            {
                request = AsyncGPUReadback.Request(work, WorkWords * sizeof(uint), 0),
                fresh = fresh,
                rounds = rounds,
                cap = cap,
            });
        }

        /// <summary>
        /// Takes in the statistics requests that have arrived (the oldest first) and lets the tuner learn from them. With <paramref name="wait"/> it waits
        /// for every request in flight: for tests and tools, not for the step.
        /// </summary>
        public void PollStatistics(bool wait)
        {
            while (statsRequests.Count > 0)
            {
                StatsRequest next = statsRequests.Peek();
                if (wait)
                    next.request.WaitForCompletion();
                else
                {
                    next.request.Update();
                    if (!next.request.done)
                        return;
                }
                statsRequests.Dequeue();
                if (next.request.hasError)
                    continue;

                var words = next.request.GetData<uint>();
                var sample = new ColoringSample
                {
                    fresh = next.fresh,
                    rounds = next.rounds,
                    cap = next.cap,
                    released = words[StatReleased],
                    leftover = words[StatFinished],
                    forced = words[1],
                    colors = words[ColorCountWord],
                    lastRound = words[StatLastRound],
                };
                LastSample = sample;
                SamplesSeen++;
                tuner.Observe(sample);
            }
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

        void BindAll(AvbdBodyBuffers bodies, GraphicsBuffer records, GraphicsBuffer list, GraphicsBuffer other, GraphicsBuffer colorBodies)
        {
            EnsureScratch(bodies.Capacity);
            foreach (int kernel in new[] { resetKernel, beginKernel, initKernel, keepKernel, markKernel, releaseKernel, roundKernel, roundPlanKernel, forceKernel, countKernel, planKernel, bucketKernel })
                Bind(kernel, bodies, records, list, other, colorBodies);
        }

        void Bind(int kernel, AvbdBodyBuffers bodies, GraphicsBuffer records, GraphicsBuffer list, GraphicsBuffer other, GraphicsBuffer colorBodies)
        {
            shader.SetBuffer(kernel, BodyColorId, bodies.Color);
            shader.SetBuffer(kernel, BodyFlagsId, bodies.Flags);
            shader.SetBuffer(kernel, BodyMassId, bodies.Mass);
            shader.SetBuffer(kernel, BodyRangeId, bodies.Range);
            shader.SetBuffer(kernel, RecordsId, records);
            shader.SetBuffer(kernel, ListId, list);
            shader.SetBuffer(kernel, OtherId, other);
            shader.SetBuffer(kernel, ColorBodiesId, colorBodies);
            shader.SetBuffer(kernel, WorkId, work);
            shader.SetBuffer(kernel, ArgsId, args);
            shader.SetBuffer(kernel, ScratchId, scratch);
            shader.SetBuffer(kernel, FinishArgsId, finishArgs);
        }

        public void Dispose()
        {
            if (disposed)
                return;
            disposed = true;
            // A readback in flight reads the work buffer: let it finish before the buffer goes.
            while (statsRequests.Count > 0)
                statsRequests.Dequeue().request.WaitForCompletion();
            work?.Release();
            work = null;
            args?.Release();
            args = null;
            scratch?.Release();
            scratch = null;
            finishArgs?.Release();
            finishArgs = null;
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
