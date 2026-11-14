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
    /// The host runs the rounds and reads back two small things: after every batch of rounds whether bodies are still waiting, and
    /// the number of bodies per color (the dispatch ranges of the primal sweep). Both go away when the primal sweep is dispatched
    /// indirectly (a later stage).
    /// Every color is recomputed from scratch each time <see cref="Run"/> is called; keeping colors between steps (temporal coherence)
    /// is not implemented.
    /// </summary>
    internal sealed class AvbdGraphColoring : IDisposable
    {
        /// <summary>Mirrors AVBD_COLOR_MAX.</summary>
        public const int MaxColors = 64;

        /// <summary>Mirrors AVBD_COLOR_PENDING.</summary>
        public const uint Pending = 0xFFFFFFFEu;

        // Mirror AVBD_COLOR_CONTROL_WORDS, _COUNTS_OFFSET, _CURSORS_OFFSET and _WORK_WORDS.
        public const int ControlWords = 4;
        public const int CountsOffset = 4;
        public const int CursorsOffset = 68;
        public const int WorkWords = 132;

        /// <summary>Rounds dispatched between two readbacks.</summary>
        const int RoundsPerBatch = 8;

        readonly ComputeShader shader;
        readonly int resetKernel, beginKernel, initKernel, roundKernel, countKernel, bucketKernel;
        GraphicsBuffer work;
        bool disposed;

        static readonly int BodyMetaId = Shader.PropertyToID("_BodyMeta");
        static readonly int BodyMassId = Shader.PropertyToID("_BodyMass");
        static readonly int BodyRangeId = Shader.PropertyToID("_BodyRange");
        static readonly int RecordsId = Shader.PropertyToID("_Constraints");
        static readonly int ListId = Shader.PropertyToID("_ConstraintList");
        static readonly int ColorBodiesId = Shader.PropertyToID("_ColorBodies");
        static readonly int WorkId = Shader.PropertyToID("_ColorWork");
        static readonly int OrderId = Shader.PropertyToID("_Order");

        /// <summary>Rounds the last <see cref="Run"/> dispatched (a multiple of the batch size).</summary>
        public int LastRounds { get; private set; }

        public AvbdGraphColoring(ComputeShader shader)
        {
            this.shader = shader != null ? shader : throw new InvalidOperationException("The AvbdShaderSet has no coloring shader.");
            resetKernel = shader.FindKernel("ColorReset");
            beginKernel = shader.FindKernel("ColorBegin");
            initKernel = shader.FindKernel("ColorInit");
            roundKernel = shader.FindKernel("ColorRound");
            countKernel = shader.FindKernel("ColorCount");
            bucketKernel = shader.FindKernel("ColorBucket");
            work = new GraphicsBuffer(GraphicsBuffer.Target.Structured, WorkWords, sizeof(uint));
            work.SetData(new uint[WorkWords]);
        }

        /// <summary>
        /// Colors the solved bodies among slots [0, bodyCount) and fills <paramref name="colorBodies"/> with the slots grouped by color
        /// (order inside a color is unspecified; it does not change the result of a sweep). Returns the number of colors and, in
        /// <paramref name="offsets"/>, the start of each color in the bucket array, with the total in entry [colors].
        /// </summary>
        /// <param name="records">Constraint records, read through the lists.</param>
        /// <param name="list">Per-body constraint lists; the offsets and counts are in the body Range array.</param>
        public int Run(AvbdBodyBuffers bodies, int bodyCount, GraphicsBuffer records, GraphicsBuffer list, GraphicsBuffer colorBodies,
            GraphColoringOrder order, out int[] offsets)
        {
            if (disposed) throw new ObjectDisposedException(nameof(AvbdGraphColoring));

            LastRounds = 0;
            offsets = new int[1];
            if (bodyCount <= 0)
                return 0;

            foreach (int kernel in new[] { resetKernel, beginKernel, initKernel, roundKernel, countKernel, bucketKernel })
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

            // Number of bodies per color, the start of each color, and the bucket fill.
            AvbdDispatch.Linear(shader, countKernel, bodyCount);
            var words = new uint[WorkWords];
            work.GetData(words);

            int colors = 0;
            for (int c = 0; c < MaxColors; c++)
                if (words[CountsOffset + c] != 0u)
                    colors = c + 1;

            offsets = new int[colors + 1];
            var cursors = new uint[Math.Max(colors, 1)];
            for (int c = 0; c < colors; c++)
            {
                cursors[c] = (uint)offsets[c];
                offsets[c + 1] = offsets[c] + (int)words[CountsOffset + c];
            }

            if (colors > 0)
            {
                work.SetData(cursors, 0, CursorsOffset, colors);
                AvbdDispatch.Linear(shader, bucketKernel, bodyCount);
            }
            return colors;
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
        }

        public void Dispose()
        {
            if (disposed)
                return;
            disposed = true;
            work?.Release();
            work = null;
        }
    }
}
