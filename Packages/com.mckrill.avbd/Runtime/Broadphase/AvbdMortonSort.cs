using System;
using UnityEngine;

namespace McKrill.Avbd
{
    /// <summary>
    /// Stage S: sorts the body slots along a Morton curve (<c>AvbdMortonSort.compute</c>), so that bodies that are near each other in space
    /// are near each other in <see cref="Order"/>. Stages B and C visit the bodies in that order.
    ///
    /// <see cref="Order"/> is always a permutation of the slots [0, <see cref="SortedCount"/>): the solved bodies first, by Morton key, then
    /// the rest. It is an order and nothing else, so using one that is out of date gives the same results as slot order. It is only a
    /// permutation of the slots that exist now if <see cref="SortedCount"/> equals the body count of the consumer; a consumer checks that
    /// (<see cref="Covers"/>) and visits the slots in slot order otherwise.
    ///
    /// Everything stays on the GPU; there is no readback outside tests (<see cref="ReadSorted"/>).
    /// </summary>
    internal sealed class AvbdMortonSort : IDisposable
    {
        // Mirror AVBD_SORT_*.
        public const int ControlWords = 4;
        public const int MaxRadiusWord = 0;
        public const int AxisBits = 10;
        public const int KeyBits = 30;
        public const uint InvalidKey = 0x3FFFFFFFu;
        public const int RadixBits = 6;
        public const int Buckets = 64;
        public const int Passes = 5;
        public const int Tile = 256;
        public const int ScanThreads = 1024;
        public const int MaxTiles = 65535;

        /// <summary>The most body slots the sort handles: <see cref="MaxTiles"/> tiles.</summary>
        public const int MaxBodies = MaxTiles * Tile;

        readonly ComputeShader shader;
        readonly int beginKernel, boundsKernel, keysKernel, countKernel, scanKernel, scatterKernel;
        readonly int bodyCapacity;

        GraphicsBuffer keysA, keysB, valuesA, valuesB; // the sort ping-pongs between the two; the result of the last pass is in B (an odd number of passes)
        GraphicsBuffer histogram;
        GraphicsBuffer control;
        bool disposed;

        static readonly int BodyStateId = Shader.PropertyToID("_BodyState");
        static readonly int BodyMassId = Shader.PropertyToID("_BodyMass");
        static readonly int BodyShapeId = Shader.PropertyToID("_BodyShape");
        static readonly int BodyMetaId = Shader.PropertyToID("_BodyMeta");
        static readonly int ControlId = Shader.PropertyToID("_SortControl");
        static readonly int KeysInId = Shader.PropertyToID("_KeysIn");
        static readonly int ValuesInId = Shader.PropertyToID("_ValuesIn");
        static readonly int KeysOutId = Shader.PropertyToID("_KeysOut");
        static readonly int ValuesOutId = Shader.PropertyToID("_ValuesOut");
        static readonly int HistogramId = Shader.PropertyToID("_Histogram");
        static readonly int SortCountId = Shader.PropertyToID("_SortCount");
        static readonly int TilesId = Shader.PropertyToID("_Tiles");
        static readonly int PassId = Shader.PropertyToID("_Pass");

        /// <param name="bodyCapacity">Slot capacity of the body buffers this sort serves, at most <see cref="MaxBodies"/>.</param>
        public AvbdMortonSort(ComputeShader shader, int bodyCapacity)
        {
            this.shader = shader != null ? shader : throw new InvalidOperationException("The AvbdShaderSet has no Morton sort shader.");
            if (bodyCapacity < 1) throw new ArgumentOutOfRangeException(nameof(bodyCapacity));
            if (bodyCapacity > MaxBodies)
                throw new ArgumentOutOfRangeException(nameof(bodyCapacity), "The sort handles at most " + MaxBodies + " body slots.");

            beginKernel = shader.FindKernel("SortBegin");
            boundsKernel = shader.FindKernel("SortBounds");
            keysKernel = shader.FindKernel("SortKeys");
            countKernel = shader.FindKernel("SortCount");
            scanKernel = shader.FindKernel("SortScan");
            scatterKernel = shader.FindKernel("SortScatter");

            this.bodyCapacity = bodyCapacity;
            int tiles = TilesFor(bodyCapacity);

            keysA = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bodyCapacity, sizeof(uint));
            keysB = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bodyCapacity, sizeof(uint));
            valuesA = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bodyCapacity, sizeof(uint));
            valuesB = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bodyCapacity, sizeof(uint));
            histogram = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Buckets * tiles, sizeof(uint));
            control = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ControlWords, sizeof(uint));

            // Before the first run the order is the slot order.
            var identity = new uint[bodyCapacity];
            for (int i = 0; i < identity.Length; i++)
                identity[i] = (uint)i;
            valuesB.SetData(identity);
            keysB.SetData(new uint[bodyCapacity]);
        }

        /// <summary>Tiles (thread groups of <see cref="Tile"/> keys) the sort uses for <paramref name="count"/> bodies.</summary>
        public static int TilesFor(int count) => (count + Tile - 1) / Tile;

        /// <summary>
        /// The slots in Morton order: valid in [0, <see cref="SortedCount"/>) after <see cref="Run"/>, the identity order before the first run.
        /// </summary>
        public GraphicsBuffer Order => valuesB;

        /// <summary>The body count of the last <see cref="Run"/>: <see cref="Order"/> is a permutation of the slots below it.</summary>
        public int SortedCount { get; private set; }

        /// <summary>Counts the runs.</summary>
        public int Version { get; private set; }

        /// <summary>True if <see cref="Order"/> is a permutation of the slots [0, <paramref name="bodyCount"/>) (it was made for that many).</summary>
        public bool Covers(int bodyCount) => SortedCount == bodyCount && bodyCount > 0;

        /// <summary>
        /// Sorts the slots [0, bodies.HighWaterMark) by the Morton key of their position (the positions the body arrays hold now). Pending
        /// body edits are applied first.
        /// </summary>
        public void Run(AvbdBodyBuffers bodies)
        {
            if (disposed) throw new ObjectDisposedException(nameof(AvbdMortonSort));
            if (bodies == null) throw new ArgumentNullException(nameof(bodies));
            if (bodies.Capacity != bodyCapacity)
                throw new ArgumentException("The body buffers have capacity " + bodies.Capacity + ", the sort was made for " + bodyCapacity + ".", nameof(bodies));

            bodies.ApplyEdits();
            int count = bodies.HighWaterMark;
            SortedCount = count;
            Version++;
            if (count <= 0)
                return;

            foreach (int kernel in new[] { beginKernel, boundsKernel, keysKernel, countKernel, scanKernel, scatterKernel })
            {
                shader.SetBuffer(kernel, BodyStateId, bodies.State);
                shader.SetBuffer(kernel, BodyMassId, bodies.Mass);
                shader.SetBuffer(kernel, BodyShapeId, bodies.Shape);
                shader.SetBuffer(kernel, BodyMetaId, bodies.Meta);
                shader.SetBuffer(kernel, ControlId, control);
                shader.SetBuffer(kernel, HistogramId, histogram);
            }

            shader.Dispatch(beginKernel, 1, 1, 1);
            AvbdDispatch.Linear(shader, boundsKernel, count);

            // The input of the first pass: key and slot of every body, in buffer A.
            shader.SetBuffer(keysKernel, KeysOutId, keysA);
            shader.SetBuffer(keysKernel, ValuesOutId, valuesA);
            AvbdDispatch.Linear(shader, keysKernel, count);

            int tiles = TilesFor(count);
            shader.SetInt(SortCountId, count);
            shader.SetInt(TilesId, tiles);
            for (int pass = 0; pass < Passes; pass++)
            {
                bool fromA = (pass & 1) == 0;
                GraphicsBuffer keysIn = fromA ? keysA : keysB;
                GraphicsBuffer valuesIn = fromA ? valuesA : valuesB;
                GraphicsBuffer keysOut = fromA ? keysB : keysA;
                GraphicsBuffer valuesOut = fromA ? valuesB : valuesA;

                shader.SetInt(PassId, pass);
                shader.SetBuffer(countKernel, KeysInId, keysIn);
                shader.Dispatch(countKernel, tiles, 1, 1);
                shader.Dispatch(scanKernel, 1, 1, 1);
                shader.SetBuffer(scatterKernel, KeysInId, keysIn);
                shader.SetBuffer(scatterKernel, ValuesInId, valuesIn);
                shader.SetBuffer(scatterKernel, KeysOutId, keysOut);
                shader.SetBuffer(scatterKernel, ValuesOutId, valuesOut);
                shader.Dispatch(scatterKernel, tiles, 1, 1);
            }
        }

        /// <summary>The sorted keys and the slots that go with them, as of the last <see cref="Run"/> (read back from the GPU).</summary>
        public (uint[] keys, uint[] order) ReadSorted()
        {
            if (disposed) throw new ObjectDisposedException(nameof(AvbdMortonSort));
            var keys = new uint[SortedCount];
            var order = new uint[SortedCount];
            if (SortedCount > 0)
            {
                keysB.GetData(keys, 0, 0, SortedCount);
                valuesB.GetData(order, 0, 0, SortedCount);
            }
            return (keys, order);
        }

        /// <summary>Largest bounding-sphere radius among the solved bodies at the last <see cref="Run"/> (read back from the GPU).</summary>
        public float ReadMaxRadius()
        {
            var words = new uint[ControlWords];
            control.GetData(words);
            return BitConverter.ToSingle(BitConverter.GetBytes(words[MaxRadiusWord]), 0);
        }

        public void Dispose()
        {
            if (disposed)
                return;
            disposed = true;
            Release(ref keysA);
            Release(ref keysB);
            Release(ref valuesA);
            Release(ref valuesB);
            Release(ref histogram);
            Release(ref control);
        }

        static void Release(ref GraphicsBuffer buffer)
        {
            buffer?.Dispose();
            buffer = null;
        }
    }
}
