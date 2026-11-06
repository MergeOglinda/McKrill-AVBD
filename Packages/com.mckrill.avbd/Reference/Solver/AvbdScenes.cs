using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// Scenes copied number-for-number from upstream avbd-demo3d scenes.h, for parity tests and
    /// benchmarks. They live in upstream's z-up frame and assume the default gravity (0, 0, -10).
    /// Each builder clears the solver first, like upstream.
    /// </summary>
    public static class AvbdScenes
    {
        /// <summary>A big static ground slab and one box dropped from z = 4.</summary>
        public static void Ground(ReferenceSolver s)
        {
            s.Clear();
            s.AddBody(new float3(100f, 100f, 1f), 0f, 0.5f, new float3(0f, 0f, 0f), new float3(0f, 0f, 0f));
            s.AddBody(new float3(1f, 1f, 1f), 1f, 0.5f, new float3(0f, 0f, 4f));
        }

        /// <summary>Ten unit boxes in a column above a ground slab, each starting 0.5 above the one below.</summary>
        public static void Stack(ReferenceSolver s)
        {
            s.Clear();
            s.AddBody(new float3(100f, 100f, 1f), 0f, 0.5f, new float3(0f, 0f, 0f));
            for (int i = 0; i < 10; i++)
                s.AddBody(new float3(1f, 1f, 1f), 1f, 0.5f, new float3(0f, 0f, i * 1.5f + 1f));
        }

        /// <summary>A triangular pyramid of 1 x 0.5 x 0.5 boxes, <paramref name="size"/> boxes along the base (upstream: 16).</summary>
        public static void Pyramid(ReferenceSolver s, int size = 16)
        {
            s.Clear();
            s.AddBody(new float3(100f, 100f, 1f), 0f, 0.5f, new float3(0f, 0f, -0.5f));

            for (int y = 0; y < size; y++)
                for (int x = 0; x < size - y; x++)
                    s.AddBody(new float3(1f, 0.5f, 0.5f), 1f, 0.5f,
                        new float3(x * 1.01f + y * 0.5f - size / 2f, 0f, y * 0.85f + 0.5f));
        }

        /// <summary>A 20-link chain of hinged boxes hanging from a static first link, with a far-away ground.</summary>
        public static void Rope(ReferenceSolver s)
        {
            s.Clear();
            s.AddBody(new float3(100f, 100f, 1f), 0f, 0.5f, new float3(0f, 0f, -20f));

            AvbdBody prev = null;
            for (int i = 0; i < 20; i++)
            {
                AvbdBody curr = s.AddBody(new float3(1f, 0.5f, 0.5f), i == 0 ? 0f : 1f, 0.5f, new float3((float)i, 0f, 10f));
                if (prev != null)
                    s.AddJoint(prev, curr, new float3(0.5f, 0f, 0f), new float3(-0.5f, 0f, 0f));
                prev = curr;
            }
        }

        /// <summary>A 2x2x2 block hanging from a static anchor on a spring (stiffness 100, rest length 4).</summary>
        public static void Spring(ReferenceSolver s)
        {
            s.Clear();
            s.AddBody(new float3(100f, 100f, 1f), 0f, 0.5f, new float3(0f, 0f, 0f));

            AvbdBody anchor = s.AddBody(new float3(1f, 1f, 1f), 0f, 0.5f, new float3(0f, 0f, 14f));
            AvbdBody block = s.AddBody(new float3(2f, 2f, 2f), 1f, 0.5f, new float3(0f, 0f, 8f));
            s.AddSpring(anchor, block, float3.zero, float3.zero, 100f, 4f);
        }
    }
}
