using System.Collections.Generic;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// Greedy first-fit graph coloring of the dynamic bodies, used by <see cref="GaussSeidelMode.Colored"/>.
    /// Two dynamic bodies are adjacent if a force that affects dynamics joins them. Static bodies,
    /// world-anchored joints and IgnoreCollision forces create no edges. Bodies are visited newest
    /// first (the order upstream's solver visits them), and each takes the lowest color not used by an
    /// already-colored neighbor, so the result is deterministic.
    /// </summary>
    public static class GraphColoring
    {
        /// <summary>
        /// Colors the bodies. Sets <see cref="AvbdBody.Color"/> on every body (-1 for static ones) and
        /// fills <paramref name="groups"/> with one list per color, color 0 first, each list in visit order.
        /// Returns the number of colors.
        /// </summary>
        public static int Color(IReadOnlyList<AvbdBody> bodiesInCreationOrder, List<List<AvbdBody>> groups)
        {
            groups.Clear();
            for (int i = 0; i < bodiesInCreationOrder.Count; i++)
                bodiesInCreationOrder[i].Color = -1;

            var used = new HashSet<int>();

            for (int i = bodiesInCreationOrder.Count - 1; i >= 0; i--)
            {
                AvbdBody body = bodiesInCreationOrder[i];
                if (!body.IsDynamic)
                    continue;

                used.Clear();
                for (int k = 0; k < body.Forces.Count; k++)
                {
                    AvbdForce f = body.Forces[k];
                    if (!f.AffectsDynamics)
                        continue;

                    AvbdBody other = f.BodyA == body ? f.BodyB : f.BodyA;
                    if (other == null || !other.IsDynamic || other.Color < 0)
                        continue;

                    used.Add(other.Color);
                }

                int color = 0;
                while (used.Contains(color))
                    color++;

                body.Color = color;
                while (groups.Count <= color)
                    groups.Add(new List<AvbdBody>());
                groups[color].Add(body);
            }

            return groups.Count;
        }
    }
}
