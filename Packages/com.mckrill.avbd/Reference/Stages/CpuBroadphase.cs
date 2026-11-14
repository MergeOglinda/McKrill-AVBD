using System.Collections.Generic;
using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle for stage B: the set of pairs that <c>AvbdBroadphase.compute</c> must find. Brute force over all pairs of live bodies,
    /// with the same rules as the shader (and as the reference solver's broadphase, minus the pairs of two static bodies):
    ///   a pair needs at least one solved body (not free, not kinematic, mass above zero),
    ///   the bounding spheres (center = position, radius = |size| / 2) touch: |dp|^2 &lt;= (rA + rB)^2,
    ///   no joint or spring joins the two bodies (a broken one still does, and contacts do not count: this is the narrowphase's exclusion),
    ///   body A is the higher slot; the kind is dynamic-dynamic if both are solved, else dynamic-static.
    /// A body at a non-finite position makes no pairs. The GPU's pair order is unspecified, so a pair set is a set of keys.
    /// </summary>
    internal static class CpuBroadphase
    {
        /// <summary>Mirrors Classify in the shader.</summary>
        public enum BodyClass
        {
            Free,
            Solved,
            Static,
        }

        public static BodyClass Classify(CpuBodyArrays bodies, int slot)
        {
            uint flags = bodies.Meta[slot].ident.x;
            if ((flags & (uint)BodyFlags.Free) != 0u)
                return BodyClass.Free;
            if ((flags & (uint)BodyFlags.Kinematic) == 0u && bodies.Mass[slot].momentMass.w > 0f)
                return BodyClass.Solved;
            return BodyClass.Static;
        }

        public static float Radius(CpuBodyArrays bodies, int slot) => math.length(bodies.Shape[slot].sizeFriction.xyz * 0.5f);

        /// <summary>A pair as one number, for sets and comparisons. Slots must be below 2^30.</summary>
        public static ulong Key(uint bodyA, uint bodyB, PairKind kind) => ((ulong)bodyA << 34) | ((ulong)bodyB << 2) | (ulong)kind;

        public static ulong Key(PairRecord pair) => Key(pair.BodyA, pair.BodyB, pair.Kind);

        public static string Describe(ulong key) => "(" + (key >> 34) + ", " + ((key >> 2) & 0xFFFFFFFFu) + ", " + (PairKind)(key & 3u) + ")";

        /// <summary>Largest radius among the solved bodies; zero if there are none.</summary>
        public static float MaxRadius(CpuBodyArrays bodies, int bodyCount)
        {
            float max = 0f;
            for (int s = 0; s < bodyCount; s++)
                if (Classify(bodies, s) == BodyClass.Solved)
                {
                    float r = Radius(bodies, s);
                    if (float.IsFinite(r) && r > max)
                        max = r;
                }
            return max;
        }

        /// <param name="constraints">Constraint records; contacts and the ones with a world end exclude nothing.</param>
        public static SortedSet<ulong> Pairs(CpuBodyArrays bodies, int bodyCount, ConstraintRecord[] constraints, int constraintCount)
        {
            var joined = new HashSet<ulong>();
            for (int i = 0; i < constraintCount; i++)
            {
                uint a = constraints[i].ident.x;
                uint b = constraints[i].ident.y;
                if (a == AvbdConst.Invalid || b == AvbdConst.Invalid)
                    continue;
                if (ConstraintFlagLayout.KindOf(constraints[i].ident.z) == ConstraintKind.Contact)
                    continue;
                joined.Add(Key(System.Math.Max(a, b), System.Math.Min(a, b), PairKind.DynamicDynamic));
            }

            var result = new SortedSet<ulong>();
            for (int a = bodyCount - 1; a >= 0; a--)
            {
                BodyClass classA = Classify(bodies, a);
                if (classA == BodyClass.Free)
                    continue;
                float3 positionA = bodies.State[a].position.xyz;
                if (!math.all(math.isfinite(positionA)))
                    continue;

                for (int b = a - 1; b >= 0; b--)
                {
                    BodyClass classB = Classify(bodies, b);
                    if (classB == BodyClass.Free || (classA != BodyClass.Solved && classB != BodyClass.Solved))
                        continue;
                    float3 positionB = bodies.State[b].position.xyz;
                    if (!math.all(math.isfinite(positionB)))
                        continue;

                    float3 dp = positionA - positionB;
                    float r = Radius(bodies, a) + Radius(bodies, b);
                    if (!(math.dot(dp, dp) <= r * r))
                        continue;
                    if (joined.Contains(Key((uint)a, (uint)b, PairKind.DynamicDynamic)))
                        continue;

                    PairKind kind = classA == BodyClass.Solved && classB == BodyClass.Solved ? PairKind.DynamicDynamic : PairKind.DynamicStatic;
                    result.Add(Key((uint)a, (uint)b, kind));
                }
            }
            return result;
        }
    }
}
