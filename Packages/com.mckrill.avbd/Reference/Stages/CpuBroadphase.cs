using System.Collections.Generic;
using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle for stage B: the set of pairs that <c>AvbdBroadphase.compute</c> must find. Brute force over all pairs of live bodies,
    /// with the same rules as the shader (and as the reference solver's broadphase, minus the pairs of two static bodies):
    ///   a pair needs at least one solved body (not free, not kinematic, mass above zero),
    ///   the bounding spheres (center = position, radius = |size| / 2) touch: |dp|^2 &lt;= (rA + rB)^2,
    ///   the shapes are not visibly apart (<see cref="BoxesSeparated(float3, quaternion, float3, float3, quaternion, float3)"/>: world bounding boxes, face axes; a sphere and a box: the exact distance, <see cref="SphereBoxSeparated"/>; two spheres: no test),
    ///   no joint or spring joins the two bodies (a broken one, as far as the records say, no longer does, and contacts do not count: this is the narrowphase's exclusion),
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
            uint flags = bodies.Flags[slot];
            if ((flags & (uint)BodyFlags.Free) != 0u)
                return BodyClass.Free;
            if ((flags & (uint)BodyFlags.Kinematic) == 0u && bodies.Mass[slot].momentMass.w > 0f)
                return BodyClass.Solved;
            return BodyClass.Static;
        }

        /// <summary>True if the body is a sphere (the shape kind in the flags word).</summary>
        public static bool IsSphere(CpuBodyArrays bodies, int slot) => BodyFlagLayout.KindOf(bodies.Flags[slot]) == ShapeKind.Sphere;

        /// <summary>The radius of the bounding sphere: half the diagonal of a box, the radius of a sphere (half of size.x, its diameter).</summary>
        public static float Radius(CpuBodyArrays bodies, int slot)
        {
            float3 size = bodies.Shape[slot].sizeFriction.xyz;
            return IsSphere(bodies, slot) ? size.x * 0.5f : math.length(size * 0.5f);
        }

        /// <summary>
        /// Mirrors BoxesSeparated in the shader: true if the boxes (center, rotation, full sizes) are apart by more than
        /// <see cref="AvbdBroadphase.SeparationPad"/> along a world axis, or along a face axis of either box. The narrow phase finds no contact
        /// for such a pair. A NaN never separates.
        /// </summary>
        public static bool BoxesSeparated(float3 positionA, quaternion rotationA, float3 sizeA, float3 positionB, quaternion rotationB, float3 sizeB)
        {
            float pad = AvbdBroadphase.SeparationPad;
            float3 halfA = sizeA * 0.5f;
            float3 halfB = sizeB * 0.5f;
            float3 a0 = AvbdQuat.Rotate(rotationA, new float3(1f, 0f, 0f));
            float3 a1 = AvbdQuat.Rotate(rotationA, new float3(0f, 1f, 0f));
            float3 a2 = AvbdQuat.Rotate(rotationA, new float3(0f, 0f, 1f));
            float3 b0 = AvbdQuat.Rotate(rotationB, new float3(1f, 0f, 0f));
            float3 b1 = AvbdQuat.Rotate(rotationB, new float3(0f, 1f, 0f));
            float3 b2 = AvbdQuat.Rotate(rotationB, new float3(0f, 0f, 1f));

            float3 aabbA = math.abs(a0) * halfA.x + math.abs(a1) * halfA.y + math.abs(a2) * halfA.z;
            float3 aabbB = math.abs(b0) * halfB.x + math.abs(b1) * halfB.y + math.abs(b2) * halfB.z;
            float3 d = positionA - positionB;
            if (math.any(math.abs(d) > aabbA + aabbB + pad))
                return true;

            float Extent(float3 h, float3 x0, float3 x1, float3 x2, float3 n) => h.x * math.abs(math.dot(n, x0)) + h.y * math.abs(math.dot(n, x1)) + h.z * math.abs(math.dot(n, x2));
            bool Separates(float3 n) => math.abs(math.dot(d, n)) - Extent(halfA, a0, a1, a2, n) - Extent(halfB, b0, b1, b2, n) > pad;

            return Separates(a0) || Separates(a1) || Separates(a2) || Separates(b0) || Separates(b1) || Separates(b2);
        }

        /// <summary>
        /// Mirrors SphereBoxSeparated in the shader: true if a sphere (center, radius) is apart from a box (center, rotation, full sizes) by more than
        /// <see cref="AvbdBroadphase.SeparationPad"/>: the distance from the center to the closest point of the box against the radius. A NaN never separates.
        /// </summary>
        public static bool SphereBoxSeparated(float3 sphereCenter, float radius, float3 boxCenter, quaternion boxRotation, float3 boxSize)
        {
            float3 half = boxSize * 0.5f;
            float3 d = sphereCenter - boxCenter;
            float3 p = new float3(math.dot(d, AvbdQuat.Rotate(boxRotation, new float3(1f, 0f, 0f))),
                                  math.dot(d, AvbdQuat.Rotate(boxRotation, new float3(0f, 1f, 0f))),
                                  math.dot(d, AvbdQuat.Rotate(boxRotation, new float3(0f, 0f, 1f))));
            float3 e = p - math.clamp(p, -half, half);
            float reach = radius + AvbdBroadphase.SeparationPad;
            return math.dot(e, e) > reach * reach;
        }

        /// <summary>
        /// The shape test of the shader's TestPair on the bodies as the GPU holds them (position and rotation of the state record, full size): two spheres are never
        /// separated by it (the sphere test is the exact one), a sphere and a box are tested exactly, two boxes by <see cref="BoxesSeparated(float3, quaternion, float3, float3, quaternion, float3)"/>.
        /// </summary>
        public static bool BoxesSeparated(CpuBodyArrays bodies, int a, int b)
        {
            bool sphereA = IsSphere(bodies, a);
            bool sphereB = IsSphere(bodies, b);
            if (sphereA && sphereB)
                return false;
            if (sphereA)
                return SphereBoxSeparated(bodies.State[a].position.xyz, Radius(bodies, a), bodies.State[b].position.xyz, new quaternion(bodies.State[b].rotation), bodies.Shape[b].sizeFriction.xyz);
            if (sphereB)
                return SphereBoxSeparated(bodies.State[b].position.xyz, Radius(bodies, b), bodies.State[a].position.xyz, new quaternion(bodies.State[a].rotation), bodies.Shape[a].sizeFriction.xyz);
            return BoxesSeparated(bodies.State[a].position.xyz, new quaternion(bodies.State[a].rotation), bodies.Shape[a].sizeFriction.xyz,
                                  bodies.State[b].position.xyz, new quaternion(bodies.State[b].rotation), bodies.Shape[b].sizeFriction.xyz);
        }

        /// <summary>A pair as one number, for sets and comparisons. Slots must be below 2^30.</summary>
        public static ulong Key(uint bodyA, uint bodyB, PairKind kind) => ((ulong)bodyA << 34) | ((ulong)bodyB << 2) | (ulong)kind;

        public static ulong Key(PairRecord pair) => Key(pair.BodyA, pair.BodyB, pair.Kind);

        public static string Describe(ulong key) => "(" + (key >> 34) + ", " + ((key >> 2) & 0xFFFFFFFFu) + ", " + (PairKind)(key & 3u) + ")";

        /// <summary>What BpClassify decides: see <see cref="Classification"/>.</summary>
        public struct GridModel
        {
            public int LiveCount;
            public int SolvedCount;
            public int SmallBin;
            public int FineBin;
            public bool TwoLevels;
            public float MaxSmallRadius;
            public float CoarseCell;
            public float FineCell;
            public int LargeCount;
        }

        const int Bins = AvbdBroadphase.RadiusBins;

        /// <summary>Mirrors RadiusBin in the shader.</summary>
        public static int RadiusBin(float radius)
        {
            float scaled = math.log2(math.max(radius, 1.0e-30f)) * AvbdBroadphase.BinsPerOctave + AvbdBroadphase.BinOctaves * AvbdBroadphase.BinsPerOctave;
            return (int)math.clamp(math.floor(scaled), 0f, Bins - 1);
        }

        /// <summary>Mirrors BinUpper in the shader.</summary>
        public static float BinUpper(int bin) => math.exp2((bin + 1f) / AvbdBroadphase.BinsPerOctave - AvbdBroadphase.BinOctaves) * 1.001f;

        static int RankBin(int[] histogram, int rank)
        {
            int cumulative = 0;
            for (int bin = 0; bin < Bins; bin++)
            {
                cumulative += histogram[bin];
                if (cumulative > rank)
                    return bin;
            }
            return Bins - 1;
        }

        static int HighestBinBelow(int[] histogram, int limit)
        {
            for (int bin = System.Math.Min(limit, Bins - 1); bin >= 0; bin--)
                if (histogram[bin] != 0)
                    return bin;
            return 0;
        }

        // The largest bin that still holds a small body (three-avbd's largestSmall, on bins).
        static int SmallBin(int[] histogram, int count)
        {
            int cut = System.Math.Min(RankBin(histogram, count >> 1) + AvbdBroadphase.LargeFactorBins, Bins - 1);
            if (count > AvbdBroadphase.MaxLarge)
                cut = System.Math.Max(cut, RankBin(histogram, count - AvbdBroadphase.MaxLarge - 1));
            return HighestBinBelow(histogram, cut);
        }

        /// <summary>
        /// Mirrors BpHistogram and BpClassify: which radius bin is the largest of the small bodies, where the fine level ends, whether there are two
        /// levels, the cell sizes, and how many bodies are large. (three-avbd's uploadStatics, with the median and the cut-offs taken on bins of a
        /// sixteenth of an octave, so that the GPU needs no sort and no readback.)
        /// </summary>
        public static GridModel Classification(CpuBodyArrays bodies, int bodyCount)
        {
            var all = new int[Bins];
            var solved = new int[Bins];
            var model = new GridModel();
            for (int s = 0; s < bodyCount; s++)
            {
                BodyClass kind = Classify(bodies, s);
                float radius = Radius(bodies, s);
                if (kind == BodyClass.Free || !float.IsFinite(radius))
                    continue;
                int bin = RadiusBin(radius);
                all[bin]++;
                model.LiveCount++;
                if (kind == BodyClass.Solved)
                {
                    solved[bin]++;
                    model.SolvedCount++;
                }
            }
            if (model.LiveCount == 0)
            {
                model.CoarseCell = 1f;
                model.FineCell = 1f;
                return model;
            }

            int smallBin = SmallBin(all, model.LiveCount);
            int targetBin = smallBin;
            if (model.SolvedCount > 0)
                targetBin = System.Math.Min(smallBin, SmallBin(solved, model.SolvedCount) + AvbdBroadphase.LargeFactorBins);
            int fineBin = HighestBinBelow(all, targetBin);
            bool two = fineBin + AvbdBroadphase.LargeFactorBins < smallBin;

            model.SmallBin = smallBin;
            model.FineBin = two ? fineBin : smallBin;
            model.TwoLevels = two;
            model.MaxSmallRadius = BinUpper(smallBin);
            model.CoarseCell = math.max(2f * BinUpper(smallBin) * AvbdBroadphase.CellMargin, 1.0e-4f);
            model.FineCell = two ? math.max(2f * BinUpper(fineBin) * AvbdBroadphase.CellMargin, 1.0e-4f) : model.CoarseCell;
            for (int bin = smallBin + 1; bin < Bins; bin++)
                model.LargeCount += all[bin];
            return model;
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
                if (ConstraintTopology.IsBroken(constraints[i]))
                    continue; // the reference solver removes a broken joint at the next step's initialize
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
                    if (BoxesSeparated(bodies, a, b))
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
