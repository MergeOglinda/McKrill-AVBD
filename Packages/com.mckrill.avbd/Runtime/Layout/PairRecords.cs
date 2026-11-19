using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>Kind of a candidate pair, the bin it is stored in. Mirrors AVBD_PAIR_* in AvbdBroadphaseParams.hlsl.</summary>
    internal enum PairKind : uint
    {
        /// <summary>Both bodies are solved (not free, not kinematic, mass above zero).</summary>
        DynamicDynamic = 0,

        /// <summary>One body is solved, the other is live but not solved (mass 0 or kinematic).</summary>
        DynamicStatic = 1,
    }

    /// <summary>
    /// 16 B. One candidate pair of the broadphase (stage B), read by the narrowphase (stage N).
    /// Mirrors PairRecord in AvbdBroadphaseParams.hlsl.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct PairRecord
    {
        /// <summary>x = body A (the higher slot), y = body B (the lower slot), z = <see cref="PairKind"/>, w = <see cref="AvbdBroadphase.PairSphereA"/> and <see cref="AvbdBroadphase.PairSphereB"/> bits (zero for two boxes).</summary>
        public uint4 pair;

        public uint BodyA => pair.x;
        public uint BodyB => pair.y;
        public PairKind Kind => (PairKind)pair.z;
        public bool BodyAIsSphere => (pair.w & AvbdBroadphase.PairSphereA) != 0u;
        public bool BodyBIsSphere => (pair.w & AvbdBroadphase.PairSphereB) != 0u;
    }
}
