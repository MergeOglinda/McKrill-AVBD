#ifndef MCKRILL_AVBD_BROADPHASE_PARAMS_INCLUDED
#define MCKRILL_AVBD_BROADPHASE_PARAMS_INCLUDED

#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdCommon.hlsl"

// Constants and the pair record of stage B (broadphase). Mirrored by Runtime/Broadphase/AvbdBroadphase.cs and
// Runtime/Layout/PairRecords.cs; a test checks them against each other.

// Pair kinds (the "pair type" bins of the pair list).
#define AVBD_PAIR_DYNAMIC_DYNAMIC 0u // both bodies are solved
#define AVBD_PAIR_DYNAMIC_STATIC  1u // one body is solved, the other is static (mass 0) or kinematic

// Words of the control buffer. Cleared at the start of every broadphase run.
#define AVBD_BP_MAX_RADIUS     0 // largest bounding-sphere radius of a solved body, as float bits (non-negative floats order like uints)
#define AVBD_BP_STATIC_COUNT   1 // bodies in the static list
#define AVBD_BP_DYNAMIC_PAIRS  2 // pairs found in the dynamic-dynamic bin. May exceed the bin's capacity: then the pairs beyond it are dropped
#define AVBD_BP_STATIC_PAIRS   3 // the same for the dynamic-static bin
#define AVBD_BP_CONTROL_WORDS  4

// Cells are a little larger than the largest possible contact distance (twice the largest radius), so that two spheres that
// overlap are always in the same or in adjacent cells, even after the rounding of position / cell size.
#define AVBD_BP_CELL_MARGIN 1.01

// 16 B. One candidate pair. Body A is the body with the higher slot, like the reference solver's pair loop.
struct PairRecord
{
    uint4 pair; // x = body A, y = body B, z = AVBD_PAIR_*, w unused (zero)
};

#endif // MCKRILL_AVBD_BROADPHASE_PARAMS_INCLUDED
