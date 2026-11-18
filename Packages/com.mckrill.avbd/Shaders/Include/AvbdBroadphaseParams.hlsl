#ifndef MCKRILL_AVBD_BROADPHASE_PARAMS_INCLUDED
#define MCKRILL_AVBD_BROADPHASE_PARAMS_INCLUDED

#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdCommon.hlsl"

// Constants and the pair record of stage B (broadphase). Mirrored by Runtime/Broadphase/AvbdBroadphase.cs and
// Runtime/Layout/PairRecords.cs; a test checks them against each other.

// Pair kinds (the "pair type" bins of the pair list).
#define AVBD_PAIR_DYNAMIC_DYNAMIC 0u // both bodies are solved
#define AVBD_PAIR_DYNAMIC_STATIC  1u // one body is solved, the other is static (mass 0) or kinematic

// Words of the control buffer. Cleared at the start of every broadphase run (the histograms too).
#define AVBD_BP_MAX_RADIUS     0 // upper bound of the largest radius among the small bodies (the ones in the grid), as float bits; sizes the coarse cells
#define AVBD_BP_LARGE_COUNT    1 // bodies in the large list (radius above every small body: tested against all, see AvbdBroadphase.compute)
#define AVBD_BP_DYNAMIC_PAIRS  2 // pairs found in the dynamic-dynamic bin. May exceed the bin's capacity: then the pairs beyond it are dropped
#define AVBD_BP_STATIC_PAIRS   3 // the same for the dynamic-static bin
#define AVBD_BP_SMALL_BIN      4 // radius bin of the largest small body: a body above it is large
#define AVBD_BP_FINE_BIN       5 // radius bin of the largest body of the fine level (the same as AVBD_BP_SMALL_BIN with one level)
#define AVBD_BP_TWO_LEVELS     6 // 1 if the grid has a fine and a coarse level
#define AVBD_BP_COARSE_CELL    7 // cell size of the coarse level, as float bits
#define AVBD_BP_FINE_CELL      8 // cell size of the fine level, as float bits (the coarse cell size with one level)
#define AVBD_BP_LIVE_COUNT     9 // bodies in the radius statistics (live, finite radius)
#define AVBD_BP_SOLVED_COUNT  10 // the solved ones among them
#define AVBD_BP_STAT_WORDS    16 // words before the histograms
#define AVBD_BP_HIST_ALL      16  // = AVBD_BP_STAT_WORDS: histogram of the radii of all live bodies
#define AVBD_BP_HIST_SOLVED   656 // = AVBD_BP_STAT_WORDS + AVBD_BP_RADIUS_BINS: histogram of the radii of the solved bodies
#define AVBD_BP_CONTROL_WORDS 1296 // = AVBD_BP_STAT_WORDS + 2 * AVBD_BP_RADIUS_BINS

// Radii are binned on a log scale for the classification (see BpClassify): AVBD_BP_BINS_PER_OCTAVE bins per doubling, from 2^-AVBD_BP_BIN_OCTAVES
// to 2^AVBD_BP_BIN_OCTAVES meters; a radius outside goes into the first or last bin. Doubling a radius is a shift by AVBD_BP_BINS_PER_OCTAVE bins.
#define AVBD_BP_BINS_PER_OCTAVE 16
#define AVBD_BP_BIN_OCTAVES     20
#define AVBD_BP_RADIUS_BINS     640 // = 2 * AVBD_BP_BIN_OCTAVES * AVBD_BP_BINS_PER_OCTAVE

// Bodies above this factor times the median radius are large, up to AVBD_BP_MAX_LARGE of the largest (three-avbd's LARGE_FACTOR and MAX_LARGE):
// a large body is tested against every body and does not size the cells.
#define AVBD_BP_LARGE_FACTOR_BINS 16 // = AVBD_BP_BINS_PER_OCTAVE: a factor of 2 is one octave
#define AVBD_BP_MAX_LARGE        64

// Bits of _BodyCell.w, the grid word of a body. AVBD_INVALID: the body is not in the grid and not in the large list (free, not finite).
#define AVBD_BP_STATIC_FLAG 0x80000000u // a static (or kinematic) body
#define AVBD_BP_LARGE_FLAG  0x00000002u // a large body (in the large list, not in the grid)
#define AVBD_BP_COARSE_FLAG 0x00000001u // a body of the coarse level

// The grid is a hash table of AVBD_BP_SCAN_GROUP-aligned size, counted into per-cell lists (counting sort) and scanned in blocks of this many cells.
#define AVBD_BP_SCAN_GROUP 1024

// Cells are a little larger than the largest possible contact distance (twice the largest radius of a level), so that two spheres that
// overlap are always in the same or in adjacent cells, even after the rounding of position / cell size.
#define AVBD_BP_CELL_MARGIN 1.01

// A candidate pair is dropped when the world bounding boxes, or a face axis of either box, separate the two boxes by more than this
// (meters). The narrow phase finds no contact for a pair as soon as any axis of its separating-axis test shows a gap above zero (a face
// axis is one of them; a gap between the world bounding boxes means the boxes are disjoint, which the 15 axes find too), so a pair dropped
// here is one the narrow phase would drop. The pad keeps the rounding of the two computations out of it.
#define AVBD_BP_SEPARATION_PAD 0.001

// 16 B. One candidate pair. Body A is the body with the higher slot, like the reference solver's pair loop.
struct PairRecord
{
    uint4 pair; // x = body A, y = body B, z = AVBD_PAIR_*, w unused (zero)
};

#endif // MCKRILL_AVBD_BROADPHASE_PARAMS_INCLUDED
