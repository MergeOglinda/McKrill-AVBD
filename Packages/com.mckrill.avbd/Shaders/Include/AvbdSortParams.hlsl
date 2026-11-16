#ifndef MCKRILL_AVBD_SORT_PARAMS_INCLUDED
#define MCKRILL_AVBD_SORT_PARAMS_INCLUDED

#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdCommon.hlsl"

// Constants of stage S (Morton sort). Mirrored by Runtime/Broadphase/AvbdMortonSort.cs; a test checks the two against each other.

// Words of the control buffer. Cleared at the start of every run.
#define AVBD_SORT_CONTROL_MAX_RADIUS 0 // largest bounding-sphere radius of a solved body, as float bits (like AVBD_BP_MAX_RADIUS)
#define AVBD_SORT_CONTROL_WORDS      4

// The key: AVBD_SORT_AXIS_BITS bits of the grid cell of the body per axis, interleaved (a Morton code of 3 * AVBD_SORT_AXIS_BITS bits).
#define AVBD_SORT_AXIS_BITS 10u
#define AVBD_SORT_KEY_BITS  30u
// The key of a body that is not solved, or whose position is not finite. It is the largest key, so those bodies come last.
#define AVBD_SORT_INVALID_KEY 0x3FFFFFFFu

// The radix sort: AVBD_SORT_PASSES passes of AVBD_SORT_RADIX_BITS bits (AVBD_SORT_PASSES * AVBD_SORT_RADIX_BITS >= AVBD_SORT_KEY_BITS).
#define AVBD_SORT_RADIX_BITS 6u
#define AVBD_SORT_BUCKETS    64u
#define AVBD_SORT_PASSES     5u

// Keys per tile: one thread group of this many threads ranks the keys of a tile, one key per thread.
#define AVBD_SORT_TILE 256u
// Threads of the scan group (one group scans the histograms of all tiles).
#define AVBD_SORT_SCAN_THREADS 1024u
// A dispatch has at most 65,535 groups in one dimension, so at most this many tiles (16,776,960 bodies).
#define AVBD_SORT_MAX_TILES 65535u

#endif // MCKRILL_AVBD_SORT_PARAMS_INCLUDED
