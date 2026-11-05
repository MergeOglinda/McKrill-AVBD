#ifndef MCKRILL_AVBD_COMMON_INCLUDED
#define MCKRILL_AVBD_COMMON_INCLUDED

// Shared definitions for all AVBD compute shaders.
// Include by package path:
//   #include "Packages/com.mckrill.avbd/Shaders/Include/AvbdCommon.hlsl"

// Thread group size used by 1D kernels.
#define AVBD_GROUP_SIZE 64

// Known pattern written by the smoke kernel and checked by the smoke test.
#define AVBD_SMOKE_PATTERN(i) ((i) * 3u + 7u)

#endif // MCKRILL_AVBD_COMMON_INCLUDED
