#ifndef MCKRILL_AVBD_INDIRECT_INCLUDED
#define MCKRILL_AVBD_INDIRECT_INCLUDED

#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdCommon.hlsl"

// Indirect dispatches whose size is a count the GPU computed. Mirrored by Runtime/Solver/AvbdIndirect.cs; a test checks the two against
// each other.
//
// The arguments of one dispatch are three words (groups x, y, z) in a raw buffer that DispatchIndirect reads. A dispatch is at most 65,535
// groups in one dimension, so a count of more than AVBD_INDIRECT_ROW_GROUPS groups is laid out in rows of that many groups (x = the row
// width, y = the number of rows); the thread with SV_DispatchThreadID (x, y) handles item y * AVBD_INDIRECT_ROW_GROUPS * AVBD_GROUP_SIZE + x.
// The last group may overshoot the count, so a kernel compares the item with the count it reads from the GPU.

#define AVBD_INDIRECT_ARGS_STRIDE 12 // bytes per dispatch
#define AVBD_INDIRECT_ROW_GROUPS  65535u

// Arguments for `count` items. A count of zero is a 0 x 1 x 1 dispatch, which does nothing.
uint3 AvbdIndirectGroups(uint count)
{
    uint groups = (count + AVBD_GROUP_SIZE - 1u) / AVBD_GROUP_SIZE;
    uint rows = max(1u, (groups + AVBD_INDIRECT_ROW_GROUPS - 1u) / AVBD_INDIRECT_ROW_GROUPS);
    return uint3(min(groups, AVBD_INDIRECT_ROW_GROUPS), rows, 1u);
}

// The item a thread of such a dispatch handles.
uint AvbdIndirectItem(uint3 id)
{
    return id.y * (AVBD_INDIRECT_ROW_GROUPS * AVBD_GROUP_SIZE) + id.x;
}

#endif
