#ifndef MCKRILL_AVBD_CONTACTS_INCLUDED
#define MCKRILL_AVBD_CONTACTS_INCLUDED

#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdCommon.hlsl"

// Contact constants and the layout of the contact region. Mirrors ContactLayout in Runtime/Layout/ConstraintRecords.cs.
//
// Manifolds live in a pool of slots (stage N): slot s has a contact count (_PairCounts[s]), the pair of bodies it belongs to
// (_SlotBodies[s]) and room for AVBD_MAX_CONTACTS contact records at _ContactBase + s * AVBD_MAX_CONTACTS in the constraint array.

#define AVBD_MAX_CONTACTS 8u

#define AVBD_COLLISION_MARGIN  0.01
#define AVBD_STICK_THRESHOLD   0.00001

// Words of the manifold pool's control buffer (_NpControl).
#define AVBD_NP_FREE_TOP      0 // number of free slots: the top of the stack of free slots
#define AVBD_NP_OVERFLOW      1 // candidate pairs dropped because no slot was free, since the buffer was made
#define AVBD_NP_CONTROL_WORDS 2

// Dense index of the pair of slots (a, b) with a > b. No stage uses it any more (it was the manifold slot of the narrow phase over all
// pairs); the CPU oracle still stores its manifolds this way and the topology tests use it.
uint AvbdPairIndex(uint a, uint b)
{
    return (a * (a - 1u)) / 2u + b;
}

// Inverse of AvbdPairIndex: x = the higher slot a, y = the lower slot b. The float square root is corrected by the two loops.
uint2 AvbdPairBodies(uint pair)
{
    uint a = (uint)((1.0 + sqrt(1.0 + 8.0 * (float)pair)) * 0.5);
    while ((a * (a - 1u)) / 2u > pair)
        a--;
    while (((a + 1u) * a) / 2u <= pair)
        a++;
    return uint2(a, pair - (a * (a - 1u)) / 2u);
}

// First tangent of the contact basis whose normal is n (upstream's orthonormal(): rows are the normal, t1 and t2 = cross(n, t1)).
float3 AvbdContactTangent1(float3 n)
{
    float3 t = abs(n.x) > abs(n.z) ? float3(-n.y, n.x, 0.0) : float3(0.0, -n.z, n.y);
    return normalize(t);
}

#endif // MCKRILL_AVBD_CONTACTS_INCLUDED
