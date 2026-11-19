#ifndef MCKRILL_AVBD_SPHERE_COLLISION_INCLUDED
#define MCKRILL_AVBD_SPHERE_COLLISION_INCLUDED

#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdContacts.hlsl"
#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdBoxCollision.hlsl"

// Narrow phase for spheres. (The dispatch over the shape kinds of a pair, AvbdCollideShapes, is in AvbdNarrowphase.compute.) A port of three-avbd's sphereSphere and sphereBox (wgsl-collision.ts), into
// the same statics as AvbdCollideBoxes (g_contactCount, g_contactRA, g_contactRB, g_contactKey, g_normalBA and the poses). A pair of a sphere and
// anything has at most one contact, with the key AVBD_SPHERE_FEATURE, so the old contact of the pair is always the one to merge with.
// Like the boxes, a pair that is apart has no contact: a sphere contact exists when the surfaces overlap or touch, and the collision margin only
// shifts the constraint value of the contact (AvbdInitContacts).
// The radius of a sphere is half of its size.x. Vector-only, like AvbdBoxCollision.hlsl, so that Tools/HlslParity can compile it as C++.

// Sphere against sphere. The normal is along the line of centers, from B to A.
int AvbdCollideSpheres(float3 posA, float4 rotA, float radiusA, float3 posB, float4 rotB, float radiusB)
{
    g_contactCount = 0;
    g_posA = posA;
    g_rotA = rotA;
    g_posB = posB;
    g_rotB = rotB;

    float3 d = posB - posA;
    float dist = length(d);
    if (!(dist <= radiusA + radiusB))
        return 0; // apart (a NaN is apart too)

    float3 nAB = dist > 0.0 ? d / dist : float3(0.0, 0.0, 1.0);
    g_normalBA = -nAB;
    AvbdAddContact(posA + nAB * radiusA, posB - nAB * radiusB, AVBD_SPHERE_FEATURE);
    return g_contactCount;
}

// Sphere S against box X. Fills the statics for the pair (A, B) = (sphere, box) if sphereIsA, else (box, sphere). The closest point of the box is
// the contact point on the box, or, when the center of the sphere is inside the box, the point of the face of least penetration.
int AvbdCollideSphereBox(bool sphereIsA, float3 posA, float4 rotA, float3 sizeA, float3 posB, float4 rotB, float3 sizeB)
{
    g_contactCount = 0;
    g_posA = posA;
    g_rotA = rotA;
    g_posB = posB;
    g_rotB = rotB;

    float3 sphereCenter = sphereIsA ? posA : posB;
    float radius = (sphereIsA ? sizeA.x : sizeB.x) * 0.5;
    float3 boxCenter = sphereIsA ? posB : posA;
    float4 boxRotation = sphereIsA ? rotB : rotA;
    float3 boxHalf = (sphereIsA ? sizeB : sizeA) * 0.5;

    float3 p = AvbdQuatRotate(AvbdQuatConjugate(boxRotation), sphereCenter - boxCenter);
    float3 q = clamp(p, -boxHalf, boxHalf);
    float3 onBoxLocal = q;
    float3 nLocal; // from the box toward the sphere, in the box frame

    if (any(p != q))
    {
        float3 d = p - q;
        float dist = length(d);
        if (!(dist <= radius))
            return 0; // apart
        nLocal = d / dist;
    }
    else
    {
        // The center is inside the box: out through the face of least penetration.
        float3 depth = boxHalf - abs(p);
        // The axis of the least depth, the lower axis on a tie (x, then y if it is less deep than x, then z if it is less deep than both). One-hot
        // vector and dot products instead of a runtime index: a vector indexed by a variable becomes a local array on some backends (and DX12 refuses
        // it on the left of an assignment), which can cost occupancy in the whole kernel.
        bool yBeatsX = depth.y < depth.x;
        bool zBeatsBest = depth.z < (yBeatsX ? depth.y : depth.x);
        float3 axis = zBeatsBest ? float3(0.0, 0.0, 1.0) : (yBeatsX ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0));
        float s = dot(p, axis) >= 0.0 ? 1.0 : -1.0;
        nLocal = axis * s;
        onBoxLocal = onBoxLocal * (float3(1.0, 1.0, 1.0) - axis) + axis * (s * dot(boxHalf, axis));
    }

    float3 n = AvbdQuatRotate(boxRotation, nLocal);        // from the box toward the sphere
    float3 onBox = boxCenter + AvbdQuatRotate(boxRotation, onBoxLocal);
    float3 onSphere = sphereCenter - n * radius;

    // The normal of a contact points from B to A: from the box toward the sphere if the sphere is A, the other way if it is B.
    if (sphereIsA)
    {
        g_normalBA = n;
        AvbdAddContact(onSphere, onBox, AVBD_SPHERE_FEATURE);
    }
    else
    {
        g_normalBA = -n;
        AvbdAddContact(onBox, onSphere, AVBD_SPHERE_FEATURE);
    }
    return g_contactCount;
}

#endif // MCKRILL_AVBD_SPHERE_COLLISION_INCLUDED
