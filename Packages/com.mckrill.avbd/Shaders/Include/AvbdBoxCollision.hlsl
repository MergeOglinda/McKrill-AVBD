#ifndef MCKRILL_AVBD_BOX_COLLISION_INCLUDED
#define MCKRILL_AVBD_BOX_COLLISION_INCLUDED

#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdQuaternion.hlsl"

// Box-box narrow phase: separating-axis test over 15 axes, then either face clipping (up to 8 contacts) or one edge-edge contact. A port of
// Reference/Solver/BoxCollision.cs, which ports upstream's collide.cpp; the feature keys are the same so that warm starting matches.
//
// Written with vectors only and no out parameters, so that Tools/HlslParity can compile this file as C++ and compare it with upstream.
// The result is left in the g_ variables below, which are per-thread (static) in a compute shader.

#define AVBD_BOX_MAX_CONTACTS 8
#define AVBD_POLY_VERTS 16

#define AVBD_SAT_AXIS_EPSILON    1.0e-6
#define AVBD_PLANE_EPSILON       1.0e-5
#define AVBD_CONTACT_MERGE_DIST_SQ 1.0e-6

// Packed into the top byte of the feature key, so they must not change.
#define AVBD_AXIS_FACE_A 0
#define AVBD_AXIS_FACE_B 1
#define AVBD_AXIS_EDGE   2

struct AvbdObb
{
    float3 center;
    float3 halfExtent;
    float3 axis0;
    float3 axis1;
    float3 axis2;
};

struct AvbdSatAxis
{
    int type;
    int indexA;
    int indexB;
    float separation;
    float3 normalAB;
    bool valid;
    bool gap; // set when the axis separates the boxes
};

struct AvbdFaceAxes
{
    float3 u;
    float3 v;
    float extentU;
    float extentV;
};

// Input poses of the pair being collided (set by AvbdCollideBoxes).
static float3 g_posA;
static float4 g_rotA;
static float3 g_posB;
static float4 g_rotB;

// Output: contacts in A's and B's body space, their feature keys, and the contact normal pointing from B to A (the first row of the basis).
static int    g_contactCount;
static int    g_contactKey[AVBD_BOX_MAX_CONTACTS];
static float3 g_contactRA[AVBD_BOX_MAX_CONTACTS];
static float3 g_contactRB[AVBD_BOX_MAX_CONTACTS];
static float3 g_contactMid[AVBD_BOX_MAX_CONTACTS];
static float3 g_normalBA;

// Clipping scratch: two polygons of AVBD_POLY_VERTS vertices, the first at 0 and the second at AVBD_POLY_VERTS.
static float3 g_clip[2 * AVBD_POLY_VERTS];

AvbdObb AvbdMakeObb(float3 position, float4 rotation, float3 size)
{
    AvbdObb box;
    box.center = position;
    box.halfExtent = size * 0.5;
    box.axis0 = AvbdQuatRotate(rotation, float3(1.0, 0.0, 0.0));
    box.axis1 = AvbdQuatRotate(rotation, float3(0.0, 1.0, 0.0));
    box.axis2 = AvbdQuatRotate(rotation, float3(0.0, 0.0, 1.0));
    return box;
}

float3 AvbdObbAxis(AvbdObb box, int i)
{
    return i == 0 ? box.axis0 : (i == 1 ? box.axis1 : box.axis2);
}

float AvbdObbHalf(AvbdObb box, int i)
{
    return i == 0 ? box.halfExtent.x : (i == 1 ? box.halfExtent.y : box.halfExtent.z);
}

float AvbdAbsDot(float3 a, float3 b)
{
    return abs(dot(a, b));
}

float3 AvbdSupportPoint(AvbdObb box, float3 dir)
{
    float sx = dot(dir, box.axis0) >= 0.0 ? 1.0 : -1.0;
    float sy = dot(dir, box.axis1) >= 0.0 ? 1.0 : -1.0;
    float sz = dot(dir, box.axis2) >= 0.0 ? 1.0 : -1.0;

    return box.center
        + box.axis0 * (box.halfExtent.x * sx)
        + box.axis1 * (box.halfExtent.y * sy)
        + box.axis2 * (box.halfExtent.z * sz);
}

AvbdFaceAxes AvbdGetFaceAxes(AvbdObb box, int axisIndex)
{
    AvbdFaceAxes f;
    if (axisIndex == 0)
    {
        f.u = box.axis1;
        f.v = box.axis2;
        f.extentU = box.halfExtent.y;
        f.extentV = box.halfExtent.z;
    }
    else if (axisIndex == 1)
    {
        f.u = box.axis0;
        f.v = box.axis2;
        f.extentU = box.halfExtent.x;
        f.extentV = box.halfExtent.z;
    }
    else
    {
        f.u = box.axis0;
        f.v = box.axis1;
        f.extentU = box.halfExtent.x;
        f.extentV = box.halfExtent.y;
    }
    return f;
}

int AvbdChooseIncidentFaceAxis(AvbdObb box, float3 referenceNormal)
{
    int axis = 0;
    float best = -3.402823466e+38;

    for (int i = 0; i < 3; ++i)
    {
        float d = AvbdAbsDot(AvbdObbAxis(box, i), referenceNormal);
        if (d > best)
        {
            best = d;
            axis = i;
        }
    }

    return axis;
}

// Sutherland-Hodgman clip of the polygon at g_clip[srcBase ..] against the half-space dot(planeNormal, p) <= planeOffset. The result goes to
// the other half of g_clip. Returns its vertex count.
int AvbdClipPolygonAgainstPlane(int srcBase, int inCount, float3 planeNormal, float planeOffset)
{
    if (inCount <= 0)
        return 0;

    int dstBase = AVBD_POLY_VERTS - srcBase;
    int outCount = 0;
    float3 a = g_clip[srcBase + inCount - 1];
    float da = dot(planeNormal, a) - planeOffset;

    for (int i = 0; i < inCount; ++i)
    {
        float3 b = g_clip[srcBase + i];
        float db = dot(planeNormal, b) - planeOffset;

        bool aInside = da <= AVBD_PLANE_EPSILON;
        bool bInside = db <= AVBD_PLANE_EPSILON;

        if (aInside != bInside)
        {
            float t = 0.0;
            float denom = da - db;
            if (abs(denom) > AVBD_SAT_AXIS_EPSILON)
                t = clamp(da / denom, 0.0, 1.0);

            if (outCount < AVBD_POLY_VERTS)
            {
                g_clip[dstBase + outCount] = a + (b - a) * t;
                outCount++;
            }
        }

        if (bInside && outCount < AVBD_POLY_VERTS)
        {
            g_clip[dstBase + outCount] = b;
            outCount++;
        }

        a = b;
        da = db;
    }

    return outCount;
}

void AvbdAddContact(float3 xA, float3 xB, int featureKey)
{
    float3 midpoint = (xA + xB) * 0.5;

    for (int i = 0; i < g_contactCount; ++i)
    {
        float3 d = midpoint - g_contactMid[i];
        if (dot(d, d) < AVBD_CONTACT_MERGE_DIST_SQ)
            return;
    }

    if (g_contactCount >= AVBD_BOX_MAX_CONTACTS)
        return;

    g_contactKey[g_contactCount] = featureKey;
    g_contactRA[g_contactCount] = AvbdQuatRotate(AvbdQuatConjugate(g_rotA), xA - g_posA);
    g_contactRB[g_contactCount] = AvbdQuatRotate(AvbdQuatConjugate(g_rotB), xB - g_posB);
    g_contactMid[g_contactCount] = midpoint;
    g_contactCount++;
}

AvbdSatAxis AvbdTestAxis(AvbdObb boxA, AvbdObb boxB, float3 delta, float3 axis, int type, int indexA, int indexB, AvbdSatAxis best)
{
    float lenSq = dot(axis, axis);
    if (lenSq < AVBD_SAT_AXIS_EPSILON)
        return best;

    float invLen = 1.0 / sqrt(lenSq);
    float3 n = axis * invLen;
    if (dot(n, delta) < 0.0)
        n = -n;

    float centerDist = abs(dot(delta, n));

    float rA =
        boxA.halfExtent.x * AvbdAbsDot(n, boxA.axis0) +
        boxA.halfExtent.y * AvbdAbsDot(n, boxA.axis1) +
        boxA.halfExtent.z * AvbdAbsDot(n, boxA.axis2);

    float rB =
        boxB.halfExtent.x * AvbdAbsDot(n, boxB.axis0) +
        boxB.halfExtent.y * AvbdAbsDot(n, boxB.axis1) +
        boxB.halfExtent.z * AvbdAbsDot(n, boxB.axis2);

    float separation = centerDist - (rA + rB);
    if (separation > 0.0)
    {
        best.gap = true;
        return best;
    }

    if (!best.valid || separation > best.separation)
    {
        best.valid = true;
        best.type = type;
        best.indexA = indexA;
        best.indexB = indexB;
        best.separation = separation;
        best.normalAB = n;
    }

    return best;
}

// The edge of the box along axisIndex that is furthest in direction dir. The two end points are returned in a pair of statics.
static float3 g_edge0;
static float3 g_edge1;

void AvbdSupportEdge(AvbdObb box, int axisIndex, float3 dir)
{
    int axis1 = (axisIndex + 1) % 3;
    int axis2 = (axisIndex + 2) % 3;

    float sign1 = dot(dir, AvbdObbAxis(box, axis1)) >= 0.0 ? 1.0 : -1.0;
    float sign2 = dot(dir, AvbdObbAxis(box, axis2)) >= 0.0 ? 1.0 : -1.0;

    float3 edgeCenter = box.center
        + AvbdObbAxis(box, axis1) * (AvbdObbHalf(box, axis1) * sign1)
        + AvbdObbAxis(box, axis2) * (AvbdObbHalf(box, axis2) * sign2);

    g_edge0 = edgeCenter - AvbdObbAxis(box, axisIndex) * AvbdObbHalf(box, axisIndex);
    g_edge1 = edgeCenter + AvbdObbAxis(box, axisIndex) * AvbdObbHalf(box, axisIndex);
}

static float3 g_closest0;
static float3 g_closest1;

void AvbdClosestPointsOnSegments(float3 p0, float3 p1, float3 q0, float3 q1)
{
    float3 d1 = p1 - p0;
    float3 d2 = q1 - q0;
    float3 r = p0 - q0;
    float a = dot(d1, d1);
    float e = dot(d2, d2);
    float f = dot(d2, r);

    float s = 0.0;
    float t = 0.0;

    if (a <= AVBD_SAT_AXIS_EPSILON && e <= AVBD_SAT_AXIS_EPSILON)
    {
        g_closest0 = p0;
        g_closest1 = q0;
        return;
    }

    if (a <= AVBD_SAT_AXIS_EPSILON)
    {
        t = clamp(f / e, 0.0, 1.0);
    }
    else
    {
        float c = dot(d1, r);
        if (e <= AVBD_SAT_AXIS_EPSILON)
        {
            s = clamp(-c / a, 0.0, 1.0);
        }
        else
        {
            float b = dot(d1, d2);
            float denom = a * e - b * b;

            if (abs(denom) > AVBD_SAT_AXIS_EPSILON)
                s = clamp((b * f - c * e) / denom, 0.0, 1.0);

            t = (b * s + f) / e;

            if (t < 0.0)
            {
                t = 0.0;
                s = clamp(-c / a, 0.0, 1.0);
            }
            else if (t > 1.0)
            {
                t = 1.0;
                s = clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }

    g_closest0 = p0 + d1 * s;
    g_closest1 = q0 + d2 * t;
}

int AvbdBuildFaceManifold(AvbdObb boxA, AvbdObb boxB, bool referenceIsA, int referenceAxis, float3 normalAB)
{
    AvbdObb referenceBox;
    AvbdObb incidentBox;
    if (referenceIsA)
    {
        referenceBox = boxA;
        incidentBox = boxB;
    }
    else
    {
        referenceBox = boxB;
        incidentBox = boxA;
    }
    float3 referenceOutward = referenceIsA ? normalAB : -normalAB;

    // Reference face frame.
    float3 refAxis = AvbdObbAxis(referenceBox, referenceAxis);
    float refSign = dot(referenceOutward, refAxis) >= 0.0 ? 1.0 : -1.0;
    float3 refNormal = refAxis * refSign;
    float3 refCenter = referenceBox.center + refNormal * AvbdObbHalf(referenceBox, referenceAxis);
    AvbdFaceAxes refFace = AvbdGetFaceAxes(referenceBox, referenceAxis);

    int incidentAxis = AvbdChooseIncidentFaceAxis(incidentBox, refNormal);

    // Incident face, facing the reference face.
    float3 incAxis = AvbdObbAxis(incidentBox, incidentAxis);
    float incSign = dot(incAxis, refNormal) > 0.0 ? -1.0 : 1.0;
    float3 incNormal = incAxis * incSign;
    float3 incCenter = incidentBox.center + incNormal * AvbdObbHalf(incidentBox, incidentAxis);
    AvbdFaceAxes incFace = AvbdGetFaceAxes(incidentBox, incidentAxis);

    g_clip[0] = incCenter + incFace.u * incFace.extentU + incFace.v * incFace.extentV;
    g_clip[1] = incCenter - incFace.u * incFace.extentU + incFace.v * incFace.extentV;
    g_clip[2] = incCenter - incFace.u * incFace.extentU - incFace.v * incFace.extentV;
    g_clip[3] = incCenter + incFace.u * incFace.extentU - incFace.v * incFace.extentV;
    int count = 4;

    // Clip the incident face against the four side planes of the reference face. The polygon ping-pongs between the halves of g_clip.
    float3 n0 = refFace.u;
    float o0 = dot(n0, refCenter) + refFace.extentU;
    count = AvbdClipPolygonAgainstPlane(0, count, n0, o0);
    if (count == 0)
        return 0;

    float3 n1 = -refFace.u;
    float o1 = dot(n1, refCenter) + refFace.extentU;
    count = AvbdClipPolygonAgainstPlane(AVBD_POLY_VERTS, count, n1, o1);
    if (count == 0)
        return 0;

    float3 n2 = refFace.v;
    float o2 = dot(n2, refCenter) + refFace.extentV;
    count = AvbdClipPolygonAgainstPlane(0, count, n2, o2);
    if (count == 0)
        return 0;

    float3 n3 = -refFace.v;
    float o3 = dot(n3, refCenter) + refFace.extentV;
    count = AvbdClipPolygonAgainstPlane(AVBD_POLY_VERTS, count, n3, o3);
    if (count == 0)
        return 0;

    // After four clips the polygon is back in the first half.
    g_contactCount = 0;
    int featurePrefix = (referenceIsA ? AVBD_AXIS_FACE_A : AVBD_AXIS_FACE_B) << 24;
    featurePrefix |= (referenceAxis & 0xFF) << 16;
    featurePrefix |= (incidentAxis & 0xFF) << 8;

    for (int i = 0; i < count && g_contactCount < AVBD_BOX_MAX_CONTACTS; ++i)
    {
        float3 pIncident = g_clip[i];
        float planeDist = dot(pIncident - refCenter, refNormal);
        if (planeDist > AVBD_PLANE_EPSILON)
            continue;

        float3 pReference = pIncident - refNormal * planeDist;
        float3 xA = referenceIsA ? pReference : pIncident;
        float3 xB = referenceIsA ? pIncident : pReference;

        AvbdAddContact(xA, xB, featurePrefix | (i & 0xFF));
    }

    if (g_contactCount == 0)
    {
        float3 xA = AvbdSupportPoint(boxA, normalAB);
        float3 xB = AvbdSupportPoint(boxB, -normalAB);
        AvbdAddContact(xA, xB, featurePrefix);
    }

    return g_contactCount;
}

int AvbdBuildEdgeContact(AvbdObb boxA, AvbdObb boxB, int axisA, int axisB, float3 normalAB)
{
    AvbdSupportEdge(boxA, axisA, normalAB);
    float3 a0 = g_edge0;
    float3 a1 = g_edge1;
    AvbdSupportEdge(boxB, axisB, -normalAB);
    float3 b0 = g_edge0;
    float3 b1 = g_edge1;

    AvbdClosestPointsOnSegments(a0, a1, b0, b1);
    float3 xA = g_closest0;
    float3 xB = g_closest1;

    g_contactCount = 0;
    int featureKey = (AVBD_AXIS_EDGE << 24) | ((axisA & 0xFF) << 8) | (axisB & 0xFF);
    AvbdAddContact(xA, xB, featureKey);

    if (g_contactCount == 0)
    {
        xA = AvbdSupportPoint(boxA, normalAB);
        xB = AvbdSupportPoint(boxB, -normalAB);
        AvbdAddContact(xA, xB, featureKey);
    }

    return g_contactCount;
}

// Collides two boxes (full sizes, not half extents). Returns the number of contacts, 0 if the boxes do not overlap, and leaves them in the
// g_contact* statics together with g_normalBA, the contact normal pointing from B to A.
int AvbdCollideBoxes(float3 posA, float4 rotA, float3 sizeA, float3 posB, float4 rotB, float3 sizeB)
{
    g_contactCount = 0;
    g_posA = posA;
    g_rotA = rotA;
    g_posB = posB;
    g_rotB = rotB;

    AvbdObb boxA = AvbdMakeObb(posA, rotA, sizeA);
    AvbdObb boxB = AvbdMakeObb(posB, rotB, sizeB);
    float3 delta = boxB.center - boxA.center;

    AvbdSatAxis bestFace;
    bestFace.type = 0;
    bestFace.indexA = -1;
    bestFace.indexB = -1;
    bestFace.separation = -3.402823466e+38;
    bestFace.normalAB = float3(0.0, 0.0, 0.0);
    bestFace.valid = false;
    bestFace.gap = false;

    AvbdSatAxis bestEdge = bestFace;

    for (int i = 0; i < 3; ++i)
    {
        bestFace = AvbdTestAxis(boxA, boxB, delta, AvbdObbAxis(boxA, i), AVBD_AXIS_FACE_A, i, -1, bestFace);
        if (bestFace.gap)
            return 0;
    }

    for (int i2 = 0; i2 < 3; ++i2)
    {
        bestFace = AvbdTestAxis(boxA, boxB, delta, AvbdObbAxis(boxB, i2), AVBD_AXIS_FACE_B, -1, i2, bestFace);
        if (bestFace.gap)
            return 0;
    }

    for (int i3 = 0; i3 < 3; ++i3)
    {
        for (int j = 0; j < 3; ++j)
        {
            float3 axis = cross(AvbdObbAxis(boxA, i3), AvbdObbAxis(boxB, j));
            bestEdge = AvbdTestAxis(boxA, boxB, delta, axis, AVBD_AXIS_EDGE, i3, j, bestEdge);
            if (bestEdge.gap)
                return 0;
        }
    }

    if (!bestFace.valid)
        return 0;

    AvbdSatAxis best = bestFace;
    if (bestEdge.valid)
    {
        if (0.95 * bestEdge.separation > bestFace.separation + 0.01)
            best = bestEdge;
    }

    g_normalBA = -best.normalAB;

    if (best.type == AVBD_AXIS_EDGE)
        return AvbdBuildEdgeContact(boxA, boxB, best.indexA, best.indexB, best.normalAB);

    if (best.type == AVBD_AXIS_FACE_A)
        return AvbdBuildFaceManifold(boxA, boxB, true, best.indexA, best.normalAB);

    return AvbdBuildFaceManifold(boxA, boxB, false, best.indexB, best.normalAB);
}

#endif // MCKRILL_AVBD_BOX_COLLISION_INCLUDED
