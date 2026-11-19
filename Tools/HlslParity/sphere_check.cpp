// Checks the package's HLSL sphere narrowphase (AvbdSphereCollision.hlsl, compiled as C++ through hlsl_shim.h). Upstream avbd-demo3d has no spheres, so
// there is nothing to compare bit for bit; instead every result is checked against an independent computation in double precision:
//   * sphere against sphere, sphere against box (the sphere as A and as B): a contact exists exactly when the surfaces overlap or touch;
//   * the contact has the key AVBD_SPHERE_FEATURE; its point on the sphere is on the sphere's surface and its point on the box is on the box's surface;
//   * the normal is a unit vector from B to A, and dot(normal, xA - xB) is minus the penetration depth (the constraint value of the normal row);
//   * a sphere whose center is inside the box is pushed out through the face of least penetration.
// Exit status 1 on any failure.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace hlsl
{
#include "hlsl_shim.h"
#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdQuaternion.hlsl"
#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdContacts.hlsl"
#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdBoxCollision.hlsl"
#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdSphereCollision.hlsl"
}

namespace
{
uint64_t rngState = 0x1234567887654321ull;
double rnd()
{
    rngState ^= rngState << 13;
    rngState ^= rngState >> 7;
    rngState ^= rngState << 17;
    return (double)((rngState >> 11) & 0x1FFFFFFFFFFFFFull) / 9007199254740992.0;
}
double rnd(double lo, double hi) { return lo + (hi - lo) * rnd(); }

struct V
{
    double x, y, z;
};
V operator+(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V operator-(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V operator*(V a, double s) { return {a.x * s, a.y * s, a.z * s}; }
double dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double len(V a) { return std::sqrt(dot(a, a)); }
V fromH(hlsl::float3 v) { return {v.x, v.y, v.z}; }
hlsl::float3 toH(V v) { return hlsl::float3((float)v.x, (float)v.y, (float)v.z); }

struct Q
{
    double x, y, z, w;
};

Q randomQ()
{
    Q q{rnd(-1, 1), rnd(-1, 1), rnd(-1, 1), rnd(-1, 1)};
    double l = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (l < 1e-3)
        return {0, 0, 0, 1};
    return {q.x / l, q.y / l, q.z / l, q.w / l};
}

V rotate(Q q, V v) // double precision, independent of the HLSL
{
    V u{q.x, q.y, q.z};
    V c{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
    V t = c * 2.0;
    V tc{u.y * t.z - u.z * t.y, u.z * t.x - u.x * t.z, u.x * t.y - u.y * t.x};
    return v + t * q.w + tc;
}
Q conj(Q q) { return {-q.x, -q.y, -q.z, q.w}; }
hlsl::float4 toH(Q q) { return hlsl::float4((float)q.x, (float)q.y, (float)q.z, (float)q.w); }

long failures = 0;
long cases = 0;
long contacts = 0;
long inside = 0;

void fail(const char *what, int which, double a, double b)
{
    if (failures++ < 20)
        std::printf("FAIL %s (case %d): %.6g vs %.6g\n", what, which, a, b);
}

const double tol = 2e-4; // float arithmetic on values of a few meters

void checkSphereSphere(int id)
{
    V pa{rnd(-3, 3), rnd(-3, 3), rnd(-3, 3)};
    V pb = pa + V{rnd(-3, 3), rnd(-3, 3), rnd(-3, 3)};
    double ra = rnd(0.2, 1.5), rb = rnd(0.2, 1.5);
    Q qa = randomQ(), qb = randomQ();
    // float positions as the shader sees them
    V fa = fromH(toH(pa)), fb = fromH(toH(pb));
    double dist = len(fb - fa);
    int n = hlsl::AvbdCollideSpheres(toH(pa), toH(qa), (float)ra, toH(pb), toH(qb), (float)rb);
    cases++;
    bool expect = dist <= (double)(float)ra + (double)(float)rb;
    if (std::fabs(dist - (ra + rb)) < 1e-4)
        return; // on the edge: float rounding decides
    if ((n == 1) != expect)
        return fail("sphere-sphere contact exists", id, n, expect);
    if (n == 0)
        return;
    contacts++;
    if (hlsl::g_contactKey[0] != AVBD_SPHERE_FEATURE)
        fail("key", id, hlsl::g_contactKey[0], AVBD_SPHERE_FEATURE);
    V xA = fromH(hlsl::AvbdQuatRotate(toH(qa), hlsl::g_contactRA[0])) + fa;
    V xB = fromH(hlsl::AvbdQuatRotate(toH(qb), hlsl::g_contactRB[0])) + fb;
    V normal = fromH(hlsl::g_normalBA);
    if (std::fabs(len(normal) - 1.0) > tol)
        fail("normal is a unit vector", id, len(normal), 1.0);
    if (std::fabs(len(xA - fa) - ra) > tol)
        fail("xA on the sphere A", id, len(xA - fa), ra);
    if (std::fabs(len(xB - fb) - rb) > tol)
        fail("xB on the sphere B", id, len(xB - fb), rb);
    if (dist > 1e-3 && dot(normal, fa - fb) < 0.0)
        fail("normal points from B to A", id, dot(normal, fa - fb), 0.0);
    if (std::fabs(dot(normal, xA - xB) - (-(ra + rb - dist))) > tol)
        fail("penetration", id, dot(normal, xA - xB), -(ra + rb - dist));
}

// A sphere against a box, in both orders. Returns nothing; counts into the globals.
void checkSphereBox(int id, bool sphereIsA, bool centerInside)
{
    V half{rnd(0.2, 2.0), rnd(0.2, 2.0), rnd(0.2, 2.0)};
    Q qbox = randomQ();
    V cbox{rnd(-3, 3), rnd(-3, 3), rnd(-3, 3)};
    double r = rnd(0.2, 1.5);

    V pLocal; // the sphere's center in the box frame
    if (centerInside)
        pLocal = {rnd(-0.95, 0.95) * half.x, rnd(-0.95, 0.95) * half.y, rnd(-0.95, 0.95) * half.z};
    else
    {
        // near the box: a point of the box's surface pushed out by up to 1.2 r
        // a point of the box's surface (a random face), pushed out in a random direction: ends up over a face, an edge or a corner
        V d{rnd(-1, 1), rnd(-1, 1), rnd(-1, 1)};
        double dl = len(d);
        d = d * (1.0 / (dl > 1e-6 ? dl : 1.0));
        double c3[3] = {rnd(-1, 1) * half.x, rnd(-1, 1) * half.y, rnd(-1, 1) * half.z};
        double h3[3] = {half.x, half.y, half.z};
        int k = (int)(rnd() * 3.0) % 3;
        c3[k] = (rnd() < 0.5 ? -1.0 : 1.0) * h3[k];
        pLocal = {c3[0], c3[1], c3[2]};
        pLocal = pLocal + d * (rnd(0.0, 1.5) * r);
    }
    Q qsph = randomQ();
    V csph = cbox + rotate(qbox, pLocal);

    // everything as the shader sees it: float poses
    V fBox = fromH(toH(cbox)), fSph = fromH(toH(csph));
    Q fqb{(float)qbox.x, (float)qbox.y, (float)qbox.z, (float)qbox.w};

    // independent expectation
    V p = rotate(conj(fqb), fSph - fBox);
    V q{std::fmax(-half.x, std::fmin(half.x, p.x)), std::fmax(-half.y, std::fmin(half.y, p.y)), std::fmax(-half.z, std::fmin(half.z, p.z))};
    bool isInside = (p.x == q.x && p.y == q.y && p.z == q.z);
    double depthMin = 1e30;
    int face = 0;
    if (isInside)
    {
        double d3[3] = {half.x - std::fabs(p.x), half.y - std::fabs(p.y), half.z - std::fabs(p.z)};
        for (int k = 0; k < 3; k++)
            if (d3[k] < depthMin)
            {
                depthMin = d3[k];
                face = k;
            }
    }
    double dist = isInside ? 0.0 : len(p - q);
    if (!isInside && std::fabs(dist - r) < 1e-4)
        return; // on the edge
    if (isInside)
    {
        // near a tie between faces the float result may pick another face
        double d3[3] = {half.x - std::fabs(p.x), half.y - std::fabs(p.y), half.z - std::fabs(p.z)};
        for (int k = 0; k < 3; k++)
            if (k != face && std::fabs(d3[k] - depthMin) < 1e-4)
                return;
    }
    bool expect = isInside || dist <= (double)(float)r;
    double penetration = isInside ? r + depthMin : r - dist;

    hlsl::float3 sizeBox = toH(half * 2.0);
    hlsl::float3 sizeSph((float)(2 * r), (float)(2 * r), (float)(2 * r));
    int n;
    Q qA = sphereIsA ? qsph : qbox, qB = sphereIsA ? qbox : qsph;
    V pA = sphereIsA ? csph : cbox, pB = sphereIsA ? cbox : csph;
    n = hlsl::AvbdCollideSphereBox(sphereIsA, toH(pA), toH(qA), sphereIsA ? sizeSph : sizeBox, toH(pB), toH(qB), sphereIsA ? sizeBox : sizeSph);
    cases++;
    if ((n == 1) != expect)
        return fail("sphere-box contact exists", id, n, expect);
    if (n == 0)
        return;
    contacts++;
    if (isInside)
        inside++;
    if (hlsl::g_contactKey[0] != AVBD_SPHERE_FEATURE)
        fail("key", id, hlsl::g_contactKey[0], AVBD_SPHERE_FEATURE);

    V fA = fromH(toH(pA)), fB = fromH(toH(pB));
    V xA = fromH(hlsl::AvbdQuatRotate(toH(qA), hlsl::g_contactRA[0])) + fA;
    V xB = fromH(hlsl::AvbdQuatRotate(toH(qB), hlsl::g_contactRB[0])) + fB;
    V xSphere = sphereIsA ? xA : xB, xBox = sphereIsA ? xB : xA;
    V normal = fromH(hlsl::g_normalBA);

    if (std::fabs(len(normal) - 1.0) > tol)
        fail("normal is a unit vector", id, len(normal), 1.0);
    if (std::fabs(len(xSphere - fSph) - r) > tol)
        fail("point on the sphere's surface", id, len(xSphere - fSph), r);
    // the box point, in the box frame
    V bl = rotate(conj(fqb), xBox - fBox);
    if (std::fabs(bl.x) > half.x + tol || std::fabs(bl.y) > half.y + tol || std::fabs(bl.z) > half.z + tol)
        fail("point inside the box", id, std::fmax(std::fabs(bl.x) - half.x, std::fmax(std::fabs(bl.y) - half.y, std::fabs(bl.z) - half.z)), 0.0);
    if (isInside)
    {
        double comp[3] = {bl.x, bl.y, bl.z};
        double h3[3] = {half.x, half.y, half.z};
        if (std::fabs(std::fabs(comp[face]) - h3[face]) > tol)
            fail("point on the face of least penetration", id, std::fabs(comp[face]), h3[face]);
    }
    else if (len(bl - q) > tol)
        fail("point is the closest point of the box", id, len(bl - q), 0.0);
    // the normal points from B to A: from the box toward the sphere if the sphere is A
    V toSphere = fSph - xBox; // from the box point to the sphere's center (zero-length only at an exact touch)
    double sign = sphereIsA ? 1.0 : -1.0;
    if (!isInside && dist > 1e-3 && dot(normal * sign, toSphere) < 0.0)
        fail("normal direction", id, dot(normal * sign, toSphere), 0.0);
    if (std::fabs(dot(normal, xA - xB) - (-penetration)) > 5 * tol)
        fail("penetration", id, dot(normal, xA - xB), -penetration);
}
} // namespace

int main(int argc, char **argv)
{
    int count = argc > 1 ? std::atoi(argv[1]) : 200000;
    for (int i = 0; i < count; i++)
    {
        checkSphereSphere(i);
        checkSphereBox(i, true, false);
        checkSphereBox(i, false, false);
        checkSphereBox(i, true, true);
        checkSphereBox(i, false, true);
    }
    // Ties between faces of equal depth: the lower axis wins (x, then y, then z), as in the CPU twin (CpuSphereCollision). Random cases never tie.
    struct Tie { double px, py, pz; double nx, ny, nz; };
    const Tie ties[] = {
        {0.0, 0.0, 0.0, 1, 0, 0},        // the center of a cube: all three depths equal
        {0.5, -0.5, -0.5, 1, 0, 0},      // all three equal, away from the center
        {0.9, -0.9, 0.0, 1, 0, 0},       // x and y tie, z is deeper
        {0.0, 0.5, 0.5, 0, 1, 0},        // y and z tie below x
        {0.2, -0.9, -0.9, 0, -1, 0},     // y and z tie below x, on the negative side
        {0.9, 0.5, -0.95, 0, 0, -1},     // z is the least deep
    };
    for (const Tie &t : ties)
    {
        const V half = {1.0, 1.0, 1.0};
        hlsl::float4 identity(0.0f, 0.0f, 0.0f, 1.0f);
        int n = hlsl::AvbdCollideSphereBox(true, toH(V{t.px, t.py, t.pz}), identity, hlsl::float3(0.2f, 0.2f, 0.2f),
                                           hlsl::float3(0.0f, 0.0f, 0.0f), identity, toH(V{half.x * 2, half.y * 2, half.z * 2}));
        V normal = fromH(hlsl::g_normalBA);
        cases++;
        if (n != 1 || std::fabs(normal.x - t.nx) > 1e-6 || std::fabs(normal.y - t.ny) > 1e-6 || std::fabs(normal.z - t.nz) > 1e-6)
            fail("tie", -1, normal.x, t.nx);
    }
    std::printf("cases %ld, contacts %ld (centers inside a box %ld), failures %ld\n", cases, contacts, inside, failures);
    if (failures != 0)
    {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("OK: HLSL sphere narrowphase agrees with the independent computation\n");
    return 0;
}
