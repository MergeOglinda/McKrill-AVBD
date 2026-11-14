// Compares the package's HLSL box-box narrowphase (AvbdCollideBoxes, compiled as C++ through hlsl_shim.h) with upstream's
// Manifold::collide, which the package's reference solver ports. Three sources of box pairs:
//   1. every upstream demo scene, stepped with the unmodified upstream solver: all pairs of bodies that are near each other,
//      on every step, in both orders (A, B) and (B, A);
//   2. random boxes of random sizes, positions and orientations;
//   3. structured cases: axis-aligned and 90 / 45 degree orientations, stacked boxes at separations around zero and the
//      edge tolerance, edge-on-edge crossings.
// Both sides run the same float operations in the same order and the build disables FMA contraction, so the results are expected
// to be bit-identical: same contact count, same feature keys, same rA, rB, normal and first tangent. Exit status 1 on any difference.

#include "scenes.h"
#include "parity_random.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace hlsl
{
#include "hlsl_shim.h"
#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdContacts.hlsl"
#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdBoxCollision.hlsl"
}

namespace
{

struct Stats
{
    long pairs = 0;
    long overlapping = 0;
    long contacts = 0;
    long countMismatch = 0;
    long keyMismatch = 0;
    long valueMismatch = 0; // any difference above tolerance
    long notBitExact = 0;   // any difference at all
    double maxError = 0.0;
    long byType[3] = {0, 0, 0};
    long byCount[9] = {0};
};

std::map<std::string, Stats> allStats;
bool failed = false;

uint32_t bits(float x)
{
    uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

void compareVec(Stats &s, float3 a, hlsl::float3 b, bool &bitExact)
{
    float av[3] = {a.x, a.y, a.z};
    float bv[3] = {b.x, b.y, b.z};
    for (int i = 0; i < 3; i++)
    {
        if (bits(av[i]) != bits(bv[i]))
            bitExact = false;
        double e = std::fabs((double)av[i] - (double)bv[i]);
        if (!(e <= s.maxError))
            s.maxError = std::isnan(e) ? 1e30 : e;
        if (!(e <= 1e-5))
            s.valueMismatch++;
    }
}

void compare(Stats &s, Rigid *a, Rigid *b, const char *context)
{
    s.pairs++;

    Manifold::Contact up[8];
    std::memset(up, 0, sizeof up);
    float3x3 basis{};
    basis[0] = float3{9, 9, 9};
    basis[1] = float3{9, 9, 9};
    int nUp = Manifold::collide(a, b, up, basis);

    int nHl = hlsl::AvbdCollideBoxes(
        hlsl::float3(a->positionLin.x, a->positionLin.y, a->positionLin.z),
        hlsl::float4(a->positionAng.x, a->positionAng.y, a->positionAng.z, a->positionAng.w),
        hlsl::float3(a->size.x, a->size.y, a->size.z),
        hlsl::float3(b->positionLin.x, b->positionLin.y, b->positionLin.z),
        hlsl::float4(b->positionAng.x, b->positionAng.y, b->positionAng.z, b->positionAng.w),
        hlsl::float3(b->size.x, b->size.y, b->size.z));

    if (nUp != nHl)
    {
        if (s.countMismatch++ < 5)
            fprintf(stderr, "MISMATCH (%s): contact count upstream %d, hlsl %d\n", context, nUp, nHl);
        s.notBitExact++;
        return;
    }
    s.byCount[nUp]++;
    if (nUp == 0)
        return;

    s.overlapping++;
    s.contacts += nUp;

    bool bitExact = true;
    long valueBefore = s.valueMismatch;

    for (int k = 0; k < nUp; k++)
    {
        if (up[k].feature.key != hlsl::g_contactKey[k])
        {
            if (s.keyMismatch++ < 5)
                fprintf(stderr, "MISMATCH (%s): feature key %08x vs %08x\n", context, up[k].feature.key, hlsl::g_contactKey[k]);
            bitExact = false;
        }
        compareVec(s, up[k].rA, hlsl::g_contactRA[k], bitExact);
        compareVec(s, up[k].rB, hlsl::g_contactRB[k], bitExact);
    }
    unsigned type = ((unsigned)up[0].feature.key >> 24) & 0xFF;
    s.byType[type < 3 ? type : 2]++;

    compareVec(s, basis[0], hlsl::g_normalBA, bitExact);
    compareVec(s, basis[1], hlsl::AvbdContactTangent1(hlsl::g_normalBA), bitExact);

    if (!bitExact)
        s.notBitExact++;
    if (s.valueMismatch != valueBefore && s.valueMismatch < valueBefore + 100000)
    {
        static int printed = 0;
        if (printed++ < 5)
            fprintf(stderr, "MISMATCH (%s): values differ by more than 1e-5\n", context);
    }
}

} // namespace

int main(int argc, char **argv)
{
    int sceneSteps = argc > 1 ? atoi(argv[1]) : 60;
    int randomCases = argc > 2 ? atoi(argv[2]) : 300000;

    // 1. Scenes.
    for (int s = 0; s < sceneCount; s++)
    {
        Solver solver;
        scenes[s](&solver);
        Stats &st = allStats[std::string("scene ") + sceneNames[s]];
        for (int step = 0; step < sceneSteps; step++)
        {
            solver.step();
            std::vector<Rigid *> bodies;
            for (Rigid *r = solver.bodies; r; r = r->next)
                bodies.push_back(r);
            for (size_t i = 0; i < bodies.size(); i++)
                for (size_t j = 0; j < i; j++)
                {
                    float3 d = bodies[i]->positionLin - bodies[j]->positionLin;
                    float r = bodies[i]->radius + bodies[j]->radius + 2.0f;
                    if (dot(d, d) > r * r)
                        continue;
                    compare(st, bodies[i], bodies[j], sceneNames[s]);
                    compare(st, bodies[j], bodies[i], sceneNames[s]);
                }
        }
    }

    // 2. Random boxes, and 3. structured cases. Two throwaway bodies hold the poses.
    {
        Solver solver;
        Rigid *a = new Rigid(&solver, float3{1, 1, 1}, 1.0f, 0.5f, float3{0, 0, 0});
        Rigid *b = new Rigid(&solver, float3{1, 1, 1}, 1.0f, 0.5f, float3{0, 0, 0});

        Stats &rs = allStats["random boxes"];
        for (int i = 0; i < randomCases; i++)
        {
            float reach = rnd(0.5f, 3.0f);
            place(a, b, float3{0, 0, 0}, randomQuat(), randomSize(0.1f, 3.0f),
                  float3{rnd(-reach, reach), rnd(-reach, reach), rnd(-reach, reach)}, randomQuat(), randomSize(0.1f, 3.0f));
            compare(rs, a, b, "random");
        }

        // Flat ground and a box, like the demo scenes: a big thin static box under a rotated box near its surface.
        Stats &gs = allStats["box on ground"];
        for (int i = 0; i < randomCases; i++)
        {
            float3 sizeB = randomSize(0.2f, 2.0f);
            place(a, b, float3{0, 0, 0}, quat{0, 0, 0, 1}, float3{100, 100, 1},
                  float3{rnd(-5, 5), rnd(-5, 5), rnd(0.2f, 1.6f)}, randomQuat(), sizeB);
            compare(gs, a, b, "ground");
        }

        Stats &ss = allStats["grid-aligned"];
        for (int i = 0; i < randomCases; i++)
        {
            float3 sa = randomSize(0.5f, 2.0f);
            float3 sb = rnd() < 0.5f ? sa : randomSize(0.5f, 2.0f);
            float3 offset{rnd(-1.0f, 1.0f) * sa.x, rnd(-1.0f, 1.0f) * sa.y, rnd(-1.0f, 1.0f) * sa.z};
            // Often exactly on the faces: a stack with a separation near zero.
            float r = rnd();
            if (r < 0.4f)
            {
                int axis = (int)(rnd() * 3.0f) % 3;
                float gap = (rnd() < 0.5f ? 0.0f : rnd(-0.02f, 0.02f)) + (sa[axis] + sb[axis]) * 0.5f;
                offset[axis] = (rnd() < 0.5f ? -1.0f : 1.0f) * gap;
            }
            place(a, b, float3{0, 0, 0}, gridQuat(rnd() < 0.3f), sa, offset, gridQuat(rnd() < 0.3f), sb);
            compare(ss, a, b, "grid");
        }

        // Edge on edge: two long thin bars crossing at an angle, one above the other.
        Stats &es = allStats["crossing bars"];
        for (int i = 0; i < randomCases; i++)
        {
            float3 sa{rnd(1.0f, 3.0f), rnd(0.1f, 0.4f), rnd(0.1f, 0.4f)};
            float3 sb{rnd(1.0f, 3.0f), rnd(0.1f, 0.4f), rnd(0.1f, 0.4f)};
            quat qa = axisAngle(float3{0, 0, 1}, rnd(0, 6.2831853f)) * axisAngle(float3{1, 0, 0}, rnd(-0.8f, 0.8f));
            quat qb = axisAngle(float3{0, 0, 1}, rnd(0, 6.2831853f)) * axisAngle(float3{0, 1, 0}, rnd(-0.8f, 0.8f));
            place(a, b, float3{0, 0, 0}, qa, sa, float3{rnd(-0.3f, 0.3f), rnd(-0.3f, 0.3f), rnd(0.0f, 0.6f)}, qb, sb);
            compare(es, a, b, "bars");
        }
    }

    // Report.
    printf("%-26s %10s %10s %9s %7s %7s %7s %9s %9s  faceA/faceB/edge\n", "source", "pairs", "overlap", "contacts", "count", "key", "value", "not-exact", "max err");
    for (auto &kv : allStats)
    {
        const Stats &s = kv.second;
        printf("%-26s %10ld %10ld %9ld %7ld %7ld %7ld %9ld %9.2e  %ld/%ld/%ld\n", kv.first.c_str(), s.pairs, s.overlapping, s.contacts,
               s.countMismatch, s.keyMismatch, s.valueMismatch, s.notBitExact, s.maxError, s.byType[0], s.byType[1], s.byType[2]);
        if (s.countMismatch || s.keyMismatch || s.valueMismatch || s.notBitExact)
            failed = true;
    }
    // Coverage the structured cases are there for: 8-contact manifolds and edge contacts must both occur.
    long full = 0, edge = 0;
    for (auto &kv : allStats)
    {
        full += kv.second.byCount[8];
        edge += kv.second.byType[2];
    }
    printf("manifolds with 8 contacts: %ld, edge-edge manifolds: %ld\n", full, edge);
    if (full == 0 || edge == 0)
    {
        printf("COVERAGE FAILURE: the cases do not reach both of those\n");
        failed = true;
    }
    printf(failed ? "FAILED\n" : "OK: HLSL narrowphase is bit-identical to upstream collide()\n");
    return failed ? 1 : 0;
}
