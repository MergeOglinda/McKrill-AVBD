// Compares the package's HLSL contact math (AvbdContactWarmStart.hlsl, AvbdContactSolve.hlsl, compiled as C++ through hlsl_shim.h) with
// upstream's Manifold: initialize (merge with the old contacts, C0, warm start), updatePrimal and updateDual.
// Random box pairs with random old contact state (multipliers, penalties, stick flags), then random displacements from the
// start-of-step pose, as in the middle of an iteration. Exit status 1 on any difference above the tolerances.

#include "scenes.h"
#include "parity_random.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

namespace hlsl
{
#include "hlsl_shim.h"
#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdContactWarmStart.hlsl"
#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdContactSolve.hlsl"
}

namespace
{

struct Metric
{
    long compared = 0;
    long mismatches = 0; // above tolerance
    double maxError = 0.0;
};
std::map<std::string, Metric> metrics;
bool failed = false;

hlsl::float3 H(float3 v) { return hlsl::float3(v.x, v.y, v.z); }
hlsl::float4 H(quat q) { return hlsl::float4(q.x, q.y, q.z, q.w); }
float3 U(hlsl::float3 v) { return float3{v.x, v.y, v.z}; }

// Absolute error for values near 1 or below, relative for large ones (penalties reach 1e10).
void check(const char *name, const float *a, const float *b, int n, double tolerance, double scale = 0.0)
{
    Metric &m = metrics[name];
    m.compared++;
    double s = std::max(1.0, scale);
    for (int i = 0; i < n; i++)
        s = std::max(s, std::max(std::fabs((double)a[i]), std::fabs((double)b[i])));
    double worst = 0.0;
    for (int i = 0; i < n; i++)
    {
        double e = std::fabs((double)a[i] - (double)b[i]) / s;
        if (!(e <= worst))
            worst = std::isnan(e) ? 1e30 : e;
    }
    m.maxError = std::max(m.maxError, worst);
    if (!(worst <= tolerance))
    {
        if (m.mismatches++ < 3)
            fprintf(stderr, "MISMATCH %s: error %g\n", name, worst);
    }
}

void checkVec(const char *name, float3 a, hlsl::float3 b, double tolerance)
{
    float av[3] = {a.x, a.y, a.z}, bv[3] = {b.x, b.y, b.z};
    check(name, av, bv, 3, tolerance);
}

void checkMat(const char *name, const float3x3 &a, const hlsl::float3x3 &b, double tolerance)
{
    float av[9], bv[9];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
        {
            av[r * 3 + c] = a.row[r][c];
            bv[r * 3 + c] = (&b.row[r].x)[c];
        }
    check(name, av, bv, 9, tolerance);
}

void exact(const char *name, bool equal)
{
    Metric &m = metrics[name];
    m.compared++;
    if (!equal && m.mismatches++ < 3)
        fprintf(stderr, "MISMATCH %s\n", name);
}

float logUniform(float lo, float hi) { return std::exp(rnd(std::log(lo), std::log(hi))); }

void jitter(Rigid *r, float amount)
{
    r->positionLin = r->positionLin + float3{rnd(-amount, amount), rnd(-amount, amount), rnd(-amount, amount)};
    quat d = axisAngle(float3{rnd(-1, 1) + 1e-3f, rnd(-1, 1), rnd(-1, 1)}, rnd(0, amount));
    quat q = d * r->positionAng;
    float l = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    r->positionAng = quat{q.x / l, q.y / l, q.z / l, q.w / l};
}

} // namespace

int main(int argc, char **argv)
{
    int cases = argc > 1 ? atoi(argv[1]) : 100000;

    Solver solver;
    // Non-default values, so that a mix-up of the parameters is visible.
    solver.alpha = 0.97f;
    solver.gamma = 0.93f;
    solver.betaLin = 1.7e5f;
    Rigid *a = new Rigid(&solver, float3{1, 1, 1}, 1.0f, 0.5f, float3{0, 0, 0});
    Rigid *b = new Rigid(&solver, float3{1, 1, 1}, 1.0f, 0.5f, float3{0, 0, 0});
    Manifold *m = new Manifold(&solver, a, b);

    long usable = 0, warm = 0, stuck = 0;

    for (int it = 0; it < cases; it++)
    {
        // A pair that overlaps, in the shapes the scenes have: a box on a big thin one, grid-aligned stacks, random boxes.
        a->friction = rnd(0.1f, 1.0f);
        b->friction = rnd(0.1f, 1.0f);
        float kind = rnd();
        if (kind < 0.4f)
            place(a, b, float3{0, 0, 0}, quat{0, 0, 0, 1}, float3{100, 100, 1},
                  float3{rnd(-5, 5), rnd(-5, 5), rnd(0.3f, 1.2f)}, randomQuat(), randomSize(0.3f, 2.0f));
        else if (kind < 0.7f)
        {
            float3 sa = randomSize(0.5f, 2.0f);
            place(a, b, float3{0, 0, 0}, gridQuat(rnd() < 0.3f), sa, float3{rnd(-0.5f, 0.5f), rnd(-0.5f, 0.5f), (sa.z + 1.0f) * 0.5f - rnd(0.0f, 0.02f)},
                  gridQuat(rnd() < 0.3f), randomSize(0.5f, 2.0f));
        }
        else
            place(a, b, float3{0, 0, 0}, randomQuat(), randomSize(0.3f, 2.5f),
                  float3{rnd(-1.2f, 1.2f), rnd(-1.2f, 1.2f), rnd(-1.2f, 1.2f)}, randomQuat(), randomSize(0.3f, 2.5f));

        // First step: the old manifold.
        m->numContacts = 0;
        m->initialize();
        if (m->numContacts == 0)
            continue;
        usable++;

        // Give the old contacts a history.
        for (int i = 0; i < m->numContacts; i++)
        {
            m->contacts[i].penalty = float3{logUniform(1.0f, 1e8f), logUniform(1.0f, 1e8f), logUniform(1.0f, 1e8f)};
            m->contacts[i].lambda = float3{-logUniform(0.01f, 1e4f), rnd(-50, 50), rnd(-50, 50)};
            m->contacts[i].stick = rnd() < 0.5f;
        }
        hlsl::g_oldCount = m->numContacts;
        for (int i = 0; i < m->numContacts; i++)
        {
            hlsl::g_oldKey[i] = m->contacts[i].feature.key;
            hlsl::g_oldRA[i] = H(m->contacts[i].rA);
            hlsl::g_oldRB[i] = H(m->contacts[i].rB);
            hlsl::g_oldPenalty[i] = H(m->contacts[i].penalty);
            hlsl::g_oldLambda[i] = H(m->contacts[i].lambda);
            hlsl::g_oldStick[i] = m->contacts[i].stick;
        }

        // The bodies moved: nothing, a little, a lot.
        float r = rnd();
        float amount = r < 0.2f ? 0.0f : (r < 0.7f ? 0.002f : (r < 0.9f ? 0.05f : 0.3f));
        if (amount > 0.0f)
        {
            jitter(a, amount);
            jitter(b, amount);
        }

        // Second step: upstream's initialize, and the same on the HLSL side.
        m->initialize();
        int n = hlsl::AvbdCollideBoxes(H(a->positionLin), H(a->positionAng), H(a->size), H(b->positionLin), H(b->positionAng), H(b->size));
        exact("init: contact count", n == m->numContacts);
        if (n != m->numContacts || n == 0)
            continue;
        hlsl::AvbdInitContacts(solver.alpha, solver.gamma, PENALTY_MIN, PENALTY_MAX);

        for (int i = 0; i < n; i++)
        {
            const Manifold::Contact &c = m->contacts[i];
            exact("init: feature key", c.feature.key == hlsl::g_contactKey[i]);
            exact("init: stick flag", c.stick == hlsl::g_contactStick[i]);
            checkVec("init: rA", c.rA, hlsl::g_contactRA[i], 1e-6);
            checkVec("init: rB", c.rB, hlsl::g_contactRB[i], 1e-6);
            checkVec("init: C0", c.C0, hlsl::g_contactC0[i], 1e-6);
            checkVec("init: lambda", c.lambda, hlsl::g_contactLambda[i], 1e-6);
            checkVec("init: penalty", c.penalty, hlsl::g_contactPenalty[i], 1e-6);
            if (c.penalty.x != PENALTY_MIN || c.penalty.y != PENALTY_MIN)
                warm++;
            if (c.stick)
                stuck++;
        }

        // Middle of an iteration: the start-of-step pose is the pose now, the bodies then move a little (and sometimes a lot).
        a->initialLin = a->positionLin;
        a->initialAng = a->positionAng;
        b->initialLin = b->positionLin;
        b->initialAng = b->positionAng;
        float move = rnd() < 0.8f ? 0.01f : 0.1f;
        jitter(a, move);
        jitter(b, move);

        hlsl::float3 dqALin = H(a->positionLin - a->initialLin);
        hlsl::float3 dqBLin = H(b->positionLin - b->initialLin);
        // The angular displacement is upstream's own here: AvbdQuatDifference (checked by the stage tests) multiplies the quaternions in a
        // different order of operations, and with penalties up to 1e10 a rounding difference of 1e-7 in C is a visible difference in F.
        // That would hide what this program checks, which is the contact math itself.
        hlsl::float3 dqAAng = H(a->positionAng - a->initialAng);
        hlsl::float3 dqBAng = H(b->positionAng - b->initialAng);

        // Primal: the terms stamped into both bodies' systems.
        for (int side = 0; side < 2; side++)
        {
            Rigid *body = side == 0 ? a : b;
            float3x3 lhsLin{}, lhsAng{}, lhsCross{};
            float3 rhsLin{}, rhsAng{};
            m->updatePrimal(body, solver.alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);

            hlsl::float3x3 hLin(0, 0, 0, 0, 0, 0, 0, 0, 0), hAng = hLin, hCross = hLin;
            hlsl::float3 hRhsLin(0, 0, 0), hRhsAng(0, 0, 0);
            for (int i = 0; i < n; i++)
            {
                const Manifold::Contact &c = m->contacts[i];
                hlsl::AvbdContactEval e = hlsl::AvbdEvaluateContact(
                    H(m->basis[0]), H(c.C0), H(c.penalty), H(c.lambda), m->friction, H(c.rA), H(c.rB), H(a->positionAng), H(b->positionAng),
                    dqALin, dqAAng, dqBLin, dqBAng, solver.alpha);
                hlsl::AvbdContactStamp s = hlsl::AvbdStampContact(e, H(c.penalty), side == 0);
                hLin = hLin + s.lhsLin;
                hAng = hAng + s.lhsAng;
                hCross = hCross + s.lhsCross;
                hRhsLin = hRhsLin + s.rhsLin;
                hRhsAng = hRhsAng + s.rhsAng;
            }
            checkMat("primal: lhsLin", lhsLin, hLin, 2e-5);
            checkMat("primal: lhsAng", lhsAng, hAng, 2e-5);
            checkMat("primal: lhsCross", lhsCross, hCross, 2e-5);
            checkVec("primal: rhsLin", rhsLin, hRhsLin, 2e-5);
            checkVec("primal: rhsAng", rhsAng, hRhsAng, 2e-5);
        }

        // Dual: the multipliers, penalties and stick flags after the update.
        hlsl::AvbdContactDualResult expected[8];
        for (int i = 0; i < n; i++)
        {
            const Manifold::Contact &c = m->contacts[i];
            hlsl::AvbdContactEval e = hlsl::AvbdEvaluateContact(
                H(m->basis[0]), H(c.C0), H(c.penalty), H(c.lambda), m->friction, H(c.rA), H(c.rB), H(a->positionAng), H(b->positionAng),
                dqALin, dqAAng, dqBLin, dqBAng, solver.alpha);
            expected[i] = hlsl::AvbdContactDual(e, H(c.penalty), c.stick, solver.betaLin, PENALTY_MAX);
        }
        m->updateDual(solver.alpha);
        for (int i = 0; i < n; i++)
        {
            const Manifold::Contact &c = m->contacts[i];
            checkVec("dual: lambda", c.lambda, expected[i].lambda, 2e-5);
            checkVec("dual: penalty", c.penalty, expected[i].penalty, 2e-5);
            exact("dual: stick flag", c.stick == expected[i].stick);
        }
    }

    printf("%-22s %10s %10s %10s\n", "check", "compared", "mismatch", "max err");
    for (auto &kv : metrics)
    {
        printf("%-22s %10ld %10ld %10.2e\n", kv.first.c_str(), kv.second.compared, kv.second.mismatches, kv.second.maxError);
        if (kv.second.mismatches)
            failed = true;
    }
    printf("usable pairs %ld of %d; warm-started contacts %ld; contacts that were sticking %ld\n", usable, cases, warm, stuck);
    if (usable < cases / 4 || warm == 0 || stuck == 0)
    {
        printf("COVERAGE FAILURE\n");
        failed = true;
    }
    printf(failed ? "FAILED\n" : "OK: HLSL contact math matches upstream's Manifold\n");
    return failed ? 1 : 0;
}
