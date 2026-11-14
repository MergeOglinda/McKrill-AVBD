// Random box generators shared by the parity programs. Needs scenes.h (upstream's Rigid, quat, float3) included first.
#pragma once

#include <cmath>
#include <cstdint>

namespace
{

// ---- random generation -----------------------------------------------------------------------------

uint64_t rngState = 0x9E3779B97F4A7C15ull;
float rnd() // uniform in [0, 1)
{
    rngState ^= rngState << 13;
    rngState ^= rngState >> 7;
    rngState ^= rngState << 17;
    return (float)((rngState >> 40) & 0xFFFFFF) / 16777216.0f;
}
float rnd(float lo, float hi) { return lo + (hi - lo) * rnd(); }

quat randomQuat()
{
    quat q{rnd(-1, 1), rnd(-1, 1), rnd(-1, 1), rnd(-1, 1)};
    float l = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (l < 1e-3f)
        return quat{0, 0, 0, 1};
    return quat{q.x / l, q.y / l, q.z / l, q.w / l};
}

quat axisAngle(float3 axis, float angle)
{
    float h = angle * 0.5f;
    float s = std::sin(h);
    return quat{axis.x * s, axis.y * s, axis.z * s, std::cos(h)};
}

// One of the 24 axis-aligned orientations plus an optional small perturbation.
quat gridQuat(bool perturb)
{
    static const float3 axes[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    quat q = quat{0, 0, 0, 1};
    for (int i = 0; i < 3; i++)
    {
        int turns = (int)(rnd() * 4.0f);
        q = axisAngle(axes[i], turns * 1.5707963f) * q;
    }
    if (rnd() < 0.3f)
        q = axisAngle(axes[(int)(rnd() * 3.0f) % 3], 0.78539816f) * q;
    if (perturb)
        q = axisAngle(float3{rnd(-1, 1), rnd(-1, 1), rnd(-1, 1)}, rnd(0, 0.05f)) * q;
    float l = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return quat{q.x / l, q.y / l, q.z / l, q.w / l};
}

void place(Rigid *a, Rigid *b, float3 pa, quat qa, float3 sa, float3 pb, quat qb, float3 sb)
{
    a->positionLin = pa;
    a->positionAng = qa;
    a->size = sa;
    b->positionLin = pb;
    b->positionAng = qb;
    b->size = sb;
}

float3 randomSize(float lo, float hi) { return float3{rnd(lo, hi), rnd(lo, hi), rnd(lo, hi)}; }

} // namespace
