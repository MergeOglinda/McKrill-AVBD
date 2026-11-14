#ifndef MCKRILL_AVBD_CONTACT_SOLVE_INCLUDED
#define MCKRILL_AVBD_CONTACT_SOLVE_INCLUDED

#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdContacts.hlsl"
#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdQuaternion.hlsl"

// The math of one contact in the primal and dual stages. A port of EvaluateContact, UpdatePrimal and UpdateDual of
// Reference/Solver/AvbdManifold.cs (upstream's Manifold::updatePrimal and updateDual). 3x3 blocks are mathematical matrices, as in AvbdMath.hlsl.
// Plain functions on values, no buffers, so that Tools/HlslParity can compile this file as C++ and compare it with upstream.

// Contact basis: row 0 is the normal (pointing from body B to body A), rows 1 and 2 are the tangents.
float3x3 AvbdContactBasis(float3 normal)
{
    float3 t1 = AvbdContactTangent1(normal);
    float3 t2 = cross(normal, t1);
    return float3x3(normal, t1, t2);
}

struct AvbdContactEval
{
    float3x3 jALin;
    float3x3 jBLin;
    float3x3 jAAng;
    float3x3 jBAng;
    float3 C;             // Taylor-approximated constraint value
    float3 F;             // force K C + lambda, normal force <= 0 and friction clamped to the cone
    float bounds;         // friction cone radius
    float frictionScale;  // norm of the tangential force before the cone clamp
};

// Evaluates a contact at the current poses. dq* are the displacements from the start-of-step poses (linear, and AvbdQuatDifference).
AvbdContactEval AvbdEvaluateContact(
    float3 normal, float3 c0, float3 penalty, float3 lambda, float friction,
    float3 rA, float3 rB, float4 rotA, float4 rotB,
    float3 dqALin, float3 dqAAng, float3 dqBLin, float3 dqBAng, float alpha)
{
    AvbdContactEval e;
    float3x3 basis = AvbdContactBasis(normal);

    float3 rAWorld = AvbdQuatRotate(rotA, rA);
    float3 rBWorld = AvbdQuatRotate(rotB, rB);

    // Jacobians. Linear parts are +/- the basis; angular rows are r x (linear row).
    e.jALin = basis;
    e.jBLin = -basis;
    e.jAAng = float3x3(cross(rAWorld, e.jALin[0]), cross(rAWorld, e.jALin[1]), cross(rAWorld, e.jALin[2]));
    e.jBAng = float3x3(cross(rBWorld, e.jBLin[0]), cross(rBWorld, e.jBLin[1]), cross(rBWorld, e.jBLin[2]));

    e.C = c0 * (1.0 - alpha)
        + mul(e.jALin, dqALin) + mul(e.jBLin, dqBLin)
        + mul(e.jAAng, dqAAng) + mul(e.jBAng, dqBAng);

    float3 F = penalty * e.C + lambda;

    // Clamp the normal force.
    F.x = min(F.x, 0.0);

    // Clamp the norm of the friction force to the friction cone.
    e.bounds = abs(F.x) * friction;
    e.frictionScale = sqrt(F.y * F.y + F.z * F.z);
    if (e.frictionScale > e.bounds && e.frictionScale > 0.0)
    {
        F.y *= e.bounds / e.frictionScale;
        F.z *= e.bounds / e.frictionScale;
    }
    e.F = F;
    return e;
}

// The terms a contact adds to the 6x6 system of one of its bodies.
struct AvbdContactStamp
{
    float3x3 lhsLin;
    float3x3 lhsAng;
    float3x3 lhsCross;
    float3 rhsLin;
    float3 rhsAng;
};

AvbdContactStamp AvbdStampContact(AvbdContactEval e, float3 penalty, bool isA)
{
    float3x3 K = float3x3(penalty.x, 0.0, 0.0, 0.0, penalty.y, 0.0, 0.0, 0.0, penalty.z);

    // Jacobian for the body being solved.
    float3x3 jLin;
    float3x3 jAng;
    if (isA)
    {
        jLin = e.jALin;
        jAng = e.jAAng;
    }
    else
    {
        jLin = e.jBLin;
        jAng = e.jBAng;
    }

    float3x3 jLinT = transpose(jLin);
    float3x3 jAngT = transpose(jAng);
    float3x3 jAngTk = mul(jAngT, K);

    AvbdContactStamp s;
    s.lhsLin = mul(mul(jLinT, K), jLin);
    s.lhsAng = mul(jAngTk, jAng);
    s.lhsCross = mul(jAngTk, jLin);
    s.rhsLin = mul(jLinT, e.F);
    s.rhsAng = mul(jAngT, e.F);
    return s;
}

struct AvbdContactDualResult
{
    float3 lambda;
    float3 penalty;
    bool stick;
};

// Dual update of one contact: stores the force as the multiplier and ramps the penalties (Eq. 16). `stick` is only changed when the
// friction rows are inside the cone, like upstream.
AvbdContactDualResult AvbdContactDual(AvbdContactEval e, float3 penalty, bool stick, float betaLin, float penaltyMax)
{
    AvbdContactDualResult r;
    r.lambda = e.F;
    r.penalty = penalty;
    r.stick = stick;

    if (e.F.x < 0.0)
        r.penalty.x = min(penalty.x + betaLin * abs(e.C.x), penaltyMax);
    if (e.frictionScale <= e.bounds)
    {
        r.penalty.y = min(penalty.y + betaLin * abs(e.C.y), penaltyMax);
        r.penalty.z = min(penalty.z + betaLin * abs(e.C.z), penaltyMax);
        r.stick = sqrt(e.C.y * e.C.y + e.C.z * e.C.z) < AVBD_STICK_THRESHOLD;
    }
    return r;
}

#endif // MCKRILL_AVBD_CONTACT_SOLVE_INCLUDED
