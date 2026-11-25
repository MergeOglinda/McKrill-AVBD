#ifndef MCKRILL_AVBD_JOINT_DUAL_INCLUDED
#define MCKRILL_AVBD_JOINT_DUAL_INCLUDED

// The dual update of one joint (the reference solver's Joint.UpdateDual): multiplier of hard rows, penalty ramp (Eqs. 11 and 16), plastic yield of
// the angular and the linear row, and fracture. Shared by the Dual kernel (AvbdDual.compute, one thread per joint after a primal sweep) and by the
// primal kernel (AvbdPrimal.compute, with AVBD_FUSED_DUAL: the first body of a joint to be solved in a sweep updates the joint, with the poses of
// the end of the sweep before, and marks it).
//
// The including shader declares, before including this file:
//   RWStructuredBuffer<ConstraintRecord> _Constraints;
//   RWStructuredBuffer<PlasticRecord>    _Plastic;     (only with AVBD_PLASTIC_JOINTS)
//   StructuredBuffer<FractureRecord>     _Fracture;
// and includes AvbdConstraintRecords.hlsl, AvbdPlasticRecords.hlsl, AvbdSolverParams.hlsl and AvbdMath.hlsl.

// What the dual update reads of the plastic record: the limits (every plastic joint) and the rest rotation (an angular one). The caller requests them with
// AvbdJointPlasticRead as soon as it has the record, together with the poses of the two bodies, and hands them to AvbdJointDual: read inside the update, after
// the poses have been used, they cost one more round trip to memory, because a use of the poses has to wait for them before the next read can be issued.
struct JointPlasticRead
{
    float4 limits; // x = angular yield, y = bend limit, z = linear yield, w = stretch limit
    float4 rest;   // rest rotation of the angular row (the identity unless the joint is plastic)
};

// What a dual update did to the record, for AvbdJointDualStore.
struct JointDualResult
{
    bool breaks;       // the joint broke (its flags, penalties and multipliers are in the record)
    bool yielded;      // the angular row yielded: C0 of the angular row and the rest rotation changed
    bool yieldedLin;   // the linear row yielded: the anchor on A and C0 of the linear row changed
};

// The fracture thresholds of joint `index`: x = linear, y = angular. Zero for a joint that is not fracturable, which reads nothing. The caller requests
// them with the poses and hands them to AvbdJointDual, like the plastic data.
float2 AvbdJointFractureRead(uint index, uint flags)
{
    float2 threshold = float2(0.0, 0.0);
    if (flags & (AVBD_CONSTRAINT_FLAG_FRACTURABLE | AVBD_CONSTRAINT_FLAG_FRACTURABLE_LIN))
    {
        FractureRecord r = _Fracture[index >> 2u];
        float4 pair = (index & 2u) ? r.b : r.a;
        threshold = (index & 1u) ? pair.zw : pair.xy;
    }
    return threshold;
}

JointPlasticRead AvbdJointPlasticRead(ConstraintRecord c, uint flags)
{
    JointPlasticRead r;
    r.limits = float4(0.0, 0.0, 0.0, 0.0);
    r.rest = float4(0.0, 0.0, 0.0, 1.0);
#if defined(AVBD_PLASTIC_JOINTS)
    if (flags & (AVBD_CONSTRAINT_FLAG_PLASTIC | AVBD_CONSTRAINT_FLAG_PLASTIC_LIN))
    {
        r.limits = _Plastic[c.ident.w].limits;
        if (flags & AVBD_CONSTRAINT_FLAG_PLASTIC)
            r.rest = _Plastic[c.ident.w].rest;
    }
#endif
    return r;
}

// Updates `c`, a joint that is not broken, for the poses of its two bodies (a world body A is the origin with the identity rotation). `rest` returns the
// rest rotation of the angular row (the identity unless the joint is plastic): what the primal stage needs next. Writes nothing.
JointDualResult AvbdJointDual(inout ConstraintRecord c, uint flags, float3 posA, float4 rotA, float3 posB, float4 rotB, JointPlasticRead plasticRead, float2 fracture, out float4 rest)
{
    float alpha = AVBD_ALPHA;
    bool hasA = c.ident.x != AVBD_INVALID;

    float3 anchorA = hasA ? AvbdQuatRotate(rotA, c.anchorA.xyz) + posA : c.anchorA.xyz;
    float3 anchorB = AvbdQuatRotate(rotB, c.anchorB.xyz) + posB;

    float3 penaltyLin = AvbdPenaltyLin(c);
    float3 penaltyAng = AvbdPenaltyAng(c);
    float3 lambdaLin = AvbdLambdaLin(c);
    float3 lambdaAng = AvbdLambdaAng(c);
    float fractureLin = fracture.x; // thresholds of the multipliers; read only by a joint with the FRACTURABLE flags
    float fractureAng = fracture.y;

    rest = float4(0.0, 0.0, 0.0, 1.0);
#if defined(AVBD_PLASTIC_JOINTS)
    // Of the 64 B plastic record only the limits (every plastic joint) and the rest rotation (an angular one) are read, by the caller (JointPlasticRead):
    // restStart and the anchor origin (reserved) are read in the branches that need them, which a joint takes only in the step it breaks or yields, so
    // most calls read half the record.
    float4 limits = plasticRead.limits; // x = angular yield, y = bend limit, z = linear yield, w = stretch limit
    bool isPlastic = (flags & AVBD_CONSTRAINT_FLAG_PLASTIC) != 0u;
    if (isPlastic)
        rest = plasticRead.rest;
    bool yielded = false;
    bool yieldedLin = false;
#endif

    // Linear constraint
    if (dot(penaltyLin, penaltyLin) > 0.0)
    {
        float3 C = anchorA - anchorB;
        float cap = AVBD_PENALTY_MAX; // a hard constraint is limited by the maximum penalty only

        if (flags & AVBD_CONSTRAINT_FLAG_HARD_LIN)
        {
            // Stabilization
            C -= c.c0Lin.xyz * alpha;

            // Store the updated force as the multiplier.
            lambdaLin = penaltyLin * C + lambdaLin;
        }
        else
        {
            cap = min(c.anchorA.w, AVBD_PENALTY_MAX);
        }

#if defined(AVBD_PLASTIC_JOINTS)
        // Yield of the linear row: a multiplier above the yield value is clamped to it, and the joint keeps the stretch it has now (the anchor on body A
        // moves below). A multiplier above the fracture threshold does not yield: the joint breaks.
        if (flags & AVBD_CONSTRAINT_FLAG_PLASTIC_LIN)
        {
            // Squared norms: no square root unless the row yields.
            float forceSq = dot(lambdaLin, lambdaLin);
            bool fractures = (flags & AVBD_CONSTRAINT_FLAG_FRACTURABLE_LIN) && forceSq > fractureLin * fractureLin;
            if (forceSq > limits.z * limits.z && !fractures)
            {
                lambdaLin *= limits.z * rsqrt(forceSq);
                yieldedLin = true;
            }
        }
        // A yielding row does not ramp its penalty: the joint is not getting stiffer while it gives way.
        if (!yieldedLin)
#endif
        // Ramp the penalty, clamped to the material stiffness (Eq. 16).
        penaltyLin = min(penaltyLin + abs(C) * AVBD_BETA_LIN, float3(cap, cap, cap));
    }

    // Angular constraint
    if (dot(penaltyAng, penaltyAng) > 0.0)
    {
        float3 C = AvbdJointAngularError(rotA, rotB, rest) * c.c0Lin.w; // times the torque arm; rotA is the identity for a world joint
        float cap = AVBD_PENALTY_MAX;

        if (flags & AVBD_CONSTRAINT_FLAG_HARD_ANG)
        {
            C -= c.c0Ang.xyz * alpha;
            lambdaAng = penaltyAng * C + lambdaAng;
        }
        else
        {
            cap = min(c.anchorB.w, AVBD_PENALTY_MAX);
        }

#if defined(AVBD_PLASTIC_JOINTS)
        // Yield: a multiplier above the yield value is clamped to it, and the joint keeps the bend it has now (the rest rotation moves below).
        // A multiplier above the fracture threshold does not yield: the joint breaks.
        if (isPlastic)
        {
            float forceSq = dot(lambdaAng, lambdaAng);
            bool fractures = (flags & AVBD_CONSTRAINT_FLAG_FRACTURABLE) && forceSq > fractureAng * fractureAng;
            if (forceSq > limits.x * limits.x && !fractures)
            {
                lambdaAng *= limits.x * rsqrt(forceSq);
                yielded = true;
            }
        }
        // A yielding row does not ramp its penalty: the joint is not getting stiffer while it gives way.
        if (!yielded)
#endif
        penaltyAng = min(penaltyAng + abs(C) * AVBD_BETA_ANG, float3(cap, cap, cap));
    }

    bool bentOff = false;
    bool stretchedOff = false;
#if defined(AVBD_PLASTIC_JOINTS)
    if (yielded)
    {
        rest = AvbdJointRestFromPose(rotA, rotB);
        if (flags & AVBD_CONSTRAINT_FLAG_BEND_BREAK)
            bentOff = AvbdJointBend(_Plastic[c.ident.w].restStart, rest) > limits.y;
    }
    if (yieldedLin)
    {
        // The anchor on body A moves to where the anchor on body B is now (in the body space of A; a joint to the world keeps a world point), so the
        // linear error is zero from here on and the multiplier alone holds the joint at the stretch it has.
        float3 moved = hasA ? AvbdQuatRotate(AvbdQuatConjugate(rotA), anchorB - posA) : anchorB;
        c.anchorA = float4(moved, c.anchorA.w);
        if (flags & AVBD_CONSTRAINT_FLAG_STRETCH_BREAK)
            stretchedOff = length(moved - _Plastic[c.ident.w].reserved.xyz) > limits.w;
        c.c0Lin = float4(0.0, 0.0, 0.0, c.c0Lin.w); // the stabilization reference of this step must not pull the stretch back (the torque arm stays)
    }
#endif

    // Fracture tests of the two rows, and the bend and stretch limits of a plastic joint
    bool breaks = bentOff || stretchedOff
        || ((flags & AVBD_CONSTRAINT_FLAG_FRACTURABLE) && dot(lambdaAng, lambdaAng) > fractureAng * fractureAng)
        || ((flags & AVBD_CONSTRAINT_FLAG_FRACTURABLE_LIN) && dot(lambdaLin, lambdaLin) > fractureLin * fractureLin);
    if (breaks)
    {
        penaltyLin = float3(0.0, 0.0, 0.0);
        penaltyAng = float3(0.0, 0.0, 0.0);
        lambdaLin = float3(0.0, 0.0, 0.0);
        lambdaAng = float3(0.0, 0.0, 0.0);
        c.ident.z |= AVBD_CONSTRAINT_FLAG_BROKEN;
    }

#if defined(AVBD_PLASTIC_JOINTS)
    if (yielded)
    {
        // The joint now holds the bend it has: the stored C0 (the stabilization reference of this step) must not pull it back.
        c.c0Ang.xyz = float3(0.0, 0.0, 0.0);
    }
#endif

    AvbdSetConstraintState(c, penaltyLin, penaltyAng, lambdaLin, lambdaAng);

    JointDualResult result;
    result.breaks = breaks;
#if defined(AVBD_PLASTIC_JOINTS)
    result.yielded = yielded;
    result.yieldedLin = yieldedLin;
#else
    result.yielded = false;
    result.yieldedLin = false;
#endif
    return result;
}

// Write back only what the update can have changed. The three state words (the penalties and multipliers, the last 48 B of the record) change in every
// call; the others only in the call in which a joint breaks (the flags), yields in its angle (C0 of the angular row) or yields in its stretch (the anchor
// on A and C0 of the linear row). Storing the whole 128 B record would write the five unchanged words again on every call of every joint, which is more
// than half of what this stage writes. The w of c0Ang is the mark of the fused dual update: the caller that sets it does so after this store.
void AvbdJointDualStore(uint index, ConstraintRecord c, JointDualResult result, float4 rest)
{
    _Constraints[index].state0 = c.state0;
    _Constraints[index].state1 = c.state1;
    _Constraints[index].state2 = c.state2;
    if (result.breaks)
        _Constraints[index].ident.z = c.ident.z;
#if defined(AVBD_PLASTIC_JOINTS)
    if (result.yielded)
    {
        _Constraints[index].c0Ang = c.c0Ang;
        _Plastic[c.ident.w].rest = rest;
    }
    if (result.yieldedLin)
    {
        _Constraints[index].anchorA = c.anchorA;
        _Constraints[index].c0Lin = c.c0Lin;
    }
#endif
}

#endif // MCKRILL_AVBD_JOINT_DUAL_INCLUDED
