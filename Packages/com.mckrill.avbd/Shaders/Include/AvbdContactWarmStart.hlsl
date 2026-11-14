#ifndef MCKRILL_AVBD_CONTACT_WARM_START_INCLUDED
#define MCKRILL_AVBD_CONTACT_WARM_START_INCLUDED

#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdContacts.hlsl"
#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdBoxCollision.hlsl"

// The part of the manifold's Initialize that follows collision: merging the new contacts with the manifold of the last step, then the
// constraint value at the start of the step and the warm start (Eq. 19). A port of AvbdManifold.Initialize (upstream's Manifold::initialize).
// Uses the statics of AvbdBoxCollision.hlsl: AvbdCollideBoxes has just filled g_contact*, g_normalBA and the poses g_posA .. g_rotB.

// The old manifold, filled by the caller before AvbdInitContacts.
static int    g_oldCount;
static int    g_oldKey[AVBD_BOX_MAX_CONTACTS];
static float3 g_oldRA[AVBD_BOX_MAX_CONTACTS];
static float3 g_oldRB[AVBD_BOX_MAX_CONTACTS];
static float3 g_oldPenalty[AVBD_BOX_MAX_CONTACTS];
static float3 g_oldLambda[AVBD_BOX_MAX_CONTACTS];
static bool   g_oldStick[AVBD_BOX_MAX_CONTACTS];

// Result, for the g_contactCount contacts. g_contactRA and g_contactRB are updated in place (an old anchor is kept if the old contact stuck).
static float3 g_contactC0[AVBD_BOX_MAX_CONTACTS];
static float3 g_contactPenalty[AVBD_BOX_MAX_CONTACTS];
static float3 g_contactLambda[AVBD_BOX_MAX_CONTACTS];
static bool   g_contactStick[AVBD_BOX_MAX_CONTACTS];

float3 AvbdClamp3(float3 v, float lo, float hi)
{
    return float3(clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi));
}

void AvbdInitContacts(float alpha, float gamma, float penaltyMin, float penaltyMax)
{
    float3 normal = g_normalBA;
    float3 t1 = AvbdContactTangent1(normal);
    float3 t2 = cross(normal, t1);

    for (int i = 0; i < g_contactCount; ++i)
    {
        // Merge with the old contact of the same feature: it keeps its multiplier, penalty and stick flag, and its anchors if it stuck.
        float3 penalty = float3(0.0, 0.0, 0.0);
        float3 lambda = float3(0.0, 0.0, 0.0);
        bool stick = false;

        for (int j = 0; j < g_oldCount; ++j)
        {
            if (g_contactKey[i] == g_oldKey[j])
            {
                penalty = g_oldPenalty[j];
                lambda = g_oldLambda[j];
                stick = g_oldStick[j];
                if (g_oldStick[j])
                {
                    g_contactRA[i] = g_oldRA[j];
                    g_contactRB[i] = g_oldRB[j];
                }
                break;
            }
        }

        // Error at the start-of-step pose, in the contact basis, plus the margin.
        float3 xA = AvbdQuatRotate(g_rotA, g_contactRA[i]) + g_posA;
        float3 xB = AvbdQuatRotate(g_rotB, g_contactRB[i]) + g_posB;
        float3 d = xA - xB;
        g_contactC0[i] = float3(dot(normal, d) + AVBD_COLLISION_MARGIN, dot(t1, d), dot(t2, d));

        // Warm start the multipliers and penalties (Eq. 19); the penalty is clamped to a safe range.
        g_contactLambda[i] = lambda * alpha * gamma;
        g_contactPenalty[i] = AvbdClamp3(penalty * gamma, penaltyMin, penaltyMax);
        g_contactStick[i] = stick;
    }
}

#endif // MCKRILL_AVBD_CONTACT_WARM_START_INCLUDED
