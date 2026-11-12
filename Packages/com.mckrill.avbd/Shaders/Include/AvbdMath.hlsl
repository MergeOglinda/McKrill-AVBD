#ifndef MCKRILL_AVBD_MATH_INCLUDED
#define MCKRILL_AVBD_MATH_INCLUDED

// Quaternions are float4 (x, y, z, w), the same convention as Unity.Mathematics.
// These mirror Reference/Math/AvbdQuat.cs.

// Hamilton product a * b (b is applied first).
float4 AvbdQuatMul(float4 a, float4 b)
{
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
                  a.w * b.w - dot(a.xyz, b.xyz));
}

// q' = normalize(q + 0.5 * (dw, 0) * q): upstream's "quat + float3".
float4 AvbdIntegrateAngular(float4 q, float3 dw)
{
    float4 r = q + 0.5 * AvbdQuatMul(float4(dw, 0.0), q);
    return normalize(r);
}

#endif // MCKRILL_AVBD_MATH_INCLUDED
