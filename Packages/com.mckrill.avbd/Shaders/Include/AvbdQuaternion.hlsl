#ifndef MCKRILL_AVBD_QUATERNION_INCLUDED
#define MCKRILL_AVBD_QUATERNION_INCLUDED

// Quaternion helpers on plain vectors. Split from AvbdMath.hlsl, which also has 3x3 matrices, so that the narrowphase can be compiled as C++ for the
// parity harness in Tools/HlslParity.

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

// Conjugate over squared length, like upstream's inverse(). No normalization.
float4 AvbdQuatInverse(float4 q)
{
    return float4(-q.xyz, q.w) / dot(q, q);
}

// Rotates v by q using upstream's formula v + 2w (u x v) + u x (2 (u x v)).
float3 AvbdQuatRotate(float4 q, float3 v)
{
    float3 u = q.xyz;
    float3 t = cross(u, v) * 2.0;
    return v + t * q.w + cross(u, t);
}

// Upstream's "quat - quat": the vector part of 2 * a * inverse(b), with no sign canonicalization.
float3 AvbdQuatDifference(float4 a, float4 b)
{
    return AvbdQuatMul(a, AvbdQuatInverse(b)).xyz * 2.0;
}

// Conjugate: the inverse of a unit quaternion. Upstream's conjugate().
float4 AvbdQuatConjugate(float4 q)
{
    return float4(-q.xyz, q.w);
}

#endif // MCKRILL_AVBD_QUATERNION_INCLUDED
