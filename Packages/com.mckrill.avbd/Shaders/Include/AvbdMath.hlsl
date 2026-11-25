#ifndef MCKRILL_AVBD_MATH_INCLUDED
#define MCKRILL_AVBD_MATH_INCLUDED

#include "Packages/com.mckrill.avbd/Shaders/Include/AvbdQuaternion.hlsl"

// 3x3 matrices are mathematical matrices: float3x3(r0, r1, r2) takes ROWS, m[i] is row i and mul(m, v) is m * v.
// (Unity.Mathematics on the C# side is column-major; the oracle and the tests treat float3x3 as the mathematical matrix too.)

// Cross-product matrix: AvbdSkew(a) * v == cross(a, v).
float3x3 AvbdSkew(float3 a)
{
    return float3x3(0.0, -a.z, a.y,
                    a.z, 0.0, -a.x,
                    -a.y, a.x, 0.0);
}

float3x3 AvbdDiagonal(float3 d)
{
    return float3x3(d.x, 0.0, 0.0,
                    0.0, d.y, 0.0,
                    0.0, 0.0, d.z);
}

float3x3 AvbdIdentity3()
{
    return AvbdDiagonal(float3(1.0, 1.0, 1.0));
}

// a * b^T: entry [i][j] = a_i * b_j.
float3x3 AvbdOuter(float3 a, float3 b)
{
    return float3x3(a.x * b, a.y * b, a.z * b);
}

// Diagonal matrix of the column lengths (upstream's diagonalize()).
float3x3 AvbdDiagonalize(float3x3 m)
{
    return AvbdDiagonal(float3(length(float3(m[0][0], m[1][0], m[2][0])),
                               length(float3(m[0][1], m[1][1], m[2][1])),
                               length(float3(m[0][2], m[1][2], m[2][2]))));
}

#endif // MCKRILL_AVBD_MATH_INCLUDED
