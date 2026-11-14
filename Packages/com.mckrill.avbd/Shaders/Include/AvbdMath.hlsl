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

// Solves A x = b for a symmetric positive definite 6x6 matrix with an LDL^T factorization, no pivoting.
// Mirrors Mat6.TrySolveLDLT in the reference. a is row-major (a[row * 6 + column]); only the lower triangle is read, and a is
// overwritten by the factorization. y holds b on entry and x on success. Returns false if a pivot is not above 1e-20 or is NaN (y is then meaningless).
bool AvbdSolveLdlt6(inout float a[36], inout float y[6])
{
    float v[6];
    bool ok = true; // no early return inside the unrolled loops: a failed pivot is replaced by 1 and reported at the end

    [unroll]
    for (uint j = 0; j < 6; j++)
    {
        [unroll]
        for (uint k = 0; k < j; k++)
            v[k] = a[j * 6 + k] * a[k * 6 + k];

        float dj = a[j * 6 + j];
        [unroll]
        for (uint k2 = 0; k2 < j; k2++)
            dj -= a[j * 6 + k2] * v[k2];

        if (!(dj > 1e-20))
        {
            ok = false;
            dj = 1.0;
        }
        a[j * 6 + j] = dj;

        float inv = 1.0 / dj;
        [unroll]
        for (uint i = j + 1; i < 6; i++)
        {
            float s = a[i * 6 + j];
            [unroll]
            for (uint k3 = 0; k3 < j; k3++)
                s -= a[i * 6 + k3] * v[k3];
            a[i * 6 + j] = s * inv;
        }
    }

    // Forward substitution: L y = b
    [unroll]
    for (uint i1 = 0; i1 < 6; i1++)
    {
        float s = y[i1];
        [unroll]
        for (uint k4 = 0; k4 < i1; k4++)
            s -= a[i1 * 6 + k4] * y[k4];
        y[i1] = s;
    }

    // Diagonal: D z = y
    [unroll]
    for (uint i2 = 0; i2 < 6; i2++)
        y[i2] /= a[i2 * 6 + i2];

    // Backward substitution: L^T x = z
    [unroll]
    for (int i3 = 5; i3 >= 0; i3--)
    {
        float s = y[i3];
        [unroll]
        for (int k5 = i3 + 1; k5 < 6; k5++)
            s -= a[k5 * 6 + i3] * y[k5];
        y[i3] = s;
    }

    return ok;
}

#endif // MCKRILL_AVBD_MATH_INCLUDED
