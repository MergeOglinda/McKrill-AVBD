// Just enough of HLSL to compile the package's vector-only include files (Shaders/Include/Avbd*.hlsl) as C++.
// Included inside a namespace by the harness, so it cannot clash with the upstream solver's own float3.
// Only constructs the includes actually use are provided; a missing one is a compile error, which is the point.
// build.sh gives every float literal in the includes an f suffix first: HLSL literals are float, C++ literals are double.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

typedef uint32_t uint;

struct float3
{
    float x, y, z;
    float3() = default;
    float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    float &operator[](int i) { return (&x)[i]; }
    const float &operator[](int i) const { return (&x)[i]; }
};

// xyz aliases x, y and z, like the swizzle it stands for. (GNU anonymous struct in a union.)
struct float4
{
    union
    {
        struct { float x, y, z, w; };
        float3 xyz;
    };
    float4() = default;
    float4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    float4(float3 v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
};

struct uint2
{
    uint x, y;
    uint2() = default;
    uint2(uint x_, uint y_) : x(x_), y(y_) {}
};

inline float3 operator+(float3 a, float3 b) { return float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline float3 operator-(float3 a, float3 b) { return float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline float3 operator-(float3 a) { return float3(-a.x, -a.y, -a.z); }
inline float3 operator*(float3 a, float3 b) { return float3(a.x * b.x, a.y * b.y, a.z * b.z); }
inline float3 operator*(float3 a, float s) { return float3(a.x * s, a.y * s, a.z * s); }
inline float3 operator*(float s, float3 a) { return float3(s * a.x, s * a.y, s * a.z); }
inline float3 operator/(float3 a, float s) { return float3(a.x / s, a.y / s, a.z / s); }

inline float4 operator+(float4 a, float4 b) { return float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
inline float4 operator*(float s, float4 a) { return float4(s * a.x, s * a.y, s * a.z, s * a.w); }
inline float4 operator/(float4 a, float s) { return float4(a.x / s, a.y / s, a.z / s, a.w / s); }

struct float3x3
{
    float3 row[3];
    float3x3() = default;
    float3x3(float a, float b, float c, float d, float e, float f, float g, float h, float i)
    {
        row[0] = float3(a, b, c);
        row[1] = float3(d, e, f);
        row[2] = float3(g, h, i);
    }
    float3x3(float3 r0, float3 r1, float3 r2)
    {
        row[0] = r0;
        row[1] = r1;
        row[2] = r2;
    }
    float3 &operator[](int i) { return row[i]; }
    const float3 &operator[](int i) const { return row[i]; }
};

inline float abs(float x) { return ::fabsf(x); }
inline float sqrt(float x) { return ::sqrtf(x); }
inline float min(float a, float b) { return a < b ? a : b; }
inline float max(float a, float b) { return a > b ? a : b; }
inline float clamp(float x, float a, float b) { return max(a, min(b, x)); }

inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
inline float length(float3 v) { return ::sqrtf(dot(v, v)); }
inline float length(float4 v) { return ::sqrtf(dot(v, v)); }
// Upstream's normalize is v / length(v); HLSL's is not specified to the bit, the parity check allows for that.
inline float3 normalize(float3 v) { return v / length(v); }
inline float4 normalize(float4 v) { return v / length(v); }
inline float3 cross(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline float3 &operator+=(float3 &a, float3 b) { a = a + b; return a; }

// Matrices are mathematical: rows are rows, mul(a, b) is the product a b and mul(a, v) is a v.
inline float3x3 operator-(float3x3 a) { return float3x3(-a.row[0], -a.row[1], -a.row[2]); }
inline float3 mul(float3x3 a, float3 v) { return float3(dot(a.row[0], v), dot(a.row[1], v), dot(a.row[2], v)); }
inline float3x3 transpose(float3x3 a)
{
    return float3x3(a.row[0].x, a.row[1].x, a.row[2].x, a.row[0].y, a.row[1].y, a.row[2].y, a.row[0].z, a.row[1].z, a.row[2].z);
}
inline float3x3 mul(float3x3 a, float3x3 b)
{
    float3x3 r;
    for (int i = 0; i < 3; i++)
    {
        float3 ar = a.row[i];
        r.row[i] = float3(ar.x * b.row[0].x + ar.y * b.row[1].x + ar.z * b.row[2].x,
                          ar.x * b.row[0].y + ar.y * b.row[1].y + ar.z * b.row[2].y,
                          ar.x * b.row[0].z + ar.y * b.row[1].z + ar.z * b.row[2].z);
    }
    return r;
}
inline float3x3 operator+(float3x3 a, float3x3 b) { return float3x3(a.row[0] + b.row[0], a.row[1] + b.row[1], a.row[2] + b.row[2]); }

// Used by AvbdSphereCollision.hlsl. any(a != b) of two float3 is "a differs from b in some component".
inline float3 abs(float3 v) { return float3(::fabsf(v.x), ::fabsf(v.y), ::fabsf(v.z)); }
inline float3 clamp(float3 v, float3 lo, float3 hi) { return float3(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y), clamp(v.z, lo.z, hi.z)); }
inline bool operator!=(float3 a, float3 b) { return a.x != b.x || a.y != b.y || a.z != b.z; }
inline bool any(bool b) { return b; }
