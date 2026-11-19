#ifndef MCKRILL_AVBD_EXPLOSIONS_INCLUDED
#define MCKRILL_AVBD_EXPLOSIONS_INCLUDED

// Explosions: one-shot radial impulses applied to the bodies in stage P (kernel PredictExplosions of AvbdPredict.compute).
// Mirrored by Reference/Stages/CpuExplosion.cs, which the tests compare against. Must match Runtime/Solver/ExplosionRecord.cs.

// Most explosions one step can apply (AvbdSolver.MaxExplosionsPerStep).
#define AVBD_MAX_EXPLOSIONS 16

// An explosion that moves a body not at the center of mass gives it a spin: the impulse acts at a point this fraction of the body's radius
// (half its diagonal) away from the center of mass, in a direction hashed from the slot.
#define AVBD_EXPLOSION_KICK 0.25

// 48 B. One explosion, applied in the step it was queued in. Only float4 and uint4 members, like every record (layout rule P1).
struct ExplosionRecord
{
    float4 centerRadius; // xyz = center, w = radius: a body whose center of mass is farther away is not touched
    float4 impulse;      // x = N s that a body at the center would receive (its velocity changes by impulse / mass)
    uint4  ident;        // x = seed: makes the hashed kicks of different explosions different
};

// A 32-bit integer hash (PCG), the same arithmetic as AvbdExplosion.Hash on the CPU.
uint AvbdExplosionHash(uint x)
{
    uint state = x * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Three hashed numbers in [-1, 1) for a body: 23 random bits each, so every step of the arithmetic is exact.
float3 AvbdExplosionRandom(uint slot, uint seed)
{
    uint h0 = AvbdExplosionHash(slot ^ AvbdExplosionHash(seed));
    uint h1 = AvbdExplosionHash(h0 + 0x9E3779B9u);
    uint h2 = AvbdExplosionHash(h1 + 0x9E3779B9u);
    return float3((float)(h0 >> 9u), (float)(h1 >> 9u), (float)(h2 >> 9u)) * (1.0 / 4194304.0) - 1.0;
}

#endif // MCKRILL_AVBD_EXPLOSIONS_INCLUDED
