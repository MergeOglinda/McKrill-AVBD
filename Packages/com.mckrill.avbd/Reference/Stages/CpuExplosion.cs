using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>One explosion for <see cref="CpuPredict"/>: what <see cref="AvbdSolver.Explode"/> queues, with the seed the solver gave it.</summary>
    internal struct CpuExplosion
    {
        public float3 center;
        public float radius;
        public float impulse;
        public uint seed;
    }

    /// <summary>
    /// CPU oracle for the explosions of stage P: the same arithmetic as ApplyExplosions in Shaders/Compute/AvbdPredict.compute and the hash in
    /// Shaders/Include/AvbdExplosions.hlsl.
    /// </summary>
    internal static class CpuExplosionMath
    {
        /// <summary>Mirrors AVBD_EXPLOSION_KICK.</summary>
        public const float Kick = 0.25f;

        /// <summary>Mirrors AvbdExplosionHash (PCG).</summary>
        public static uint Hash(uint x)
        {
            unchecked
            {
                uint state = x * 747796405u + 2891336453u;
                uint word = ((state >> (int)((state >> 28) + 4u)) ^ state) * 277803737u;
                return (word >> 22) ^ word;
            }
        }

        /// <summary>Mirrors AvbdExplosionRandom: three numbers in [-1, 1) for a body and an explosion.</summary>
        public static float3 Random(uint slot, uint seed)
        {
            unchecked
            {
                uint h0 = Hash(slot ^ Hash(seed));
                uint h1 = Hash(h0 + 0x9E3779B9u);
                uint h2 = Hash(h1 + 0x9E3779B9u);
                return new float3((float)(h0 >> 9), (float)(h1 >> 9), (float)(h2 >> 9)) * (1f / 4194304f) - 1f;
            }
        }

        /// <summary>
        /// Mirrors ApplyExplosions: changes the velocities of the body (linear, previous linear and angular) for every explosion that reaches it.
        /// Returns true if one did.
        /// </summary>
        public static bool Apply(uint slot, float mass, float3 moments, float3 size, quaternion rotation, float3 position,
            CpuExplosion[] explosions, int count, ref float3 linearVel, ref float3 prevLinearVel, ref float3 angularVel)
        {
            bool hit = false;
            for (int i = 0; i < count; i++)
            {
                CpuExplosion e = explosions[i];
                float3 d = position - e.center;
                float distanceSq = math.dot(d, d);
                if (!(distanceSq < e.radius * e.radius))
                    continue;

                float dist = math.sqrt(distanceSq);
                float falloff = 1f - dist / e.radius;
                falloff *= falloff;

                float3 random = Random(slot, e.seed);
                float3 direction;
                if (dist > 1e-4f)
                {
                    direction = d / dist;
                }
                else
                {
                    float3 pick = Random(slot, e.seed ^ 0x85EBCA6Bu);
                    float pickLength = math.length(pick);
                    direction = pickLength > 1e-6f ? pick / pickLength : new float3(1f, 0f, 0f);
                }

                float3 impulse = direction * (e.impulse * falloff);
                float3 dv = impulse / mass;
                linearVel += dv;
                prevLinearVel += dv;

                float3 offset = random * (Kick * 0.5f * math.length(size));
                float3 angularImpulse = math.cross(offset, impulse);
                float3 safeMoments = math.max(moments, 1e-9f);
                float3 bodySpin = AvbdQuat.Rotate(AvbdQuat.Conjugate(rotation), angularImpulse) / safeMoments;
                angularVel += AvbdQuat.Rotate(rotation, bodySpin);
                hit = true;
            }
            return hit;
        }
    }
}
