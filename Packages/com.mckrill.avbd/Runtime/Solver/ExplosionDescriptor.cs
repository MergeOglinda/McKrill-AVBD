using System;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>The volume an explosion reaches.</summary>
    public enum ExplosionRegion
    {
        /// <summary>A sphere: every body whose center of mass is closer than the radius. The push falls off with the distance from the center.</summary>
        Sphere = 0,

        /// <summary>
        /// An oriented box: every body whose center of mass is inside it. The push falls off with how far the body is toward the nearest face of the
        /// box, measured in units of the box's half size along each axis (0 at the center, 1 at a face), so a flat box is a slab and a long one a corridor.
        /// </summary>
        Box = 1,
    }

    /// <summary>
    /// An explosion with its region and its filter, for <see cref="AvbdSolver.Explode(in ExplosionDescriptor)"/>. Make one with <see cref="Sphere"/> or
    /// <see cref="Box"/> (a default struct has an empty filter and hits nothing), then change the filter fields if needed.
    /// </summary>
    public struct ExplosionDescriptor
    {
        /// <summary>World position of the center of the explosion. The push goes away from this point, whatever the region.</summary>
        public float3 center;

        /// <summary>Impulse in N s that a body at the center would receive. May be negative (an implosion).</summary>
        public float impulse;

        public ExplosionRegion region;

        /// <summary>Reach of a <see cref="ExplosionRegion.Sphere"/>, in meters; above zero.</summary>
        public float radius;

        /// <summary>Half the size of a <see cref="ExplosionRegion.Box"/> along each of its axes; every component above zero.</summary>
        public float3 halfExtents;

        /// <summary>Orientation of a <see cref="ExplosionRegion.Box"/>. Normalized when the explosion is queued.</summary>
        public quaternion rotation;

        /// <summary>
        /// The filter: a body is hit only if its group mask (<see cref="BodyDescriptor.groupMask"/>) shares a bit with this one ...
        /// </summary>
        public uint includeMask;

        /// <summary>... and shares none with this one. Static and kinematic bodies are never hit, whatever the filter.</summary>
        public uint excludeMask;

        /// <summary>A sphere that hits every body in reach.</summary>
        public static ExplosionDescriptor Sphere(float3 center, float radius, float impulse)
        {
            return new ExplosionDescriptor
            {
                center = center,
                impulse = impulse,
                region = ExplosionRegion.Sphere,
                radius = radius,
                halfExtents = new float3(1f),
                rotation = quaternion.identity,
                includeMask = 0xFFFFFFFFu,
                excludeMask = 0u,
            };
        }

        /// <summary>An oriented box that hits every body in it.</summary>
        public static ExplosionDescriptor Box(float3 center, float3 halfExtents, quaternion rotation, float impulse)
        {
            return new ExplosionDescriptor
            {
                center = center,
                impulse = impulse,
                region = ExplosionRegion.Box,
                radius = math.length(halfExtents),
                halfExtents = halfExtents,
                rotation = rotation,
                includeMask = 0xFFFFFFFFu,
                excludeMask = 0u,
            };
        }

        /// <summary>A copy with the filter <paramref name="include"/> and <paramref name="exclude"/> (see <see cref="includeMask"/>).</summary>
        public ExplosionDescriptor WithFilter(uint include, uint exclude = 0u)
        {
            ExplosionDescriptor copy = this;
            copy.includeMask = include;
            copy.excludeMask = exclude;
            return copy;
        }

        internal void Validate()
        {
            if (!math.all(math.isfinite(center)) || !float.IsFinite(impulse))
                throw new ArgumentException("An explosion needs finite values.");

            switch (region)
            {
                case ExplosionRegion.Sphere:
                    if (!float.IsFinite(radius))
                        throw new ArgumentException("An explosion needs finite values.");
                    if (!(radius > 0f))
                        throw new ArgumentException("The radius of an explosion must be above zero.", nameof(radius));
                    break;
                case ExplosionRegion.Box:
                    if (!math.all(math.isfinite(halfExtents)) || !math.all(math.isfinite(rotation.value)))
                        throw new ArgumentException("An explosion needs finite values.");
                    if (!math.all(halfExtents > 0f))
                        throw new ArgumentException("The half extents of an explosion box must be above zero.", nameof(halfExtents));
                    if (!(math.lengthsq(rotation.value) > 1e-12f))
                        throw new ArgumentException("The rotation of an explosion box must not be zero.", nameof(rotation));
                    break;
                default:
                    throw new ArgumentException("Unknown explosion region.", nameof(region));
            }
        }

        // The record the shader reads. The rotation is normalized here, once, so the shader and the oracle use the same numbers.
        internal ExplosionRecord ToRecord(uint seed)
        {
            bool box = region == ExplosionRegion.Box;
            return new ExplosionRecord
            {
                centerRadius = new float4(center, box ? math.length(halfExtents) : radius),
                boxRotation = box ? math.normalize(rotation.value) : new float4(0f, 0f, 0f, 1f),
                halfExtentsImpulse = new float4(box ? halfExtents : new float3(1f), impulse),
                ident = new uint4(seed, (uint)region, includeMask, excludeMask),
            };
        }
    }
}
