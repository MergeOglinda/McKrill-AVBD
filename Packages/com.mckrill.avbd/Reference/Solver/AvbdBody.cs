using System.Collections.Generic;
using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// State of one rigid box, a port of upstream's Rigid. A body with mass 0 (density 0) is static:
    /// it is never solved and never moves. Positions are the center of mass; the box axes are the
    /// principal axes, so the moment of inertia is the diagonal <see cref="Moment"/>.
    /// </summary>
    public sealed class AvbdBody
    {
        /// <summary>Creation index inside the owning solver.</summary>
        public int Index { get; internal set; }

        public float3 PositionLin;
        public quaternion PositionAng = quaternion.identity;

        /// <summary>Pose at the start of the step (x-).</summary>
        public float3 InitialLin;
        public quaternion InitialAng = quaternion.identity;

        /// <summary>Inertial target of the step (Eq. 2).</summary>
        public float3 InertialLin;
        public quaternion InertialAng = quaternion.identity;

        public float3 VelocityLin;
        public float3 VelocityAng;
        public float3 PrevVelocityLin;

        /// <summary>Full box widths in each dimension.</summary>
        public float3 Size;
        public float Mass;
        public float3 Moment;
        public float Friction;

        /// <summary>Bounding radius used by the broadphase.</summary>
        public float Radius;

        /// <summary>
        /// Constraints acting on this body, in creation order. Upstream keeps them in a linked list with
        /// the newest first, and the solver visits them newest-first to match.
        /// </summary>
        public readonly List<AvbdForce> Forces = new List<AvbdForce>();

        /// <summary>Graph color assigned by the Colored mode, or -1 (static body, or not colored yet).</summary>
        public int Color = -1;

        public bool IsDynamic => Mass > 0f;

        public AvbdBody(float3 size, float density, float friction, float3 position, float3 velocity)
        {
            PositionLin = position;
            VelocityLin = velocity;
            PrevVelocityLin = velocity;
            Size = size;
            Friction = friction;

            Mass = size.x * size.y * size.z * density;
            Moment = new float3(
                (size.y * size.y + size.z * size.z) / 12f * Mass,
                (size.x * size.x + size.z * size.z) / 12f * Mass,
                (size.x * size.x + size.y * size.y) / 12f * Mass);
            Radius = math.length(size * 0.5f);
        }

        /// <summary>True if any constraint (including a manifold or IgnoreCollision) joins this body and other.</summary>
        public bool ConstrainedTo(AvbdBody other)
        {
            for (int i = 0; i < Forces.Count; i++)
            {
                AvbdForce f = Forces[i];
                if ((f.BodyA == this && f.BodyB == other) || (f.BodyA == other && f.BodyB == this))
                    return true;
            }
            return false;
        }
    }
}
