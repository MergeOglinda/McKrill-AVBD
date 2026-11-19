using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// CPU oracle of Include/AvbdSphereCollision.hlsl: sphere against sphere and sphere against box, one contact each, after three-avbd's sphereSphere and
    /// sphereBox (wgsl-collision.ts). The reference solver has no spheres, so this is not in <c>Reference/Solver</c>; <see cref="BoxCollision"/> stays the
    /// box-box test. Bodies are given at the start-of-step pose (PositionLin, PositionAng), with Size the full size of the shape record (a sphere: its
    /// diameter in every component).
    /// </summary>
    internal static class CpuSphereCollision
    {
        /// <summary>
        /// Collides two bodies of which at least one is a sphere. Returns the number of contacts (0 or 1), writes the contact into
        /// <paramref name="contacts"/> (offsets in the body spaces, the feature key <see cref="ContactLayout.SphereFeature"/>) and the contact basis,
        /// whose first row is the normal from B to A.
        /// </summary>
        public static int Collide(bool sphereA, bool sphereB, AvbdBody bodyA, AvbdBody bodyB, AvbdManifold.Contact[] contacts, ref float3x3 basis)
        {
            float3 normalBA;
            float3 xA;
            float3 xB;

            if (sphereA && sphereB)
            {
                float radiusA = bodyA.Size.x * 0.5f;
                float radiusB = bodyB.Size.x * 0.5f;
                float3 d = bodyB.PositionLin - bodyA.PositionLin;
                float dist = math.length(d);
                if (!(dist <= radiusA + radiusB))
                    return 0;
                float3 nAB = dist > 0f ? d / dist : new float3(0f, 0f, 1f);
                normalBA = -nAB;
                xA = bodyA.PositionLin + nAB * radiusA;
                xB = bodyB.PositionLin - nAB * radiusB;
            }
            else
            {
                AvbdBody sphere = sphereA ? bodyA : bodyB;
                AvbdBody box = sphereA ? bodyB : bodyA;
                float radius = sphere.Size.x * 0.5f;
                float3 half = box.Size * 0.5f;

                float3 p = AvbdQuat.Rotate(AvbdQuat.Conjugate(box.PositionAng), sphere.PositionLin - box.PositionLin);
                float3 q = math.clamp(p, -half, half);
                float3 onBoxLocal = q;
                float3 nLocal; // from the box toward the sphere, in the box frame
                if (math.any(p != q))
                {
                    float3 d = p - q;
                    float dist = math.length(d);
                    if (!(dist <= radius))
                        return 0;
                    nLocal = d / dist;
                }
                else
                {
                    float3 depth = half - math.abs(p);
                    int k = 0;
                    if (depth.y < depth[k])
                        k = 1;
                    if (depth.z < depth[k])
                        k = 2;
                    float s = p[k] >= 0f ? 1f : -1f;
                    nLocal = float3.zero;
                    nLocal[k] = s;
                    onBoxLocal[k] = s * half[k];
                }

                float3 n = AvbdQuat.Rotate(box.PositionAng, nLocal); // from the box toward the sphere
                float3 onBox = box.PositionLin + AvbdQuat.Rotate(box.PositionAng, onBoxLocal);
                float3 onSphere = sphere.PositionLin - n * radius;
                if (sphereA)
                {
                    normalBA = n;
                    xA = onSphere;
                    xB = onBox;
                }
                else
                {
                    normalBA = -n;
                    xA = onBox;
                    xB = onSphere;
                }
            }

            contacts[0] = new AvbdManifold.Contact
            {
                featureKey = ContactLayout.SphereFeature,
                rA = AvbdQuat.Rotate(AvbdQuat.Conjugate(bodyA.PositionAng), xA - bodyA.PositionLin),
                rB = AvbdQuat.Rotate(AvbdQuat.Conjugate(bodyB.PositionAng), xB - bodyB.PositionLin),
            };
            basis = CpuContactMath.Basis(normalBA);
            return 1;
        }
    }
}
