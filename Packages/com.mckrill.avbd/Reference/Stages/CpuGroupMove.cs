using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>CPU oracle of Shaders/Compute/AvbdGroupMove.compute (<see cref="AvbdGroupMover"/>). Any change to the kernel must be made here too.</summary>
    internal static class CpuGroupMove
    {
        public static void Run(CpuBodyArrays bodies, int count, uint groupMask, GroupMoveFields fields, float3 translation, float3 linearVelocity, float3 angularVelocity)
        {
            for (int slot = 0; slot < count; slot++)
            {
                if (bodies.IsFree(slot))
                    continue;
                if ((bodies.Meta[slot].ident.y & groupMask) == 0u)
                    continue;
                if (!(bodies.Mass[slot].momentMass.w > 0f))
                    continue;

                if ((fields & GroupMoveFields.Position) != 0)
                {
                    BodyStateRecord s = bodies.State[slot];
                    s.position = new float4(s.position.xyz + translation, s.position.w);
                    s.startPosition = s.position;
                    s.startRotation = s.rotation;
                    bodies.State[slot] = s;
                }

                if ((fields & (GroupMoveFields.LinearVelocity | GroupMoveFields.AngularVelocity)) != 0)
                {
                    BodyMotionRecord v = bodies.Motion[slot];
                    if ((fields & GroupMoveFields.LinearVelocity) != 0)
                    {
                        v.linearVel = new float4(linearVelocity, v.linearVel.w);
                        v.prevLinearVel = new float4(linearVelocity, v.prevLinearVel.w);
                    }
                    if ((fields & GroupMoveFields.AngularVelocity) != 0)
                        v.angularVel = new float4(angularVelocity, v.angularVel.w);
                    bodies.Motion[slot] = v;
                }
            }
        }
    }
}
