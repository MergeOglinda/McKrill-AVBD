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

        /// <summary>Oracle of the kernel <c>HoldCapture</c>: for every slot below <paramref name="count"/>, the offset of the body from <paramref name="point"/> and w = 1 if it
        /// is in the group (not free, in the mask, with mass), else all zero.</summary>
        public static float4[] HoldCapture(CpuBodyArrays bodies, int count, uint groupMask, float3 point)
        {
            var offsets = new float4[count];
            for (int slot = 0; slot < count; slot++)
                if (IsMember(bodies, slot, groupMask))
                    offsets[slot] = new float4(bodies.State[slot].position.xyz - point, 1f);
            return offsets;
        }

        /// <summary>Oracle of the kernel <c>HoldApply</c>: every body that was remembered is put at <paramref name="point"/> plus its offset and gets the linear velocity.</summary>
        public static void Hold(CpuBodyArrays bodies, int count, uint groupMask, float3 point, float3 linearVelocity, float4[] offsets)
        {
            for (int slot = 0; slot < count; slot++)
            {
                if (!IsMember(bodies, slot, groupMask) || offsets[slot].w == 0f)
                    continue;
                BodyStateRecord s = bodies.State[slot];
                s.position = new float4(point + offsets[slot].xyz, s.position.w);
                s.startPosition = s.position;
                s.startRotation = s.rotation;
                bodies.State[slot] = s;

                BodyMotionRecord v = bodies.Motion[slot];
                v.linearVel = new float4(linearVelocity, v.linearVel.w);
                v.prevLinearVel = new float4(linearVelocity, v.prevLinearVel.w);
                bodies.Motion[slot] = v;
            }
        }

        // The bodies the group kernels act on: not free, in the group, with mass (kinematic bodies are in).
        static bool IsMember(CpuBodyArrays bodies, int slot, uint groupMask)
        {
            return !bodies.IsFree(slot) && (bodies.Meta[slot].ident.y & groupMask) != 0u && bodies.Mass[slot].momentMass.w > 0f;
        }
    }
}
