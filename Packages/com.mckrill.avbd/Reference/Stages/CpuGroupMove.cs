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

        /// <summary>The centroid of the bodies of a group as the hold kernels take it: the offsets from <paramref name="reference"/> are summed as integers in 1/1024 m.
        /// Returns false if the group is empty.</summary>
        public static bool Centroid(CpuBodyArrays bodies, int count, uint groupMask, float3 reference, out float3 centroid)
        {
            int3 sum = int3.zero;
            int members = 0;
            for (int slot = 0; slot < count; slot++)
            {
                if (!IsMember(bodies, slot, groupMask))
                    continue;
                sum += (int3)math.floor((bodies.State[slot].position.xyz - reference) * 1024f + 0.5f);
                members++;
            }
            centroid = members == 0 ? float3.zero : reference + (float3)sum / (1024f * members);
            return members > 0;
        }

        /// <summary>Oracle of the kernel <c>HoldApply</c>: moves every body of the group by the same translation so that the centroid is at <paramref name="anchor"/> +
        /// <paramref name="offset"/>, and sets the velocities to zero. Does nothing for an empty group.</summary>
        public static void Hold(CpuBodyArrays bodies, int count, uint groupMask, float3 reference, float3 anchor, float3 offset)
        {
            if (!Centroid(bodies, count, groupMask, reference, out float3 centroid))
                return;
            float3 translation = anchor + offset - centroid;
            for (int slot = 0; slot < count; slot++)
            {
                if (!IsMember(bodies, slot, groupMask))
                    continue;
                BodyStateRecord s = bodies.State[slot];
                s.position = new float4(s.position.xyz + translation, s.position.w);
                s.startPosition = s.position;
                s.startRotation = s.rotation;
                bodies.State[slot] = s;

                BodyMotionRecord v = bodies.Motion[slot];
                v.linearVel = new float4(float3.zero, v.linearVel.w);
                v.prevLinearVel = new float4(float3.zero, v.prevLinearVel.w);
                v.angularVel = new float4(float3.zero, v.angularVel.w);
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
