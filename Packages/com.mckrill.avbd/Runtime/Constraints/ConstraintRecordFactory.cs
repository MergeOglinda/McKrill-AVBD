using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>
    /// Builds constraint records from the parameters of the reference solver's joints and springs. Infinite stiffness becomes a
    /// flag (see <see cref="ConstraintFlags.HardLinear"/>), and the warm-start state (C0, penalties, multipliers) starts at zero.
    /// </summary>
    internal static class ConstraintRecordFactory
    {
        /// <param name="slotA">Slot of body A, or <see cref="AvbdConst.Invalid"/> to pin body B to the world point rA.</param>
        /// <param name="torqueArm">Squared length of the sum of both body sizes (the reference joint's torqueArm).</param>
        public static ConstraintRecord Joint(
            uint slotA, uint slotB, float3 rA, float3 rB,
            float stiffnessLin, float stiffnessAng, float fracture, float torqueArm)
        {
            ConstraintFlags flags = ConstraintFlags.None;

            float storedLin = stiffnessLin;
            if (float.IsPositiveInfinity(stiffnessLin))
            {
                flags |= ConstraintFlags.HardLinear;
                storedLin = 0f;
            }

            float storedAng = stiffnessAng;
            if (float.IsPositiveInfinity(stiffnessAng))
            {
                flags |= ConstraintFlags.HardAngular;
                storedAng = 0f;
            }

            float storedFracture = 0f;
            if (!float.IsPositiveInfinity(fracture))
            {
                flags |= ConstraintFlags.Fracturable;
                storedFracture = fracture;
            }

            return new ConstraintRecord
            {
                ident = new uint4(slotA, slotB, ConstraintFlagLayout.Pack(flags, ConstraintKind.Joint), 0u),
                anchorA = new float4(rA, storedLin),
                anchorB = new float4(rB, storedAng),
                c0Lin = new float4(0f, 0f, 0f, torqueArm),
                c0Ang = new float4(0f, 0f, 0f, storedFracture),
            };
        }

        public static ConstraintRecord Spring(uint slotA, uint slotB, float3 rA, float3 rB, float stiffness, float rest)
        {
            return new ConstraintRecord
            {
                ident = new uint4(slotA, slotB, ConstraintFlagLayout.Pack(ConstraintFlags.None, ConstraintKind.Spring), 0u),
                anchorA = new float4(rA, stiffness),
                anchorB = new float4(rB, rest),
            };
        }
    }
}
