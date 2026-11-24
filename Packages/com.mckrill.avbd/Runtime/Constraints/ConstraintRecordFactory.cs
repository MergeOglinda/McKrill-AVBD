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
        /// <param name="fracture">Norm of the angular multiplier above which the joint breaks; +infinity for never.</param>
        /// <param name="fractureLinear">Norm of the linear multiplier above which the joint breaks; +infinity for never. Needs a hard linear row (the multiplier of a
        /// soft row is always zero, so it could never break).</param>
        public static ConstraintRecord Joint(
            uint slotA, uint slotB, float3 rA, float3 rB,
            float stiffnessLin, float stiffnessAng, float fracture, float torqueArm, float fractureLinear = float.PositiveInfinity)
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

            float storedFractureLinear = 0f;
            if (!float.IsPositiveInfinity(fractureLinear))
            {
                if (!float.IsPositiveInfinity(stiffnessLin))
                    throw new System.ArgumentException("A linear fracture value needs a hard linear row: the multiplier of a soft row is always zero.", nameof(fractureLinear));
                flags |= ConstraintFlags.FracturableLinear;
                storedFractureLinear = fractureLinear;
            }

            return new ConstraintRecord
            {
                ident = new uint4(slotA, slotB, ConstraintFlagLayout.Pack(flags, ConstraintKind.Joint), 0u),
                anchorA = new float4(rA, storedLin),
                anchorB = new float4(rB, storedAng),
                c0Lin = new float4(0f, 0f, 0f, torqueArm),
                c0Ang = new float4(0f, 0f, 0f, storedFracture),
                reserved = new float4(storedFractureLinear, 0f, 0f, 0f),
            };
        }

        /// <summary>
        /// Makes a joint record plastic: sets the <see cref="ConstraintFlags.Plastic"/> flag if <paramref name="yield"/> is finite (and
        /// <see cref="ConstraintFlags.BendBreak"/> if <paramref name="bendLimit"/> is finite), the <see cref="ConstraintFlags.PlasticLinear"/> flag if
        /// <paramref name="yieldLinear"/> is finite (and <see cref="ConstraintFlags.StretchBreak"/> if <paramref name="stretchLimit"/> is finite), points
        /// ident.w at <paramref name="plasticSlot"/> and returns the <see cref="PlasticRecord"/> that goes there: the rest rotation starts as the identity
        /// and the anchor on body A as it is in the joint, which is the rigid joint.
        /// </summary>
        /// <param name="yield">Norm of the angular multiplier above which the joint yields. Above zero, or +infinity for no angular yield.</param>
        /// <param name="bendLimit">Radians the rest rotation may move from where it started before the joint breaks; +infinity for no limit.</param>
        /// <param name="yieldLinear">Norm of the linear multiplier above which the joint yields. Above zero, or +infinity for no linear yield.</param>
        /// <param name="stretchLimit">Meters the anchor on body A may move from where it started before the joint breaks; +infinity for no limit.</param>
        public static PlasticRecord MakePlastic(ref ConstraintRecord joint, uint plasticSlot, float yield, float bendLimit,
            float yieldLinear = float.PositiveInfinity, float stretchLimit = float.PositiveInfinity)
        {
            uint flags = 0u;
            float storedYield = 0f, storedBend = 0f, storedYieldLinear = 0f, storedStretch = 0f;
            if (!float.IsPositiveInfinity(yield))
            {
                flags |= (uint)ConstraintFlags.Plastic;
                storedYield = yield;
            }
            if (!float.IsPositiveInfinity(bendLimit))
            {
                flags |= (uint)ConstraintFlags.BendBreak;
                storedBend = bendLimit;
            }
            if (!float.IsPositiveInfinity(yieldLinear))
            {
                flags |= (uint)ConstraintFlags.PlasticLinear;
                storedYieldLinear = yieldLinear;
            }
            if (!float.IsPositiveInfinity(stretchLimit))
            {
                flags |= (uint)ConstraintFlags.StretchBreak;
                storedStretch = stretchLimit;
            }
            joint.ident.z |= flags;
            joint.ident.w = plasticSlot;
            return new PlasticRecord
            {
                rest = new float4(0f, 0f, 0f, 1f),
                limits = new float4(storedYield, storedBend, storedYieldLinear, storedStretch),
                restStart = new float4(0f, 0f, 0f, 1f),
                reserved = new float4(joint.anchorA.xyz, 0f),
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
