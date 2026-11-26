using System;
using Unity.Mathematics;

namespace McKrill.Avbd
{
    /// <summary>What a force field does to the bodies of its group.</summary>
    public enum ForceKind
    {
        /// <summary>A constant force in N on each body, in the world's frame or the body's.</summary>
        Constant = 1,

        /// <summary>Gravity, scaled: the body falls as if gravity were <c>scale</c> times as strong (0 is weightless, 1 changes nothing, negative falls up).</summary>
        GravityScale = 2,

        /// <summary>Drag in moving air: a force against the velocity relative to the air, linear plus quadratic in the speed.</summary>
        Drag = 3,

        /// <summary>A wing: lift and drag on a flat plate moving through air. Give it to the blocks that form the wings of an airplane.</summary>
        Wing = 4,

        /// <summary>
        /// A damped spring that pulls the group, as a whole, toward a point. The force depends on the centroid of the group, so every body of the group gets
        /// the same force and the group keeps its shape. Use it to hold or drag a set of blocks.
        /// </summary>
        Spring = 5,
    }

    /// <summary>The frame of the force of a <see cref="ForceKind.Constant"/> field.</summary>
    public enum ForceSpace
    {
        World = 0,

        /// <summary>The force turns with the body: a thrust along its z axis.</summary>
        Body = 1,
    }

    /// <summary>
    /// A force that acts on a group of bodies, for <see cref="AvbdSolver.AddForceField"/>. A body is in the group if its group mask
    /// (<see cref="BodyDescriptor.groupMask"/>, <see cref="AvbdBodyBuffers.SetGroupMask"/>) has a bit in common with <see cref="groupMask"/>; this is
    /// the only label, and it is already part of every body. Static and kinematic bodies are never moved. Make one with <see cref="Force"/>,
    /// <see cref="GravityScale"/>, <see cref="Drag"/>, <see cref="Wing"/> or <see cref="Spring"/>.
    /// </summary>
    /// <remarks>
    /// A new body has all bits set in its group mask, so it is in every group. Give the bodies the masks of the groups they belong to (for an airplane,
    /// 1 for the wings and 2 for everything else, say) and a field with the mask 1 acts on the wings only.
    /// </remarks>
    public struct ForceFieldDescriptor
    {
        public ForceKind kind;

        /// <summary>The group: the field acts on the bodies whose group mask has one of these bits.</summary>
        public uint groupMask;

        /// <summary>Constant: the force in N on each body of the group. Not a total shared by the group.</summary>
        public float3 force;

        /// <summary>Constant: the frame of <see cref="force"/>.</summary>
        public ForceSpace space;

        /// <summary>Gravity scale: how strong gravity is for the group, as a multiple of the solver's gravity.</summary>
        public float scale;

        /// <summary>Drag and wing: the velocity of the air in m/s (wind). Zero is still air.</summary>
        public float3 wind;

        /// <summary>Drag: the force is <c>linearDrag * v + quadraticDrag * |v| * v</c> against the velocity v relative to the air; N s/m.</summary>
        public float linearDrag;

        /// <summary>Drag: N s^2/m^2 (see <see cref="linearDrag"/>).</summary>
        public float quadraticDrag;

        /// <summary>
        /// Drag: shielding of buried bodies. 0 (the default) gives every body the full drag. Above 0 it is the number of joints of a body that is fully shielded
        /// from the air (18 for a voxel inside a chunk of voxels linked to their 18 neighbors): the drag of a body falls linearly from full at
        /// <see cref="surfaceJoints"/> joints to none at this many, so a voxel in the middle of a chunk feels no wind, one on its surface feels it all, and a
        /// voxel that broke off (fewer joints) feels it too. The count is that of the body's constraint list at the last build of the lists (a step old at most);
        /// contacts are not counted.
        /// </summary>
        public float buriedJoints;

        /// <summary>Drag: the number of joints of a body that is fully exposed, such as a voxel on a flat face (13 of 18). Used with <see cref="buriedJoints"/>.</summary>
        public float surfaceJoints;

        /// <summary>Wing: the normal of the plate, in the body's frame: the direction of lift for a body flying level. A unit vector is not required.</summary>
        public float3 normal;

        /// <summary>Wing: the area of the plate in m^2.</summary>
        public float area;

        /// <summary>Wing: the density of the air in kg/m^3.</summary>
        public float airDensity;

        /// <summary>Wing: the lift per unit of the sine of the angle of attack, as a coefficient. 2 pi for a thin airfoil, about 4 to 5 for a real wing.</summary>
        public float liftSlope;

        /// <summary>Wing: the drag of the surface at no angle of attack, as a coefficient.</summary>
        public float dragCoefficient;

        /// <summary>Spring: the point the centroid of the group is pulled to.</summary>
        public float3 target;

        /// <summary>Spring: the stiffness in N/m: the force on each body of the group is <c>stiffness * (target - centroid)</c>.</summary>
        public float stiffness;

        /// <summary>Spring: the damping ratio of the velocity of each body. 1 is critical damping, 0 none.</summary>
        public float dampingRatio;

        /// <summary>Spring: a direction (any length but not zero; zero for none) of a line through <see cref="target"/>, for <see cref="offsetGain"/>.</summary>
        public float3 lineDirection;

        /// <summary>Spring: how much the stiffness grows with the distance of the centroid of the group from the line through <see cref="target"/> along
        /// <see cref="lineDirection"/>, in 1/m: the stiffness is <c>stiffness * (1 + offsetGain * distance)</c>. 0 for a constant stiffness.</summary>
        public float offsetGain;

        /// <summary>A constant force in N on each body of the group.</summary>
        public static ForceFieldDescriptor Force(uint groupMask, float3 force, ForceSpace space = ForceSpace.World)
        {
            return new ForceFieldDescriptor { kind = ForceKind.Constant, groupMask = groupMask, force = force, space = space };
        }

        /// <summary>Gravity <paramref name="scale"/> times as strong for the group: 0.2 is a body that floats down, 0 one that hangs in the air.</summary>
        public static ForceFieldDescriptor GravityScale(uint groupMask, float scale)
        {
            return new ForceFieldDescriptor { kind = ForceKind.GravityScale, groupMask = groupMask, scale = scale };
        }

        /// <summary>Drag against the velocity relative to air that moves with <paramref name="wind"/>: <c>linear * v + quadratic * |v| * v</c>.</summary>
        public static ForceFieldDescriptor Drag(uint groupMask, float linear, float quadratic, float3 wind = default)
        {
            return new ForceFieldDescriptor { kind = ForceKind.Drag, groupMask = groupMask, linearDrag = linear, quadraticDrag = quadratic, wind = wind };
        }

        /// <summary>
        /// As <see cref="Drag(uint, float, float, float3)"/>, with the drag of a body scaled by how exposed it is, from its number of joints (see
        /// <see cref="buriedJoints"/>): <paramref name="buriedJoints"/> joints and more mean no drag, <paramref name="surfaceJoints"/> and fewer mean all of it.
        /// The defaults are those of a voxel with the 18-neighborhood joints of the voxelizer (<c>--connectivity 18</c>).
        /// </summary>
        public static ForceFieldDescriptor ShieldedDrag(uint groupMask, float linear, float quadratic, float3 wind = default, float buriedJoints = 18f, float surfaceJoints = 13f)
        {
            return new ForceFieldDescriptor
            {
                kind = ForceKind.Drag, groupMask = groupMask, linearDrag = linear, quadraticDrag = quadratic, wind = wind,
                buriedJoints = buriedJoints, surfaceJoints = surfaceJoints,
            };
        }

        /// <summary>
        /// A wing for the bodies of the group. The force on a flat plate of <paramref name="area"/> whose normal (in the body's frame) is
        /// <paramref name="normal"/>, moving through air at relative velocity v: <c>-q (liftSlope (v^ . n) n + dragCoefficient v^)</c> with
        /// <c>q = 0.5 airDensity area |v|^2</c> and v^ the direction of v. The part across the flow is lift. There is no stall: the force keeps growing
        /// with the sine of the angle of attack.
        /// </summary>
        public static ForceFieldDescriptor Wing(uint groupMask, float3 normal, float area, float liftSlope = 4.5f, float dragCoefficient = 0.05f,
            float airDensity = 1.225f, float3 wind = default)
        {
            return new ForceFieldDescriptor
            {
                kind = ForceKind.Wing, groupMask = groupMask, normal = normal, area = area, liftSlope = liftSlope,
                dragCoefficient = dragCoefficient, airDensity = airDensity, wind = wind,
            };
        }

        /// <summary>
        /// A damped spring that pulls the centroid of the group toward <paramref name="target"/>. Each body of the group gets the force
        /// <c>stiffness * (target - centroid)</c> in N (so a group of n bodies is pulled with n times that), and a damping force
        /// <c>2 dampingRatio sqrt(stiffness m) v</c> against its own velocity. The centroid is that of the bodies of the group at the start of the step;
        /// the sum of their offsets from the target must stay below 2 million m (a fixed-point sum in 32 bits).
        /// </summary>
        /// <param name="lineDirection">With <paramref name="offsetGain"/>: the stiffness is <c>stiffness * (1 + offsetGain * d)</c>, where d is the distance in m of
        /// the centroid from the line through the target along this direction. For a grab, the line of sight through the target: the farther the group is from the
        /// center of the view, the harder it is pulled back. Zero (the default) for a constant stiffness.</param>
        public static ForceFieldDescriptor Spring(uint groupMask, float3 target, float stiffness, float dampingRatio = 1f, float3 lineDirection = default, float offsetGain = 0f)
        {
            return new ForceFieldDescriptor
            {
                kind = ForceKind.Spring, groupMask = groupMask, target = target, stiffness = stiffness, dampingRatio = dampingRatio,
                lineDirection = lineDirection, offsetGain = offsetGain,
            };
        }

        internal void Validate()
        {
            switch (kind)
            {
                case ForceKind.Constant:
                    RequireFinite(force);
                    if (space != ForceSpace.World && space != ForceSpace.Body)
                        throw new ArgumentException("Unknown force space.", nameof(space));
                    break;
                case ForceKind.GravityScale:
                    if (!float.IsFinite(scale))
                        throw new ArgumentException("A force field needs finite values.");
                    break;
                case ForceKind.Drag:
                    RequireFinite(wind);
                    if (!float.IsFinite(linearDrag) || !float.IsFinite(quadraticDrag) || linearDrag < 0f || quadraticDrag < 0f)
                        throw new ArgumentException("The drag coefficients must be finite and not negative.");
                    if (!float.IsFinite(buriedJoints) || !float.IsFinite(surfaceJoints) || buriedJoints < 0f || surfaceJoints < 0f)
                        throw new ArgumentException("The joint counts of the shielding must be finite and not negative.");
                    if (buriedJoints > 0f && !(surfaceJoints < buriedJoints))
                        throw new ArgumentException("The joints of an exposed body must be fewer than those of a shielded one.");
                    break;
                case ForceKind.Wing:
                    RequireFinite(wind);
                    RequireFinite(normal);
                    if (!(math.lengthsq(normal) > 1e-12f))
                        throw new ArgumentException("The normal of a wing must not be zero.", nameof(normal));
                    if (!float.IsFinite(area) || !(area > 0f))
                        throw new ArgumentException("The area of a wing must be above zero.", nameof(area));
                    if (!float.IsFinite(airDensity) || !(airDensity > 0f))
                        throw new ArgumentException("The density of the air must be above zero.", nameof(airDensity));
                    if (!float.IsFinite(liftSlope) || !float.IsFinite(dragCoefficient) || liftSlope < 0f || dragCoefficient < 0f)
                        throw new ArgumentException("The lift slope and the drag coefficient must be finite and not negative.");
                    break;
                case ForceKind.Spring:
                    RequireFinite(target);
                    if (!float.IsFinite(stiffness) || !float.IsFinite(dampingRatio) || stiffness < 0f || dampingRatio < 0f)
                        throw new ArgumentException("The stiffness and the damping ratio of a spring must be finite and not negative.");
                    RequireFinite(lineDirection);
                    if (!float.IsFinite(offsetGain) || offsetGain < 0f)
                        throw new ArgumentException("The offset gain of a spring must be finite and not negative.", nameof(offsetGain));
                    break;
                default:
                    throw new ArgumentException("Unknown force kind.", nameof(kind));
            }
        }

        static void RequireFinite(float3 v)
        {
            if (!math.all(math.isfinite(v)))
                throw new ArgumentException("A force field needs finite values.");
        }

        // The record the shader reads. Unused members are zero.
        internal ForceFieldRecord ToRecord()
        {
            var r = new ForceFieldRecord { ident = new uint4((uint)kind, groupMask, 0u, 0u) };
            switch (kind)
            {
                case ForceKind.Constant:
                    r.value = new float4(force, 0f);
                    r.ident.z = space == ForceSpace.Body ? 1u : 0u; // AVBD_FORCE_FLAG_BODY_SPACE
                    break;
                case ForceKind.GravityScale:
                    r.coefficients = new float4(scale, 0f, 0f, 0f);
                    break;
                case ForceKind.Drag:
                    r.value = new float4(wind, 0f);
                    r.coefficients = new float4(linearDrag, quadraticDrag, buriedJoints, buriedJoints > 0f ? surfaceJoints : 0f);
                    break;
                case ForceKind.Wing:
                    r.value = new float4(wind, 0f);
                    r.axis = new float4(math.normalize(normal), 0f);
                    r.coefficients = new float4(liftSlope, dragCoefficient, 0.5f * airDensity * area, 0f);
                    break;
                case ForceKind.Spring:
                    r.value = new float4(target, 0f);
                    r.axis = new float4(math.lengthsq(lineDirection) > 1e-12f ? math.normalize(lineDirection) : float3.zero, 0f);
                    r.coefficients = new float4(stiffness, dampingRatio, offsetGain, 0f);
                    break;
            }
            return r;
        }
    }
}
