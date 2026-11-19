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
    /// <see cref="GravityScale"/>, <see cref="Drag"/> or <see cref="Wing"/>.
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
                    r.coefficients = new float4(linearDrag, quadraticDrag, 0f, 0f);
                    break;
                case ForceKind.Wing:
                    r.value = new float4(wind, 0f);
                    r.axis = new float4(math.normalize(normal), 0f);
                    r.coefficients = new float4(liftSlope, dragCoefficient, 0.5f * airDensity * area, 0f);
                    break;
            }
            return r;
        }
    }
}
