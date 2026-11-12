using System;

namespace McKrill.Avbd
{
    /// <summary>
    /// Identifies a body. The slot is the body's row in the GPU arrays. The generation makes a handle
    /// go stale when its slot is freed and reused, so an old handle can never edit a newer body.
    /// The default value is the invalid handle.
    /// </summary>
    public readonly struct BodyHandle : IEquatable<BodyHandle>
    {
        // Slot + 1, so that default(BodyHandle) is invalid.
        readonly uint id;
        readonly uint generation;

        internal BodyHandle(int slot, uint generation)
        {
            id = (uint)slot + 1u;
            this.generation = generation;
        }

        public static BodyHandle Invalid => default;

        public bool IsValid => id != 0u;

        /// <summary>Row of this body in the GPU arrays. Only meaningful for a valid handle.</summary>
        public int Slot => (int)id - 1;

        internal uint Generation => generation;

        public bool Equals(BodyHandle other) => id == other.id && generation == other.generation;

        public override bool Equals(object obj) => obj is BodyHandle other && Equals(other);

        public override int GetHashCode() => unchecked((int)(id * 397u) ^ (int)generation);

        public static bool operator ==(BodyHandle a, BodyHandle b) => a.Equals(b);

        public static bool operator !=(BodyHandle a, BodyHandle b) => !a.Equals(b);

        public override string ToString() => IsValid ? "Body(slot " + Slot + ", gen " + generation + ")" : "Body(invalid)";
    }
}
