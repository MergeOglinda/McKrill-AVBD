using System;

namespace McKrill.Avbd
{
    /// <summary>
    /// Identifies a force field of an <see cref="AvbdSolver"/> (<see cref="AvbdSolver.AddForceField"/>). A handle goes stale when its field is
    /// removed, so an old handle can never change a newer field that took the same slot. The default value is the invalid handle.
    /// </summary>
    public readonly struct ForceFieldHandle : IEquatable<ForceFieldHandle>
    {
        // Slot + 1, so that default(ForceFieldHandle) is invalid.
        readonly uint id;
        readonly uint generation;

        internal ForceFieldHandle(int slot, uint generation)
        {
            id = (uint)slot + 1u;
            this.generation = generation;
        }

        public static ForceFieldHandle Invalid => default;

        public bool IsValid => id != 0u;

        internal int Slot => (int)id - 1;

        internal uint Generation => generation;

        public bool Equals(ForceFieldHandle other) => id == other.id && generation == other.generation;

        public override bool Equals(object obj) => obj is ForceFieldHandle other && Equals(other);

        public override int GetHashCode() => unchecked((int)(id * 397u) ^ (int)generation);

        public static bool operator ==(ForceFieldHandle a, ForceFieldHandle b) => a.Equals(b);

        public static bool operator !=(ForceFieldHandle a, ForceFieldHandle b) => !a.Equals(b);
    }
}
