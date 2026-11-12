using System;

namespace McKrill.Avbd
{
    /// <summary>
    /// Capacity limits and memory cost of the per-body arrays (design doc section 5).
    /// Capacity is fixed when the buffers are created.
    /// </summary>
    public static class BodyBufferCapacity
    {
        /// <summary>Bytes of GPU memory per body slot across all body arrays.</summary>
        public const int BytesPerBody = BodyRecordStrides.Total;

        /// <summary>
        /// Largest supported capacity. The limit keeps the biggest array (64 B per body) under 2 GiB, and is
        /// far below 0x7FFFFFFF, so ids keep their top bit free (rule P6).
        /// </summary>
        public const int MaxBodies = int.MaxValue / BodyRecordStrides.Largest;

        /// <summary>Total bytes of the per-body arrays for the given capacity. Does not include the edit command buffer.</summary>
        public static long BodyArrayBytes(int capacity) => (long)capacity * BytesPerBody;

        public static bool TryValidate(int capacity, out string error)
        {
            if (capacity < 1)
            {
                error = "Capacity must be at least 1, got " + capacity + ".";
                return false;
            }
            if (capacity > MaxBodies)
            {
                error = "Capacity " + capacity + " exceeds the maximum of " + MaxBodies + " bodies.";
                return false;
            }
            error = null;
            return true;
        }

        /// <summary>Throws ArgumentOutOfRangeException if the capacity is not supported.</summary>
        public static void Validate(int capacity)
        {
            if (!TryValidate(capacity, out string error))
                throw new ArgumentOutOfRangeException(nameof(capacity), error);
        }
    }
}
