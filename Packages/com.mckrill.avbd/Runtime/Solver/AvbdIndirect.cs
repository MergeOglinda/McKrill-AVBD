using System;

namespace McKrill.Avbd
{
    /// <summary>
    /// Sizes of indirect dispatches. The HLSL side is Shaders/Include/AvbdIndirect.hlsl; a test checks the two against each other.
    /// A dispatch is at most 65,535 groups in one dimension, so a bigger count is laid out in rows of <see cref="RowGroups"/> groups.
    /// </summary>
    internal static class AvbdIndirect
    {
        /// <summary>Mirrors AVBD_INDIRECT_ARGS_STRIDE: bytes of arguments of one dispatch (groups x, y and z).</summary>
        public const int ArgsStride = 12;

        /// <summary>Mirrors AVBD_INDIRECT_ROW_GROUPS: thread groups per row (the limit of one dimension of a dispatch).</summary>
        public const int RowGroups = 65535;

        /// <summary>
        /// Thread groups of an indirect dispatch over <paramref name="count"/> items, as <c>AvbdIndirectGroups</c> computes them: x is at most
        /// <see cref="RowGroups"/> and y the number of rows of that width.
        /// </summary>
        public static (uint x, uint y) GroupsFor(long count)
        {
            long groups = (count + AvbdConst.GroupSize - 1) / AvbdConst.GroupSize;
            long rows = Math.Max(1L, (groups + RowGroups - 1) / RowGroups);
            return ((uint)Math.Min(groups, RowGroups), (uint)rows);
        }
    }
}
