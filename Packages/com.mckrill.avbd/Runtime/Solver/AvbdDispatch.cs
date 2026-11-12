using System;
using UnityEngine;

namespace McKrill.Avbd
{
    internal static class AvbdDispatch
    {
        const int MaxGroupsPerDispatch = 65535;

        static readonly int BaseId = Shader.PropertyToID("_Base");
        static readonly int CountId = Shader.PropertyToID("_Count");

        /// <summary>
        /// Dispatches a 1D kernel over <paramref name="count"/> items. The kernel reads _Base and _Count and handles item
        /// index _Base + SV_DispatchThreadID.x. Counts needing more than 65535 thread groups are split over several dispatches.
        /// </summary>
        public static void Linear(ComputeShader shader, int kernel, int count)
        {
            if (count <= 0)
                return;

            shader.SetInt(CountId, count);
            int groups = (count + AvbdConst.GroupSize - 1) / AvbdConst.GroupSize;
            for (int firstGroup = 0; firstGroup < groups; firstGroup += MaxGroupsPerDispatch)
            {
                shader.SetInt(BaseId, firstGroup * AvbdConst.GroupSize);
                shader.Dispatch(kernel, Math.Min(MaxGroupsPerDispatch, groups - firstGroup), 1, 1);
            }
        }
    }
}
