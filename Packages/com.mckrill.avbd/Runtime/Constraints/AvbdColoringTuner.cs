using System;

namespace McKrill.Avbd
{
    /// <summary>What one persistent coloring step reported (the statistics words of the work buffer, read back some frames after the step), with what the step ran with.</summary>
    internal struct ColoringSample
    {
        /// <summary>The step colored every body from scratch (all PENDING) instead of keeping the colors of the step before.</summary>
        public bool fresh;
        /// <summary>Rounds the host dispatched in that step.</summary>
        public int rounds;
        /// <summary>Colors the host dispatched in that step (the cap).</summary>
        public int cap;
        /// <summary>Bodies that lost their color because a neighbor that colors first had it.</summary>
        public uint released;
        /// <summary>Bodies still waiting after the rounds (colored by ColorFinish).</summary>
        public uint leftover;
        /// <summary>Bodies that found no color below the cap and took the last one.</summary>
        public uint forced;
        /// <summary>The number of colors in use after the step.</summary>
        public uint colors;
        /// <summary>The last round (counted from 1) in which a body took a color.</summary>
        public uint lastRound;
    }

    /// <summary>
    /// Chooses, from the statistics of earlier steps, how many coloring rounds the host dispatches and how many colors the primal sweep dispatches.
    /// The statistics arrive some frames late and the choice never affects the coloring (ColorFinish completes it, and a color that does not fit under
    /// the cap is forced into the last one): too few rounds cost one slow group for a step, too small a cap costs a few bodies solved in parallel with
    /// their neighbors for a few steps, and the tuner raises either at once and lowers them slowly.
    /// Pure C#, no GPU: the tests drive it with samples.
    /// </summary>
    internal sealed class ColoringTuner
    {
        /// <summary>Colors above the highest one in use that the sweep dispatches (each is an empty dispatch). Room for the colors a new contact may need.</summary>
        public const int Headroom = 3;

        public const int MinRounds = 2;
        public const int MaxRounds = 64;

        /// <summary>Samples without a leftover before the number of rounds goes down.</summary>
        public const int Window = 8;

        public const int InitialRounds = 8;
        public const int InitialFreshRounds = 16;

        readonly int[] rounds = { InitialRounds, InitialFreshRounds };
        readonly int[][] recent = { new int[Window], new int[Window] };
        readonly int[] next = new int[2];
        readonly int[] filled = new int[2];

        int cap = AvbdGraphColoring.MaxColors;

        /// <summary>For tests: a fixed cap and a fixed number of rounds in place of the tuned ones (zero leaves the tuned value).</summary>
        public int CapOverride { get; set; }
        public int RoundsOverride { get; set; }

        /// <summary>Colors the sweep dispatches.</summary>
        public int Cap => CapOverride > 0 ? CapOverride : cap;

        /// <summary>Rounds to dispatch in a step that keeps colors, or in a step that colors from scratch.</summary>
        public int RoundsFor(bool fresh) => RoundsOverride > 0 ? RoundsOverride : rounds[fresh ? 1 : 0];

        /// <summary>After a coloring that is known exactly (the host waited for it): the cap follows its number of colors.</summary>
        public void Reset(int colors)
        {
            cap = Math.Max(1, Math.Min(AvbdGraphColoring.MaxColors, colors + Headroom));
        }

        public void Observe(in ColoringSample s)
        {
            int k = s.fresh ? 1 : 0;

            if (s.leftover > 0u)
            {
                rounds[k] = Math.Min(MaxRounds, Math.Max(rounds[k], s.rounds) * 2);
                filled[k] = 0;
            }
            else
            {
                recent[k][next[k]] = Math.Max(1, (int)s.lastRound);
                next[k] = (next[k] + 1) % Window;
                filled[k] = Math.Min(Window, filled[k] + 1);
                if (filled[k] == Window)
                {
                    int most = 0;
                    for (int i = 0; i < Window; i++)
                        most = Math.Max(most, recent[k][i]);
                    rounds[k] = Math.Max(MinRounds, Math.Min(MaxRounds, most + 2));
                }
            }

            int need = (int)Math.Min(s.colors + (uint)Headroom, (uint)AvbdGraphColoring.MaxColors);
            if (s.forced > 0u)
                need = Math.Min(AvbdGraphColoring.MaxColors, Math.Max(need, s.cap + 4));

            if (need > cap)
                cap = need;
            else if (s.fresh && s.forced == 0u)
                cap = Math.Max(1, need); // a coloring from scratch has no drift: the cap may come down to it
        }
    }
}
