using System.Diagnostics;
using NUnit.Framework;
using Unity.Mathematics;
using Unity.PerformanceTesting;
using UnityEngine;

namespace McKrill.Avbd.Tests.Performance
{
    /// <summary>
    /// What the performance tests share: timing a solver step to the end of its GPU work, and the sample groups they report.
    ///
    /// A step is timed from the call to <see cref="AvbdSolver.Step"/> until the GPU has finished it, so the time is what a frame would wait for, not the
    /// time the CPU needs to submit the work. The wait is a synchronous read of one record of the body state buffer: the read cannot finish before
    /// the last step's work that writes the buffer. It adds a few microseconds and the latency of one readback to every sample.
    /// </summary>
    internal static class PerformanceHarness
    {
        // The same layout as the body state record (64 B): the package keeps its own struct internal, and a test only needs a 64 B element to read into.
        struct StateElement
        {
            public float4 a, b, c, d;
        }

        static readonly StateElement[] SyncTarget = new StateElement[1];

        /// <summary>Per-step wall time in milliseconds, CPU and GPU together, by phase of a scenario.</summary>
        public static SampleGroup StepGroup(string phase) => new SampleGroup("Step, " + phase, SampleUnit.Millisecond);

        /// <summary>A single value of a scenario that is not a time (a count).</summary>
        public static SampleGroup CountGroup(string name) => new SampleGroup(name, SampleUnit.Undefined);

        /// <summary>Skips the test on a device without compute shaders (the package cannot run there).</summary>
        public static void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }

        /// <summary>Returns when the GPU has finished everything submitted so far.</summary>
        public static void WaitForGpu(AvbdBodyBuffers bodies)
        {
            if (bodies.State.stride != 64)
                throw new System.InvalidOperationException("The body state record is no longer 64 B; update StateElement in PerformanceHarness.");
            bodies.State.GetData(SyncTarget, 0, 0, 1);
        }

        /// <summary>Runs one solver step and returns the milliseconds it took, with the GPU's part of it (see the class comment).</summary>
        public static double TimedStep(AvbdSolver solver, AvbdBodyBuffers bodies)
        {
            var watch = Stopwatch.StartNew();
            solver.Step();
            WaitForGpu(bodies);
            watch.Stop();
            return watch.Elapsed.TotalMilliseconds;
        }

        /// <summary>Runs <paramref name="steps"/> steps and reports each as a sample of <paramref name="group"/>.</summary>
        public static void MeasureSteps(AvbdSolver solver, AvbdBodyBuffers bodies, SampleGroup group, int steps)
        {
            for (int i = 0; i < steps; i++)
                Measure.Custom(group, TimedStep(solver, bodies));
        }

        /// <summary>Reports a count as a one-sample group.</summary>
        public static void Report(string name, double value) => Measure.Custom(CountGroup(name), value);
    }
}
