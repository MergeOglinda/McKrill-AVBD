using System.Diagnostics;
using NUnit.Framework;
using Unity.PerformanceTesting;

namespace McKrill.Avbd.Tests.Performance
{
    /// <summary>
    /// Scenario 5: the 110k brick ring of three-avbd (paper Fig. 1), hit by a ball (see <see cref="BrickRingScene"/>), with 4 solver iterations like the
    /// original's benchmark. Like it, the test runs a lead-in from the start and then a window of steps, so it times the same stretch of the simulation
    /// on every machine; the ball reaches the ring a few steps into the lead-in.
    ///
    /// Reports the scene build time, the first step, the lead-in steps (steps 2 to 60), the 100 steps after it, and the counts of bodies and of
    /// manifold slots in use. The scene has far more touching pairs than the contact pool of 500 000 slots may hold, so the test reports
    /// <c>Manifold overflow</c> (candidate pairs that found no free slot) instead of failing on it: a pair without a slot has no contacts, and
    /// the number tells how much of the scene the solver does not see. It runs in Play Mode on a GPU, with the memory of a large one (a few GB).
    /// </summary>
    [TestFixture]
    [Category("Performance")]
    public sealed class BrickRingPerformanceTests
    {
        const int Iterations = 4;
        const int LeadSteps = 60;   // including the first
        const int WindowSteps = 100;

        // The bodies in Z-curve order of their positions (the default of AvbdBodyBuffers.SpawnBatch).
        [Test, Performance, Timeout(3_600_000)]
        public void Scenario5_BrickRing110k() => RunBrickRing(builderOrder: false);

        // The same scene with the bodies in the order it is built (course by course): the baseline for the comparison with the test above.
        [Test, Performance, Timeout(3_600_000)]
        public void Scenario5_BrickRing110k_BuilderOrder() => RunBrickRing(builderOrder: true);

        static void RunBrickRing(bool builderOrder)
        {
            PerformanceHarness.RequireCompute();
            TestContext.Out.WriteLine("Brick ring: " + BrickRingScene.CountBricks() + " bricks, " + Iterations + " iterations, " + (builderOrder ? "builder order" : "Z-curve order"));

            var build = Stopwatch.StartNew();
            using (var scene = new BrickRingScene(Iterations, builderOrder))
            {
                AvbdSolver solver = scene.Solver;
                AvbdBodyBuffers bodies = scene.Bodies;
                PerformanceHarness.WaitForGpu(bodies);
                build.Stop();
                Measure.Custom(new SampleGroup("Build scene", SampleUnit.Millisecond), build.Elapsed.TotalMilliseconds);

                Measure.Custom(PerformanceHarness.StepGroup("first"), PerformanceHarness.TimedStep(solver, bodies));
                PerformanceHarness.MeasureSteps(solver, bodies, PerformanceHarness.StepGroup("lead-in"), LeadSteps - 1);
                PerformanceHarness.MeasureSteps(solver, bodies, PerformanceHarness.StepGroup("window"), WindowSteps);

                PerformanceHarness.Report("Bodies", bodies.AliveCount);
                ContactSnapshot contacts = scene.Constraints.ReadContacts();
                PerformanceHarness.Report("Manifold slots in use", contacts.UsedSlots);
                PerformanceHarness.Report("Manifold overflow", contacts.Overflow);
                TestContext.Out.WriteLine("Manifold slots in use " + contacts.UsedSlots + " of " + scene.ManifoldSlots + ", overflow " + contacts.Overflow);

                Assert.AreEqual(BrickRingScene.CountBricks() + 2, bodies.AliveCount, "every brick, the ground and the ball must be spawned");
                Assert.AreEqual(0, bodies.OverflowCount, "the body buffers must hold every body");
            }
        }
    }
}
