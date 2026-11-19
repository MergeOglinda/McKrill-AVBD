using System.Diagnostics;
using NUnit.Framework;
using Unity.PerformanceTesting;

namespace McKrill.Avbd.Tests.Performance
{
    /// <summary>
    /// The sphere twin of <see cref="FractureWallPerformanceTests"/>: the GPU Fracture Wall Demo with spheres in place of the bricks (75 x 75 spheres in 10
    /// walls, 56 250 spheres and about 168 000 joints), hit by the same volley of 15 cubes fired at the first wall at the same time. The same four
    /// scenarios (2 m and 1 m shots, each with plastic yield off and on), the same phases and the same sample group names, so that
    /// <c>Tools/PerfCompare/compare_perf.py</c> lines a sphere scenario up with its box twin (match them by the scenario name: the test names differ by the class).
    ///
    /// See <see cref="FractureWallPerformanceTests"/> for what is measured. What differs is the shape of the bodies: the wall has sphere-sphere contacts and
    /// the shots and the ground have sphere-box contacts, and the wall is a hexagonal packing (see <see cref="SphereWallScene"/>), so it is about 66 m high.
    /// </summary>
    [TestFixture]
    [Category("Performance")]
    public sealed class SphereWallPerformanceTests
    {
        const int RestSteps = 29;      // after the first step
        const int ImpactSteps = 380;   // after the volley

        [Test, Performance, Timeout(1_800_000)]
        public void Scenario1_Shots2m_PlasticYieldOff()
        {
            Run(new SphereWallConfig().WithShotSize(2f).WithPlasticJoints(false));
        }

        [Test, Performance, Timeout(1_800_000)]
        public void Scenario2_Shots2m_PlasticYieldOn()
        {
            Run(new SphereWallConfig().WithShotSize(2f).WithPlasticJoints(true));
        }

        [Test, Performance, Timeout(1_800_000)]
        public void Scenario3_Shots1m_PlasticYieldOff()
        {
            Run(new SphereWallConfig().WithShotSize(1f).WithPlasticJoints(false));
        }

        [Test, Performance, Timeout(1_800_000)]
        public void Scenario4_Shots1m_PlasticYieldOn()
        {
            Run(new SphereWallConfig().WithShotSize(1f).WithPlasticJoints(true));
        }

        static void Run(SphereWallConfig config)
        {
            PerformanceHarness.RequireCompute();
            TestContext.Out.WriteLine("Sphere wall: " + config);

            var build = Stopwatch.StartNew();
            using (var scene = new SphereWallScene(config))
            {
                AvbdSolver solver = scene.Solver;
                AvbdBodyBuffers bodies = scene.Bodies;
                PerformanceHarness.WaitForGpu(bodies); // the buffers are created and filled before the clock stops
                build.Stop();
                Measure.Custom(new SampleGroup("Build scene", SampleUnit.Millisecond), build.Elapsed.TotalMilliseconds);

                Measure.Custom(PerformanceHarness.StepGroup("first"), PerformanceHarness.TimedStep(solver, bodies));
                PerformanceHarness.MeasureSteps(solver, bodies, PerformanceHarness.StepGroup("wall at rest"), RestSteps);

                int fired = scene.ShootVolley();
                Assert.AreEqual(config.volleySize, fired, "every shot of the volley must be spawned");
                PerformanceHarness.MeasureSteps(solver, bodies, PerformanceHarness.StepGroup("after shots"), ImpactSteps);

                PerformanceHarness.Report("Bodies", bodies.AliveCount);
                PerformanceHarness.Report("Joints", scene.JointCount);
                PerformanceHarness.Report("Joints broken", scene.Constraints.BrokenCount);
                ContactSnapshot contacts = scene.Constraints.ReadContacts();
                PerformanceHarness.Report("Manifold slots in use", contacts.UsedSlots);
                PerformanceHarness.Report("Manifold overflow", contacts.Overflow);

                Assert.AreEqual(0, scene.Constraints.OverflowCount, "the joint buffers must hold every joint");
                Assert.AreEqual(0, bodies.OverflowCount, "the body buffers must hold every body");
            }
        }
    }
}
