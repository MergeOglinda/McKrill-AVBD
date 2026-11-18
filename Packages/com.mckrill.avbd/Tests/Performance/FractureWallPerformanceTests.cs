using System.Diagnostics;
using NUnit.Framework;
using Unity.PerformanceTesting;

namespace McKrill.Avbd.Tests.Performance
{
    /// <summary>
    /// Performance of the GPU Fracture Wall Demo with its default settings (75 x 75 blocks in 10 walls, 56 250 blocks and about 168 000 joints), hit by a
    /// volley of 15 cubes fired at the first wall at the same time. Four scenarios: 2 m and 1 m shots, each with plastic yield off and on.
    ///
    /// Each test builds the scene and then times the solver step by step, to the end of the GPU work (see <see cref="PerformanceHarness"/>):
    /// <list type="bullet">
    /// <item><description><c>Build scene</c>: the CPU time to spawn the bodies and add the joints and create the buffers, in milliseconds.</description></item>
    /// <item><description><c>Step, first</c>: the first step. It completes the joints (torque arms), builds the lists and the coloring, and compiles the kernels.</description></item>
    /// <item><description><c>Step, wall at rest</c>: the next 29 steps, before the shots.</description></item>
    /// <item><description><c>Step, after shots</c>: the 180 steps (3 s of simulated time) after the volley is fired: the shots arrive after about 20 steps, and the wall breaks up.</description></item>
    /// <item><description>Counts: bodies, joints, joints broken at the end, and the manifold pool (slots in use and overflow).</description></item>
    /// </list>
    /// They run in the Play Mode tab of the Test Runner (the assembly has no platform restriction) and on a GPU. They take a while: a few minutes each on a large GPU.
    /// The results go to the Performance Test Report window (Window > Analysis > Performance Test Report) and to the test result XML.
    /// </summary>
    [TestFixture]
    [Category("Performance")]
    public sealed class FractureWallPerformanceTests
    {
        const int RestSteps = 29;      // after the first step
        const int ImpactSteps = 180;   // after the volley

        [Test, Performance, Timeout(1_800_000)]
        public void Scenario1_Shots2m_PlasticYieldOff()
        {
            Run(new FractureWallConfig().WithShotSize(2f).WithPlasticJoints(false));
        }

        [Test, Performance, Timeout(1_800_000)]
        public void Scenario2_Shots2m_PlasticYieldOn()
        {
            Run(new FractureWallConfig().WithShotSize(2f).WithPlasticJoints(true));
        }

        [Test, Performance, Timeout(1_800_000)]
        public void Scenario3_Shots1m_PlasticYieldOff()
        {
            Run(new FractureWallConfig().WithShotSize(1f).WithPlasticJoints(false));
        }

        [Test, Performance, Timeout(1_800_000)]
        public void Scenario4_Shots1m_PlasticYieldOn()
        {
            Run(new FractureWallConfig().WithShotSize(1f).WithPlasticJoints(true));
        }

        static void Run(FractureWallConfig config)
        {
            PerformanceHarness.RequireCompute();
            TestContext.Out.WriteLine("Fracture wall: " + config);

            var build = Stopwatch.StartNew();
            using (var scene = new FractureWallScene(config))
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
