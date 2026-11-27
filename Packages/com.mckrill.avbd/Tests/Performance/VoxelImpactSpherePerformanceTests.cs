using System.Diagnostics;
using NUnit.Framework;
using Unity.PerformanceTesting;

namespace McKrill.Avbd.Tests.Performance
{
    /// <summary>
    /// The sphere twin of <see cref="VoxelImpactPerformanceTests.Scenario6_VoxelImpact35k"/>: the same six hollow objects (two cubes, two spheres and two
    /// toruses, 34 880 voxels and 241 872 joints), the same flight and the same settings, but every voxel is a sphere whose diameter is the edge of the box it
    /// replaces and whose mass is the voxel mass (<see cref="VoxelImpactConfig.sphereVoxels"/>). The phases and the sample group names are those of the box
    /// scenario, and so is the test name, so that <c>Tools/PerfCompare/compare_perf.py</c> lines the two up (match them by the scenario name: the test names
    /// differ by the class). The joints are the same; what differs is the contacts: sphere-sphere ones between the voxels of different objects, one point per
    /// pair, and sphere-box ones with the ground.
    ///
    /// See <see cref="VoxelImpactPerformanceTests"/> for what is measured. It runs in the Play Mode tab of the Test Runner on a GPU.
    /// </summary>
    [TestFixture]
    [Category("Performance")]
    public sealed class VoxelImpactSpherePerformanceTests
    {
        const int ApproachSteps = 12;   // after the first step
        const int ImpactSteps = 300;    // after the approach

        [Test, Performance, Timeout(3_600_000)]
        public void Scenario6_VoxelImpact35k()
        {
            PerformanceHarness.RequireCompute();
            var config = new VoxelImpactConfig { sphereVoxels = true };
            TestContext.Out.WriteLine("Voxel impact: " + config);

            var build = Stopwatch.StartNew();
            using (var scene = new VoxelImpactScene(config))
            {
                AvbdSolver solver = scene.Solver;
                AvbdBodyBuffers bodies = scene.Bodies;
                PerformanceHarness.WaitForGpu(bodies); // the buffers are created and filled before the clock stops
                build.Stop();
                Measure.Custom(new SampleGroup("Build scene", SampleUnit.Millisecond), build.Elapsed.TotalMilliseconds);

                Assert.AreEqual(scene.PlannedJointCount, scene.JointCount, "the joint buffers must hold every joint");

                Measure.Custom(PerformanceHarness.StepGroup("first"), PerformanceHarness.TimedStep(solver, bodies));
                PerformanceHarness.MeasureSteps(solver, bodies, PerformanceHarness.StepGroup("approach"), ApproachSteps);
                PerformanceHarness.MeasureSteps(solver, bodies, PerformanceHarness.StepGroup("impact"), ImpactSteps);

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
