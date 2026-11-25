using System.Diagnostics;
using NUnit.Framework;
using Unity.PerformanceTesting;

namespace McKrill.Avbd.Tests.Performance
{
    /// <summary>
    /// Scenario 6: six hollow objects made of 0.2 m voxels (two cubes, two spheres and two toruses, 34 880 voxels and 241 872 joints together) that fly at
    /// one point and collide there all at once, the pairs that face each other at 50 m/s (see <see cref="VoxelImpactScene"/>). The settings are those of the
    /// JSON Mesh Import Demo, except that the linear fracture and the linear yield are on and the solver runs 25 iterations.
    ///
    /// Times the solver step by step, to the end of the GPU work (see <see cref="PerformanceHarness"/>):
    /// <list type="bullet">
    /// <item><description><c>Build scene</c>: the CPU time to make the shapes, spawn the bodies, add the joints and create the buffers, in milliseconds.</description></item>
    /// <item><description><c>Step, first</c>: the first step. It completes the joints (torque arms), builds the lists and the coloring, and compiles the kernels.</description></item>
    /// <item><description><c>Step, approach</c>: the next 12 steps. The objects are in the air and nothing touches but the joints.</description></item>
    /// <item><description><c>Step, impact</c>: the 300 steps (5 s of simulated time) after that. The first voxels touch at about step 13 (the ones of neighboring objects on the ring), all six objects touch by step 22; they break up and fall on the ground.</description></item>
    /// <item><description>Counts: bodies, joints, joints broken at the end, and the manifold pool (slots in use and overflow at the end).</description></item>
    /// </list>
    /// It runs in the Play Mode tab of the Test Runner on a GPU. The results go to the Performance Test Report window (Window > Analysis > Performance
    /// Test Report) and to the test result XML. <see cref="Scene_HasAbout35kVoxelsAnd240kJoints"/> checks the size of the scene and needs no GPU.
    /// </summary>
    [TestFixture]
    [Category("Performance")]
    public sealed class VoxelImpactPerformanceTests
    {
        const int ApproachSteps = 12;   // after the first step
        const int ImpactSteps = 300;    // after the approach

        [Test]
        public void Scene_HasAbout35kVoxelsAnd240kJoints()
        {
            VoxelImpactScene.Count(new VoxelImpactConfig(), out int voxels, out int joints);
            TestContext.Out.WriteLine("Voxel impact scene: " + voxels + " voxels, " + joints + " joints");
            Assert.AreEqual(34_880, voxels, "voxels");
            Assert.AreEqual(241_872, joints, "joints");
            Assert.That(voxels, Is.InRange(34_000, 36_000));
            Assert.That(joints, Is.InRange(235_000, 245_000));
        }

        [Test, Performance, Timeout(3_600_000)]
        public void Scenario6_VoxelImpact35k()
        {
            PerformanceHarness.RequireCompute();
            var config = new VoxelImpactConfig();
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
