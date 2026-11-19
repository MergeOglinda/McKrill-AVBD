using NUnit.Framework;
using Unity.Mathematics;

namespace McKrill.Avbd.Tests.Performance
{
    /// <summary>
    /// A small check of the sphere wall that the performance scenarios time (not a performance test): the wall must stand at rest, hold together, and
    /// break under the volley, or the scenarios would time a wall that falls apart by itself or one the shots pass through. A 15 x 15 wall of one wall.
    /// </summary>
    [TestFixture]
    public sealed class SphereWallSceneTests
    {
        // The first float4 of a body state record is its position (the performance harness has the same 64 B element).
        struct StateElement
        {
            public float4 position, rotation, startPosition, startRotation;
        }

        [Test, Timeout(300_000)]
        public void SphereWall_StandsAtRest_AndTheVolleyBreaksJoints()
        {
            PerformanceHarness.RequireCompute();
            SphereWallConfig config = new SphereWallConfig().WithWallSize(15, 15, 1);
            using (var scene = new SphereWallScene(config))
            {
                Assert.AreEqual(15 * 15, scene.SphereCount);
                // 14 rows of 15 spheres are joined (the bottom row is static): up to three joints each, fewer at the ends of a row.
                Assert.Greater(scene.JointCount, 14 * 15 * 2, "at least the joints to the left and to the row below");
                Assert.LessOrEqual(scene.JointCount, 14 * 15 * 3);
                Assert.AreEqual(0, scene.Constraints.OverflowCount, "the joint buffers must hold every joint");

                var state = new StateElement[scene.Bodies.State.count];
                for (int i = 0; i < 60; i++)
                    scene.Solver.Step();
                scene.Bodies.State.GetData(state);

                float worst = 0f;
                for (int i = 0; i < scene.SphereCount; i++)
                    worst = math.max(worst, math.length(state[scene.Spheres[i].Slot].position.xyz - scene.SpherePositions[i]));
                Assert.Less(worst, 0.05f, "no sphere moves more than 5 cm in a second at rest");
                Assert.AreEqual(0, scene.Constraints.BrokenCount, "no joint breaks at rest");
                Assert.AreEqual(0, scene.Constraints.ReadSolveFailures(), "no solve failure at rest");

                Assert.AreEqual(config.volleySize, scene.ShootVolley());
                for (int i = 0; i < 240; i++)
                    scene.Solver.Step();

                Assert.Greater(scene.Constraints.BrokenCount, 0, "the volley (15 cubes of 200 kg at 30 m/s) breaks joints of a wall of spheres of 0.26 kg");
                Assert.AreEqual(0, scene.Constraints.ReadSolveFailures(), "no solve failure under the volley");
                Assert.AreEqual(0, scene.Bodies.OverflowCount);
            }
        }
    }
}
