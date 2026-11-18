using System;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd.Tests.Performance
{
    /// <summary>
    /// The brick ring of three-avbd (<c>brickRing</c> in <c>src/avbd3d/bench-scenes.ts</c>, paper Fig. 1) with its defaults: 40 courses of bricks laid
    /// tangentially in running bond on rings from a radius of 80 m outwards, 10 bricks deep at the bottom and one fewer every 4 courses, so the outside is
    /// stepped and the inside sheer; 110 440 bricks by this formula (three-avbd's comment says 110 332), 20 m high, 160 m across. The bricks rest exactly on each other, so it stands from the first step.
    /// A ball is placed just outside the stepped face, on its way in.
    ///
    /// Differences from the original: the world is y-up (the ring's plane is x-z, the original's (x, y, z) is here (x, z, -y)), the ground's top face is at
    /// y = 0 instead of z = 0.5, and the ball is a cube: the package has no sphere shape. The cube has the volume and the density of the 8 m sphere, so
    /// it has its mass, and it slides instead of rolling.
    /// </summary>
    internal sealed class BrickRingScene : IDisposable
    {
        public const float Radius = 80f;
        public const int Courses = 40;
        public const int Tier = 4;
        public const int Rows = 10;

        const float BrickLength = 1f;
        const float BrickHeight = 0.5f;
        const float BrickDensity = 1f;
        const float BallDensity = 10f;
        const float BallSpeed = 30f;

        public AvbdBodyBuffers Bodies { get; private set; }
        public AvbdConstraintBuffers Constraints { get; private set; }
        public AvbdSolver Solver { get; private set; }

        /// <summary>The bricks (not the ground and the ball).</summary>
        public int BrickCount { get; private set; }

        /// <summary>Manifold slots of the contact pool: its largest possible size.</summary>
        public int ManifoldSlots { get; private set; }

        /// <param name="iterations">The solver iterations per step: 4 in the original's benchmark.</param>
        public BrickRingScene(int iterations)
        {
            int bricks = CountBricks();
            int capacity = 1 + bricks + 1 + 8; // the ground, the bricks, the ball
            Bodies = new AvbdBodyBuffers(capacity);

            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            settings.iterations = iterations;
            ManifoldSlots = AvbdConstraintBuffers.MaxContactManifolds;
            Constraints = new AvbdConstraintBuffers(Bodies, 1, null, ManifoldSlots); // no joints
            Solver = new AvbdSolver(Bodies, settings, null, Constraints);

            float extent = Radius + Rows;
            float ground = Mathf.Max(200f, 4f * extent); // as in the original: room for the debris to scatter
            Bodies.Spawn(BodyDescriptor.Box(new float3(ground, 1f, ground), 0f, new float3(0f, -0.5f, 0f)));

            for (int c = 0; c < Courses; c++)
            {
                int depth = Mathf.Max(1, Rows - c / Tier);
                for (int j = 0; j < depth; j++)
                {
                    float r = Radius + 0.25f + 0.52f * j; // rings 2 cm apart: a straight brick's corners reach past its ring's outer radius
                    int n = RingCount(r);
                    for (int i = 0; i < n; i++)
                    {
                        float a = (i + (c % 2) * 0.5f) / n * 2f * Mathf.PI; // odd courses are offset by half a brick
                        // The original's (x, y, z) is (x, z, -y) here: a proper rotation, which turns a turn about z into the same turn about y.
                        BodyDescriptor brick = BodyDescriptor.Box(new float3(BrickLength, BrickHeight, BrickHeight), BrickDensity,
                            new float3(r * Mathf.Cos(a), BrickHeight * 0.5f + BrickHeight * c, -r * Mathf.Sin(a)));
                        brick.rotation = quaternion.AxisAngle(new float3(0f, 1f, 0f), a + Mathf.PI * 0.5f);
                        if (Bodies.Spawn(brick).IsValid)
                            BrickCount++;
                    }
                }
            }

            // The ball: a cube of the volume of a sphere of radius courses / 5 (8 m), just outside the stepped face, moving in at 30 m/s.
            float ballRadius = Courses / 5f;
            float side = Mathf.Pow(4f / 3f * Mathf.PI, 1f / 3f) * ballRadius;
            float outside = Radius + 0.52f * Rows + ballRadius + 2f;
            BodyDescriptor ball = BodyDescriptor.Box(new float3(side), BallDensity, new float3(0f, side * 0.5f, outside));
            ball.linearVelocity = new float3(0f, 0f, -BallSpeed);
            Bodies.Spawn(ball);

            Bodies.ApplyEdits();
        }

        /// <summary>Bricks on a ring of radius r: as many as fit on its inner edge with 2% gaps.</summary>
        static int RingCount(float r) => Mathf.FloorToInt(2f * Mathf.PI * (r - 0.25f) / 1.02f);

        /// <summary>The number of bricks the constructor will spawn (110 440 with the defaults).</summary>
        public static int CountBricks()
        {
            int total = 0;
            for (int c = 0; c < Courses; c++)
            {
                int depth = Mathf.Max(1, Rows - c / Tier);
                for (int j = 0; j < depth; j++)
                    total += RingCount(Radius + 0.25f + 0.52f * j);
            }
            return total;
        }

        public void Dispose()
        {
            Solver?.Dispose();
            Constraints?.Dispose();
            Bodies?.Dispose();
            Solver = null;
            Constraints = null;
            Bodies = null;
        }
    }
}
