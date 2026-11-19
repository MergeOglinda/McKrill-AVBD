using System;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd.Tests.Performance
{
    /// <summary>
    /// The settings of the sphere version of the GPU Fracture Wall Demo that the performance tests use. They are the demo's defaults, written out here on
    /// purpose: a test must not change when somebody changes a default in the sample. (Sample: <c>Samples~/FractureWallSphereDemo/GpuFractureWallSphereDemo.cs</c>.)
    /// The numbers that do not depend on the shape (joints, solver, shots) are those of <see cref="FractureWallConfig"/>, so a sphere scenario can be set
    /// against its box twin.
    /// </summary>
    internal sealed class SphereWallConfig
    {
        // Wall
        public int columns = 75;
        public int rows = 75;
        public int walls = 10;

        /// <summary>Radius of a sphere: 0.5 m, so a row is as wide as a row of the 1 m bricks of the box wall (the spheres in a row touch).</summary>
        public float radius = 0.5f;

        /// <summary>Density of a sphere. A sphere of 0.5 m radius has twice the volume of a 1 x 0.5 x 0.5 m brick, so half the density gives about the same mass (0.26 kg against 0.25 kg).</summary>
        public float density = 0.5f;

        public bool rigidJoints = false;
        public float jointStiffness = 5e3f;
        public float fractureThreshold = 19f;

        // Plastic joints
        public bool plasticJoints = false;
        public float yieldThreshold = 8f;
        public bool bendBreak = true;
        public float bendBreakDegrees = 30f;

        // Solver
        public int iterations = 10;

        // Shots: cubes, as in the box wall, so that the same volley hits both walls.
        public float shotSpeed = 30f;
        public float shotSize = 2f;
        public float shotMass = 200f;
        public int maxShots = 100;

        /// <summary>Shots fired at once by <see cref="SphereWallScene.ShootVolley"/>.</summary>
        public int volleySize = 15;

        public SphereWallConfig WithPlasticJoints(bool on)
        {
            plasticJoints = on;
            return this;
        }

        public SphereWallConfig WithShotSize(float size)
        {
            shotSize = size;
            return this;
        }

        public SphereWallConfig WithWallSize(int columnCount, int rowCount, int wallCount)
        {
            columns = columnCount;
            rows = rowCount;
            walls = wallCount;
            return this;
        }

        public override string ToString()
        {
            return columns + " x " + rows + " x " + walls + " spheres of radius " + radius + " m, " + (plasticJoints ? "plastic yield " + yieldThreshold + (bendBreak ? ", bend break " + bendBreakDegrees + " deg" : "") : "no plastic yield")
                + ", shots " + shotSize + " m";
        }
    }

    /// <summary>
    /// The sphere wall of the sphere demo without the sample's rendering and input: the same bodies, joints, capacities and solver settings, built through the
    /// package's public API. It is the fracture wall of <see cref="FractureWallScene"/> with a sphere in place of every brick: the spheres of a row touch, the
    /// odd rows are shifted by half a sphere (the spheres of a row then rest in the gaps of the row under it, a hexagonal packing, so the rows are
    /// sqrt(3) / 2 of a sphere apart), the bottom row is immobile, and every sphere is joined to the sphere on its left and to the two spheres under it by a
    /// ball-socket joint with a rigid angle lock that breaks above the fracture threshold (and yields first, if the joints are plastic). The ground is still a
    /// static box, and the shots are cubes: the scene has sphere-sphere contacts (the wall) and sphere-box contacts (the shots, the ground).
    /// </summary>
    internal sealed class SphereWallScene : IDisposable
    {
        const float WallSpacing = 4f; // meters between the walls along z
        const float GroundHalfExtent = 100f;
        const float Gap = 0.01f; // between neighbors: the wall starts almost at rest

        // The volley: a grid of shots in front of the first wall, all fired together at +z. Columns are spread over the width of the wall (about 76 m),
        // rows over the lower part of it (it is about 66 m high); the grid is scaled with the wall like in the box scene. The spacing is far above the
        // biggest shot, so no two shots touch.
        static readonly float[] VolleyX = { -30f, -15f, 0f, 15f, 30f };
        static readonly float[] VolleyY = { 6f, 19f, 32f };
        const float VolleyZ = -10f; // meters in front of the first wall

        public readonly SphereWallConfig Config;
        public AvbdBodyBuffers Bodies { get; private set; }
        public AvbdConstraintBuffers Constraints { get; private set; }
        public AvbdSolver Solver { get; private set; }

        /// <summary>Spheres of all the walls.</summary>
        public int SphereCount { get; private set; }

        /// <summary>The handles of the spheres, wall by wall, row by row, column by column.</summary>
        public BodyHandle[] Spheres { get; private set; }

        /// <summary>Where the spheres were spawned, in the order of <see cref="Spheres"/>.</summary>
        public float3[] SpherePositions { get; private set; }

        /// <summary>Joints that were added (a joint is not added if the buffers are full).</summary>
        public int JointCount { get; private set; }

        public int ShotCount { get; private set; }

        public SphereWallScene(SphereWallConfig config)
        {
            Config = config;
            Build();
        }

        void Build()
        {
            SphereWallConfig c = Config;
            SphereCount = c.columns * c.rows * c.walls;
            int capacity = 1 + SphereCount + c.maxShots + 8; // the ground, the spheres and the shots
            Bodies = new AvbdBodyBuffers(capacity);

            AvbdSolverSettings settings = AvbdSolverSettings.Default; // 60 Hz, gravity (0, -10, 0)
            settings.iterations = c.iterations;
            // Up to three joints per sphere. Manifold slots as in the box scene: room for what a sphere and a shot can touch.
            int manifoldSlots = Mathf.Min(SphereCount * 8 + c.maxShots * 16 + 256, AvbdConstraintBuffers.MaxContactManifolds);
            int jointCapacity = SphereCount * 3 + 8;
            Constraints = new AvbdConstraintBuffers(Bodies, jointCapacity, null, manifoldSlots, c.plasticJoints ? jointCapacity : 0);
            Solver = new AvbdSolver(Bodies, settings, null, Constraints);

            // Ground: a static slab whose top face is the plane y = 0.
            Bodies.Spawn(BodyDescriptor.Box(new float3(GroundHalfExtent * 2f, 1f, GroundHalfExtent * 2f), 0f, new float3(0f, -0.5f, 0f)));

            float yield = c.plasticJoints ? c.yieldThreshold : float.PositiveInfinity;
            float bendLimit = c.plasticJoints && c.bendBreak ? Mathf.Min(c.bendBreakDegrees * Mathf.Deg2Rad, Mathf.PI) : float.PositiveInfinity;
            float linear = c.rigidJoints ? float.PositiveInfinity : c.jointStiffness;

            float stepX = 2f * c.radius + Gap;
            float stepY = stepX * 0.8660254f; // sqrt(3) / 2: the spheres of two rows touch (as far as the gap lets them) along the slanted lines
            float centerColumn = (c.columns - 1) * 0.5f + 0.25f;
            int perWall = c.rows * c.columns;
            Spheres = new BodyHandle[SphereCount];
            SpherePositions = new float3[SphereCount];

            for (int w = 0; w < c.walls; w++)
            {
                float z = w * WallSpacing;
                int first = w * perWall;
                for (int r = 0; r < c.rows; r++)
                {
                    float y = c.radius + Gap + r * stepY;
                    for (int col = 0; col < c.columns; col++)
                    {
                        var position = new float3((col + 0.5f * (r & 1) - centerColumn) * stepX, y, z);
                        int i = first + r * c.columns + col;
                        // The bottom row is immobile: a body without mass is static.
                        Spheres[i] = Bodies.Spawn(BodyDescriptor.Sphere(c.radius, r == 0 ? 0f : c.density, position));
                        SpherePositions[i] = position;
                    }
                }

                // A sphere is joined to the sphere on its left and to the two spheres under it: with a shift of half a sphere, those are (c, c + 1) in an
                // odd row and (c - 1, c) in an even one. The bottom row is not joined to itself. Together with the joints of the neighbors, a sphere is
                // joined to all six spheres that touch it.
                for (int r = 1; r < c.rows; r++)
                    for (int col = 0; col < c.columns; col++)
                    {
                        int self = first + r * c.columns + col;
                        if (col > 0)
                            Join(self, self - 1, linear, yield, bendLimit);
                        int firstBelow = (r & 1) == 1 ? col : col - 1;
                        for (int k = 0; k < 2; k++)
                        {
                            int below = firstBelow + k;
                            if (below >= 0 && below < c.columns)
                                Join(self, first + (r - 1) * c.columns + below, linear, yield, bendLimit);
                        }
                    }
            }
            Bodies.ApplyEdits();
        }

        // A ball-socket joint at the middle between the two spheres, with a rigid angle lock. Both anchors are that point, so the joint starts satisfied.
        void Join(int a, int b, float linear, float yield, float bendLimit)
        {
            float3 anchor = (SpherePositions[a] + SpherePositions[b]) * 0.5f;
            if (Constraints.AddJoint(Spheres[a], Spheres[b], anchor - SpherePositions[a], anchor - SpherePositions[b], linear, float.PositiveInfinity,
                    Config.fractureThreshold, yield, bendLimit) >= 0)
                JointCount++;
        }

        /// <summary>
        /// Fires <see cref="SphereWallConfig.volleySize"/> cubes at the first wall at the same time: from a grid in front of it, all at the shot speed
        /// along +z, each with the shot mass (so a smaller cube is denser). The grid is the box scene's, scaled with the number of columns and rows
        /// (it is unchanged at 75 x 75). Returns the number that were spawned.
        /// </summary>
        public int ShootVolley()
        {
            SphereWallConfig c = Config;
            float size = c.shotSize;
            float density = c.shotMass / (size * size * size);
            int fired = 0;
            for (int i = 0; i < c.volleySize; i++)
            {
                float x = VolleyX[i % VolleyX.Length] * (c.columns / 75f);
                float y = VolleyY[(i / VolleyX.Length) % VolleyY.Length] * (c.rows / 75f);
                BodyDescriptor shot = BodyDescriptor.Box(new float3(size), density, new float3(x, y, VolleyZ));
                shot.linearVelocity = new float3(0f, 0f, c.shotSpeed);
                if (Bodies.Spawn(shot).IsValid)
                    fired++;
            }
            ShotCount += fired;
            return fired;
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
