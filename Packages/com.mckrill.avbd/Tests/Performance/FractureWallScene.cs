using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd.Tests.Performance
{
    /// <summary>
    /// The settings of the GPU Fracture Wall Demo that the performance tests use. They are the demo's defaults, written out here on purpose: a
    /// test must not change when somebody changes a default in the sample. (Sample: <c>Samples~/FractureWallDemo/GpuFractureWallDemo.cs</c>.)
    /// </summary>
    internal sealed class FractureWallConfig
    {
        // Wall
        public int columns = 75;
        public int rows = 75;
        public int walls = 10;
        public float3 blockSize = new float3(1f, 0.5f, 0.5f);
        public float blockDensity = 1f;
        public bool rigidJoints = false;
        public float jointStiffness = 5e3f;
        public float fractureThreshold = 40f;

        // Plastic joints
        public bool plasticJoints = false;
        public float yieldThreshold = 8f;
        public bool bendBreak = true;
        public float bendBreakDegrees = 30f;

        // Solver
        public int iterations = 10;

        // Shots
        public float shotSpeed = 30f;
        public float shotSize = 2f;
        public float shotMass = 200f;
        public int maxShots = 100;

        /// <summary>Shots fired at once by <see cref="FractureWallScene.ShootVolley"/>.</summary>
        public int volleySize = 15;

        /// <summary>
        /// Spawn the ground and the blocks in the order the wall is built (row by row), the baseline for the comparison. False (the default) places them in
        /// Z-curve order of their positions (<see cref="AvbdBodyBuffers.SpawnBatch"/>).
        /// </summary>
        public bool builderOrder = false;

        public FractureWallConfig WithBuilderOrder(bool on)
        {
            builderOrder = on;
            return this;
        }

        public FractureWallConfig WithPlasticJoints(bool on)
        {
            plasticJoints = on;
            return this;
        }

        public FractureWallConfig WithShotSize(float size)
        {
            shotSize = size;
            return this;
        }

        public override string ToString()
        {
            return columns + " x " + rows + " x " + walls + " blocks, " + (plasticJoints ? "plastic yield " + yieldThreshold + (bendBreak ? ", bend break " + bendBreakDegrees + " deg" : "") : "no plastic yield")
                + ", shots " + shotSize + " m, " + (builderOrder ? "builder order" : "Z-curve order");
        }
    }

    /// <summary>
    /// The fracture wall of the demo without the sample's rendering and input: the same bodies, joints, capacities and solver settings, built through the
    /// package's public API. The wall stands in front (the first wall at z = 0, the others behind it 4 m apart), blocks are 1 x 0.5 x 0.5 m in a running bond,
    /// the bottom row is immobile, and every block is joined to the block on its left and to the two blocks under it by a ball-socket joint with a rigid
    /// angle lock that breaks above the fracture threshold (and yields first, if the joints are plastic).
    /// </summary>
    internal sealed class FractureWallScene : IDisposable
    {
        const float WallSpacing = 4f; // meters between the walls along z
        const float GroundHalfExtent = 100f;
        const float BlockGap = 0.01f; // between neighbors: the wall starts almost at rest

        // The volley: a grid of shots in front of the first wall, all fired together at +z. Columns are spread over the width of the wall
        // (it is about 76 m wide and 38 m high), rows over its height; the spacing is far above the biggest shot, so no two shots touch.
        static readonly float[] VolleyX = { -30f, -15f, 0f, 15f, 30f };
        static readonly float[] VolleyY = { 6f, 19f, 32f };
        const float VolleyZ = -10f; // meters in front of the first wall

        public readonly FractureWallConfig Config;
        public AvbdBodyBuffers Bodies { get; private set; }
        public AvbdConstraintBuffers Constraints { get; private set; }
        public AvbdSolver Solver { get; private set; }

        /// <summary>Blocks of all the walls.</summary>
        public int BlockCount { get; private set; }

        /// <summary>Joints that were added (a joint is not added if the buffers are full).</summary>
        public int JointCount { get; private set; }

        public int ShotCount { get; private set; }

        public FractureWallScene(FractureWallConfig config)
        {
            Config = config;
            Build();
        }

        void Build()
        {
            FractureWallConfig c = Config;
            BlockCount = c.columns * c.rows * c.walls;
            int capacity = 1 + BlockCount + c.maxShots + 8; // the ground, the blocks and the shots
            Bodies = new AvbdBodyBuffers(capacity);

            AvbdSolverSettings settings = AvbdSolverSettings.Default; // 60 Hz, gravity (0, -10, 0)
            settings.iterations = c.iterations;
            // Up to three joints per block. Manifold slots as in the demo: room for what a block and a shot can touch.
            int manifoldSlots = Mathf.Min(BlockCount * 8 + c.maxShots * 16 + 256, AvbdConstraintBuffers.MaxContactManifolds);
            int jointCapacity = BlockCount * 3 + 8;
            Constraints = new AvbdConstraintBuffers(Bodies, jointCapacity, null, manifoldSlots, c.plasticJoints ? jointCapacity : 0);
            Solver = new AvbdSolver(Bodies, settings, null, Constraints);

            // The ground and every block, in one batch: the ground is element 0, then the blocks wall by wall, row by row, column by column.
            // Z-curve order (unless the config asks for the builder's) gives the slots; the handles come back in this list's order.
            int perWall = c.rows * c.columns;
            var descriptors = new List<BodyDescriptor>(1 + BlockCount);
            var positions = new float3[c.walls * perWall];
            descriptors.Add(BodyDescriptor.Box(new float3(GroundHalfExtent * 2f, 1f, GroundHalfExtent * 2f), 0f, new float3(0f, -0.5f, 0f))); // a static slab whose top face is the plane y = 0

            float3 size = c.blockSize;
            float stepX = size.x + BlockGap;
            float stepY = size.y + BlockGap;
            float centerColumn = (c.columns - 1) * 0.5f + 0.25f;
            for (int w = 0; w < c.walls; w++)
            {
                float z = w * WallSpacing;
                for (int r = 0; r < c.rows; r++)
                {
                    float y = size.y * 0.5f + BlockGap + r * stepY;
                    for (int col = 0; col < c.columns; col++)
                    {
                        var position = new float3((col + 0.5f * (r & 1) - centerColumn) * stepX, y, z);
                        positions[w * perWall + r * c.columns + col] = position;
                        // The bottom row is immobile: a body without mass is static.
                        descriptors.Add(BodyDescriptor.Box(size, r == 0 ? 0f : c.blockDensity, position));
                    }
                }
            }
            var spawned = new BodyHandle[descriptors.Count];
            Bodies.SpawnBatch(descriptors, spawned, !c.builderOrder);

            float yield = c.plasticJoints ? c.yieldThreshold : float.PositiveInfinity;
            float bendLimit = c.plasticJoints && c.bendBreak ? Mathf.Min(c.bendBreakDegrees * Mathf.Deg2Rad, Mathf.PI) : float.PositiveInfinity;
            float linear = c.rigidJoints ? float.PositiveInfinity : c.jointStiffness;

            for (int w = 0; w < c.walls; w++)
            {
                int first = w * perWall;
                BodyHandle HandleOf(int r, int col) => spawned[1 + first + r * c.columns + col];
                float3 PositionOf(int r, int col) => positions[first + r * c.columns + col];

                // A block is joined to the block on its left and to the two blocks under it: with a shift of half a block, those are (c, c + 1) in an
                // odd row and (c - 1, c) in an even one. The bottom row is not joined to itself.
                for (int r = 1; r < c.rows; r++)
                    for (int col = 0; col < c.columns; col++)
                    {
                        if (col > 0)
                            Join(HandleOf(r, col), PositionOf(r, col), HandleOf(r, col - 1), PositionOf(r, col - 1), linear, yield, bendLimit);
                        int firstBelow = (r & 1) == 1 ? col : col - 1;
                        for (int k = 0; k < 2; k++)
                        {
                            int below = firstBelow + k;
                            if (below >= 0 && below < c.columns)
                                Join(HandleOf(r, col), PositionOf(r, col), HandleOf(r - 1, below), PositionOf(r - 1, below), linear, yield, bendLimit);
                        }
                    }
            }
            Bodies.ApplyEdits();
        }

        // A ball-socket joint at the middle between the two blocks, with a rigid angle lock. Both anchors are that point, so the joint starts satisfied.
        void Join(BodyHandle a, float3 positionA, BodyHandle b, float3 positionB, float linear, float yield, float bendLimit)
        {
            float3 anchor = (positionA + positionB) * 0.5f;
            if (Constraints.AddJoint(a, b, anchor - positionA, anchor - positionB, linear, float.PositiveInfinity, Config.fractureThreshold, yield, bendLimit) >= 0)
                JointCount++;
        }

        /// <summary>
        /// Fires <see cref="FractureWallConfig.volleySize"/> cubes at the first wall at the same time: from a grid in front of it, all at the shot speed
        /// along +z, each with the shot mass (so a smaller cube is denser). Returns the number that were spawned.
        /// </summary>
        public int ShootVolley()
        {
            FractureWallConfig c = Config;
            float size = c.shotSize;
            float density = c.shotMass / (size * size * size);
            int fired = 0;
            for (int i = 0; i < c.volleySize; i++)
            {
                float x = VolleyX[i % VolleyX.Length];
                float y = VolleyY[(i / VolleyX.Length) % VolleyY.Length];
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
