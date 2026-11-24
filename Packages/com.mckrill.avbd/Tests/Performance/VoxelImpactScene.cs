using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd.Tests.Performance
{
    /// <summary>
    /// The settings of the voxel impact test. They are the JSON Mesh Import Demo's defaults, written out here on purpose (a test must not change when
    /// somebody changes a default in a sample), with the differences the scenario asks for: the linear fracture and the linear yield are on and the
    /// solver runs 25 iterations. (Samples: <c>Samples~/JsonMeshImportDemo/JsonMeshImportDemo.cs</c> and <c>Samples~/VoxelImpactDemo/VoxelImpactDemo.cs</c>.)
    /// </summary>
    internal sealed class VoxelImpactConfig
    {
        // Voxels
        public float voxelSize = 0.2f;
        public float voxelMass = 0.25f;     // kg of one voxel, whatever its size
        public float voxelGap = 0.01f;      // meters between the faces of neighboring boxes: the objects start almost at rest

        // Objects: skins of 2 voxels, in voxels. 24 / 12 / 11, 7 give 34 880 voxels and 241 872 joints for two of each.
        public int shellLayers = 2;
        public int cubeEdge = 24;           // 4.8 m
        public int sphereRadius = 12;       // 2.4 m
        public int torusMajorRadius = 11;   // 2.2 m: the middle of the tube
        public int torusMinorRadius = 7;    // 1.4 m: the tube

        // Impact: six objects on a ring, all flying at its middle
        public float ringRadius = 11f;      // meters from the meeting point to the middle of an object at the start
        public float meetingHeight = 8f;    // meters above the ground
        public float approachSpeed = 25f;   // m/s of every object: two objects that face each other close in on each other at twice that

        // Joints: rigid ball-sockets that yield and break, in force and in angle, and in force and in stretch
        public bool rigidJoints = true;
        public float jointStiffness = 5e3f;     // only for joints that are not rigid
        public float fractureThreshold = 600f;
        public bool plasticJoints = true;
        public float yieldThreshold = 400f;
        public bool bendBreak = true;
        public float bendBreakDegrees = 30f;
        public bool linearFracture = true;      // the JSON demo has it on as well
        public float linearFractureThreshold = 600f;
        public bool linearYield = true;
        public float linearYieldThreshold = 400f;
        public bool stretchBreak = true;
        public float stretchBreakMeters = 0.04f;

        // Solver
        public int iterations = 25;             // the JSON demo's is 10
        public float penaltyMin = 2000f;        // as in the JSON demo (the other settings are AvbdSolverSettings.Default, which the demo uses too)

        public override string ToString()
        {
            return "6 objects (2 cubes, 2 spheres, 2 toruses), voxels " + voxelSize + " m, skin " + shellLayers + " voxels, " + approachSpeed * 2f
                + " m/s head-on, " + iterations + " iterations, linear fracture " + (linearFracture ? linearFractureThreshold.ToString() : "off")
                + ", linear yield " + (linearYield ? linearYieldThreshold.ToString() : "off");
        }
    }

    /// <summary>
    /// The scene of the voxel impact test without the sample's rendering and input: the same bodies, joints, capacities and solver settings, built through the
    /// package's public API. Six hollow objects made of voxels (two cubes, two spheres and two toruses, in the order cube, sphere, torus, sphere, torus, cube
    /// around the ring) start on a ring of 11 m around a point 8 m above the ground, and every one flies at that point at 25 m/s. The three pairs that
    /// face each other meet head-on at 50 m/s, and all six meet at once. A voxel is a box, and the voxels of an object are joined like those of a file
    /// of the JSON Mesh Import Demo: one ball-socket joint with a rigid angle lock for every pair that shares a face or an edge, with the demo's
    /// fracture, yield, bend and stretch limits.
    ///
    /// The objects are not rotated: a joint's local anchors are then the offsets from the voxel centers to the middle between them, as in the demo.
    /// </summary>
    internal sealed class VoxelImpactScene : IDisposable
    {
        const float GroundHalfExtent = 100f;
        const int ObjectCount = 6;
        enum Kind { Cube, Sphere, Torus }
        static readonly Kind[] Ring = { Kind.Cube, Kind.Sphere, Kind.Torus, Kind.Sphere, Kind.Torus, Kind.Cube };

        public readonly VoxelImpactConfig Config;
        public AvbdBodyBuffers Bodies { get; private set; }
        public AvbdConstraintBuffers Constraints { get; private set; }
        public AvbdSolver Solver { get; private set; }

        /// <summary>Voxels of all the objects (bodies other than the ground).</summary>
        public int VoxelCount { get; private set; }

        /// <summary>Joints that the voxels call for.</summary>
        public int PlannedJointCount { get; private set; }

        /// <summary>Joints that were added (a joint is not added if the buffers are full).</summary>
        public int JointCount { get; private set; }

        public VoxelImpactScene(VoxelImpactConfig config)
        {
            Config = config;
            Build();
        }

        /// <summary>The number of voxels and of joints the scene has, without building it (no GPU needed).</summary>
        public static void Count(VoxelImpactConfig config, out int voxels, out int joints)
        {
            voxels = 0;
            joints = 0;
            VoxelShape[] shapes = MakeShapes(config);
            for (int o = 0; o < ObjectCount; o++)
            {
                VoxelShape shape = shapes[(int)Ring[o]];
                voxels += shape.Count;
                joints += shape.CountPairs();
            }
        }

        static VoxelShape[] MakeShapes(VoxelImpactConfig c)
        {
            var shapes = new VoxelShape[3];
            shapes[(int)Kind.Cube] = VoxelShape.Cube(c.cubeEdge).Shell(c.shellLayers);
            shapes[(int)Kind.Sphere] = VoxelShape.Sphere(c.sphereRadius).Shell(c.shellLayers);
            shapes[(int)Kind.Torus] = VoxelShape.Torus(c.torusMajorRadius, c.torusMinorRadius).Shell(c.shellLayers);
            return shapes;
        }

        void Build()
        {
            VoxelImpactConfig c = Config;
            VoxelShape[] shapes = MakeShapes(c);
            Count(c, out int voxelTotal, out int jointTotal);
            VoxelCount = voxelTotal;
            PlannedJointCount = jointTotal;

            Bodies = new AvbdBodyBuffers(1 + voxelTotal + 8); // the ground and the voxels

            AvbdSolverSettings settings = AvbdSolverSettings.Default; // 60 Hz, gravity (0, -10, 0)
            settings.penaltyMin = c.penaltyMin;
            settings.iterations = c.iterations;
            // Manifold slots as in the JSON demo: room for what the voxels can touch (the joined ones make no manifold).
            int manifoldSlots = Mathf.Min(voxelTotal * 8 + 256, AvbdConstraintBuffers.MaxContactManifolds);
            int jointCapacity = jointTotal + 8;
            // The plastic array only exists if the joints can be plastic: linear yield on a rigid joint needs it too.
            bool plastic = c.plasticJoints || (c.linearYield && c.rigidJoints);
            Constraints = new AvbdConstraintBuffers(Bodies, jointCapacity, null, manifoldSlots, plastic ? jointCapacity : 0);
            Solver = new AvbdSolver(Bodies, settings, null, Constraints);

            // The ground and every voxel go in one batch (the ground first, then the objects one after the other, their voxels in the order of the shape),
            // placed in Z-curve order like the JSON demo does.
            var descriptors = new List<BodyDescriptor>(1 + voxelTotal);
            descriptors.Add(BodyDescriptor.Box(new float3(GroundHalfExtent * 2f, 1f, GroundHalfExtent * 2f), 0f, new float3(0f, -0.5f, 0f)));

            // The box is a little smaller than the cell: neighbors that are not joined (those that touch along an edge or at a corner) start apart.
            float edge = c.voxelSize - Mathf.Min(c.voxelGap, 0.05f * c.voxelSize);
            var size = new float3(edge);
            float density = c.voxelMass / (edge * edge * edge);
            var positions = new float3[voxelTotal];
            int first = 0; // index in positions of the first voxel of the object
            for (int o = 0; o < ObjectCount; o++)
            {
                VoxelShape shape = shapes[(int)Ring[o]];
                float angle = o * (2f * Mathf.PI / ObjectCount);
                var toObject = new float3(Mathf.Cos(angle), 0f, Mathf.Sin(angle)); // from the meeting point to the object
                float3 center = new float3(0f, c.meetingHeight, 0f) + toObject * c.ringRadius;
                float3 velocity = -toObject * c.approachSpeed;
                for (int n = 0; n < shape.Count; n++)
                {
                    float3 position = center + shape.CellOffset(n) * c.voxelSize;
                    positions[first + n] = position;
                    BodyDescriptor voxel = BodyDescriptor.Box(size, density, position);
                    voxel.linearVelocity = velocity;
                    descriptors.Add(voxel);
                }
                first += shape.Count;
            }
            var spawned = new BodyHandle[descriptors.Count];
            Bodies.SpawnBatch(descriptors, spawned, true);

            // Joints: one for every pair, added in the order of the bodies' slots (the lower slot of the pair first, then the higher): see the JSON demo.
            var pairKeys = new ulong[jointTotal];
            var pairIds = new int[jointTotal];
            var pairA = new int[jointTotal]; // voxel numbers in the whole scene
            var pairB = new int[jointTotal];
            int pair = 0;
            first = 0;
            for (int o = 0; o < ObjectCount; o++)
            {
                VoxelShape shape = shapes[(int)Ring[o]];
                int[] pairs = shape.Pairs();
                for (int m = 0; m < pairs.Length; m += 2)
                {
                    int a = first + pairs[m];
                    int b = first + pairs[m + 1];
                    BodyHandle handleA = spawned[1 + a];
                    BodyHandle handleB = spawned[1 + b];
                    ulong key = ulong.MaxValue; // a pair with a body that was not spawned makes no joint (Join skips it): last
                    if (handleA.IsValid && handleB.IsValid)
                    {
                        uint low = (uint)Mathf.Min(handleA.Slot, handleB.Slot);
                        uint high = (uint)Mathf.Max(handleA.Slot, handleB.Slot);
                        key = ((ulong)low << 32) | high;
                    }
                    pairKeys[pair] = key;
                    pairIds[pair] = pair;
                    pairA[pair] = a;
                    pairB[pair] = b;
                    pair++;
                }
                first += shape.Count;
            }
            Array.Sort(pairKeys, pairIds);
            for (int m = 0; m < jointTotal; m++)
            {
                int a = pairA[pairIds[m]];
                int b = pairB[pairIds[m]];
                Join(spawned[1 + a], positions[a], spawned[1 + b], positions[b]);
            }
            Bodies.ApplyEdits();
        }

        // A ball-socket joint at the middle between the two voxels, with the demo's limits. Both anchors are that point, so the joint starts satisfied.
        void Join(BodyHandle a, float3 positionA, BodyHandle b, float3 positionB)
        {
            if (!a.IsValid || !b.IsValid)
                return;
            VoxelImpactConfig c = Config;
            float3 anchor = (positionA + positionB) * 0.5f;
            float linear = c.rigidJoints ? float.PositiveInfinity : c.jointStiffness;
            float yield = c.plasticJoints ? c.yieldThreshold : float.PositiveInfinity;
            float bendLimit = c.plasticJoints && c.bendBreak ? Mathf.Min(c.bendBreakDegrees * Mathf.Deg2Rad, Mathf.PI) : float.PositiveInfinity;
            // The linear options need a hard linear row (AddJoint throws otherwise), so they apply to rigid joints only.
            float fractureLinear = c.rigidJoints && c.linearFracture ? c.linearFractureThreshold : float.PositiveInfinity;
            float yieldLinear = c.rigidJoints && c.linearYield ? c.linearYieldThreshold : float.PositiveInfinity;
            float stretchLimit = c.rigidJoints && c.linearYield && c.stretchBreak ? c.stretchBreakMeters : float.PositiveInfinity;
            if (Constraints.AddJoint(a, b, anchor - positionA, anchor - positionB, linear, float.PositiveInfinity, c.fractureThreshold, yield, bendLimit,
                    fractureLinear, yieldLinear, stretchLimit) >= 0)
                JointCount++;
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
