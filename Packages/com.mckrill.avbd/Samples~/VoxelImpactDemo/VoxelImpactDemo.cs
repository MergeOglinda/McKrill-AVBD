using System.Collections.Generic;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd.Samples
{
    /// <summary>
    /// The scene of the performance test <c>Scenario6_VoxelImpact35k</c> (see <c>Tests/Performance/VoxelImpactScene.cs</c>): six hollow objects made of 0.2 m
    /// voxels, two cubes, two spheres and two toruses, 34 880 voxels and 241 872 joints together, that fly at one point from a ring around it at 25 m/s each
    /// and collide there all at once. The three pairs that face each other meet head-on at 50 m/s; neighbors on the ring close in at 25 m/s. The objects
    /// start 11 m from the point, 8 m above the ground, and the first voxels touch after about a fifth of a second.
    ///
    /// The voxels and joints are those of the JSON Mesh Import Demo, which this demo follows (every voxel is a box, neighbors that share a face or an edge are
    /// linked by a fracturing ball-socket joint with a rigid angle lock), with the settings of the test: rigid joints that break at 600 and yield at 400,
    /// in force and in angle (bend break at 30 degrees), the linear fracture and the linear yield on (stretch break at 4 cm), and 25 iterations per step. The
    /// shapes are generated (no file): the objects are skins two voxels thick (0.4 m), so the scene has few voxels for its size.
    /// At 50 m/s a pair of objects closes in by 0.83 m per step, a little more than the two skins are thick together (0.8 m): a deep first touch can
    /// push a voxel through the skin of the other object, and a few voxels get through. Raise the iterations or lower the speed to see fewer.
    ///
    /// The simulation runs entirely through the package's public API; drawing never leaves the GPU: one instanced draw call draws every box, and the vertex
    /// shader (<c>AvbdInstancedBoxes</c>) reads each box's pose from the solver's own body state buffer, as in the GPU Brick Ring Demo. The shader has no shadows.
    /// You can shoot boxes from the camera as well. Settings marked "Reset to apply" are read when the scene is built.
    ///
    /// The scene is y-up (the package default), so solver positions and rotations are used as Unity world positions and rotations.
    /// Input is read through IMGUI events, so it works with the Input Manager and with the Input System package.
    /// </summary>
    [AddComponentMenu("McKrill AVBD/Voxel Impact Demo")]
    [DisallowMultipleComponent]
    public sealed class VoxelImpactDemo : MonoBehaviour
    {
        // Mirror of the package's state record (design doc, section 3): position, rotation, start-of-step position and rotation.
        // The package keeps its own struct internal; the demo only checks the stride of the public buffer against this copy (the draw shader reads the buffer).
        struct BodyStateRecord
        {
            public float4 position;
            public float4 rotation;
            public float4 startPosition;
            public float4 startRotation;
        }

        // What the shader reads for each box (32 B, mirrors Instance in AvbdInstancedBoxes.shader): the body slot, the size and the color.
        struct InstanceData
        {
            public uint slot;
            public float sizeX, sizeY, sizeZ;
            public float4 color;
        }

        struct Shot
        {
            public BodyHandle handle;
            public float size;
        }

        enum Kind { Cube, Sphere, Torus }

        [Header("Impact (Reset to apply)")]
        [Tooltip("Speed of every object, flying at the meeting point. Two objects that face each other close in at twice this: 25 m/s is 50 m/s head-on.")]
        [SerializeField, Range(0f, 60f)] float approachSpeed = 25f;
        [Tooltip("Thickness of the skin of an object in voxels. 2 gives 34 880 voxels and 241 872 joints.")]
        [SerializeField, Range(1, 4)] int shellLayers = 2;

        [Header("Joints (Reset to apply)")]
        [Tooltip("Joints that give way for good above the yield threshold (force and angle) before they break.")]
        [SerializeField] bool plasticJoints = true;
        [SerializeField, Min(0.01f)] float fractureThreshold = 600f;
        [SerializeField, Min(0.1f)] float yieldThreshold = 400f;
        [SerializeField] bool bendBreak = true;
        [SerializeField, Range(1f, 180f)] float bendBreakDegrees = 30f;
        [Tooltip("Joints break when the force on their linear lock is above the linear fracture threshold.")]
        [SerializeField] bool linearFracture = true;
        [SerializeField, Min(0.01f)] float linearFractureThreshold = 600f;
        [Tooltip("Joints give way for good when the force on their linear lock is above the linear yield threshold.")]
        [SerializeField] bool linearYield = true;
        [SerializeField, Min(0.1f)] float linearYieldThreshold = 400f;
        [SerializeField] bool stretchBreak = true;
        [SerializeField, Range(0.01f, 2f)] float stretchBreakMeters = 0.04f;

        [Header("Solver")]
        [Tooltip("Solver iterations per step: 25 in the performance test.")]
        [SerializeField, Range(1, 30)] int iterations = 25;
        [Tooltip("Do the dual update of the joints and the contacts in the next primal sweep (one pass over the joint records and the manifolds fewer per iteration).")]
        [SerializeField] bool fuseDualUpdate = true;
        [Tooltip("Upper bound on solver steps per rendered frame. If the GPU cannot keep up, the simulation runs slower than real time instead of stalling.")]
        [SerializeField, Range(1, 4)] int maxStepsPerFrame = 1;

        [Header("Shots")]
        [Tooltip("Speed of a shot in meters per second.")]
        [SerializeField, Range(5f, 80f)] float shotSpeed = 30f;
        [Tooltip("Edge length of the shot, a cube.")]
        [SerializeField, Range(0.2f, 4f)] float shotSize = 1f;
        [Tooltip("Mass of the shot. A voxel weighs 0.25 kg.")]
        [SerializeField, Range(0.1f, 2000f)] float shotMass = 50f;
        [Tooltip("Shots alive at once. Shooting more removes the oldest.")]
        [SerializeField, Min(1)] int maxShots = 100;

        [Header("View")]
        [Tooltip("Fly camera: right drag to look, W A S D to move, Q and E for down and up, Shift for faster, mouse wheel to dolly.")]
        [SerializeField] bool controlCamera = true;
        [SerializeField, Min(0.1f)] float moveSpeed = 12f;
        [SerializeField, Min(0.01f)] float lookSensitivity = 0.2f;

        // The scene of the test, written out here on purpose (the test has its own copy of these numbers): see VoxelImpactConfig.
        const float VoxelSize = 0.2f;
        const float VoxelMass = 0.25f;
        const float VoxelGap = 0.01f;
        const int CubeEdge = 24;
        const int SphereRadius = 12;
        const int TorusMajorRadius = 11;
        const int TorusMinorRadius = 7;
        const float RingRadius = 11f;
        const float MeetingHeight = 8f;
        const float GroundHalfExtent = 100f;
        const float PenaltyMin = 2000f;
        const int ObjectCount = 6;
        static readonly Kind[] Ring = { Kind.Cube, Kind.Sphere, Kind.Torus, Kind.Sphere, Kind.Torus, Kind.Cube };

        AvbdBodyBuffers bodies;
        AvbdConstraintBuffers constraints;
        AvbdSolver solver;
        // Drawing: one instance per body. The first staticCount instances (the ground, the voxels) are written once per Reset, the shots follow.
        InstanceData[] instances;
        GraphicsBuffer instanceBuffer;
        int staticCount;
        int voxelCount;
        int jointCount;
        int stepCount;
        int brokenJoints = -1;  // read back on request: the readback waits for the GPU
        bool shotsChanged;
        float buildMs;

        readonly List<InstanceData> staticInstances = new List<InstanceData>(); // ground, voxels
        readonly List<Shot> shots = new List<Shot>();                            // oldest first
        Material boxMaterial;
        Mesh cubeMesh;
        GameObject meshSource;

        Camera cam;
        bool createdCamera;
        Vector3 savedCameraPosition;
        Quaternion savedCameraRotation;
        GameObject createdLight;
        Vector3 camPosition;
        float yaw;
        float pitch = 10f;
        readonly HashSet<KeyCode> held = new HashSet<KeyCode>();

        float accumulator;
        bool paused;
        bool behindRealTime;
        bool tunnelRisk; // set in Update, not in OnGUI: a control that appears or disappears between Layout and Repaint is an IMGUI error
        float stepMs;
        float frameMs;

        // Requests made from OnGUI, applied in Update so the IMGUI control count never changes between Layout and Repaint.
        bool requestShoot;
        bool requestReset;
        bool requestClearShots;
        bool requestStep;
        bool requestCountBroken;

        Rect panelScreenRect;
        GUIStyle boldLabel;

        // ---- lifecycle ---------------------------------------------------------------------------------

        void OnEnable()
        {
            Shader boxShader = AvbdShaderSet.Load()?.InstancedBoxes;
            if (boxShader == null || !boxShader.isSupported)
            {
                Debug.LogError("Voxel Impact Demo: the package's instanced box shader is missing or not supported on this device (it needs compute buffers in the vertex shader, shader model 4.5).");
                enabled = false;
                return;
            }
            boxMaterial = new Material(boxShader) { hideFlags = HideFlags.HideAndDontSave };

            // The cube mesh comes from a primitive that is destroyed again: it keeps the demo free of mesh assets.
            meshSource = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cubeMesh = meshSource.GetComponent<MeshFilter>().sharedMesh;
            meshSource.SetActive(false);

            SetUpCameraAndLight();
            BuildSimulation();

            // Camera: 38 m from the meeting point, to the side of the ring and a little above it, looking at the point. Reset leaves it where it is.
            camPosition = new Vector3(-20f, MeetingHeight + 6f, -32f);
            Vector3 target = new Vector3(0f, MeetingHeight - 1f, 0f);
            Vector3 euler = Quaternion.LookRotation(target - camPosition).eulerAngles;
            yaw = euler.y;
            pitch = euler.x > 180f ? euler.x - 360f : euler.x;
        }

        void OnDisable()
        {
            DestroySimulation();
            if (meshSource != null) Destroy(meshSource);
            meshSource = null;
            cubeMesh = null;
            if (boxMaterial != null) Destroy(boxMaterial);
            boxMaterial = null;

            if (createdLight != null) Destroy(createdLight);
            createdLight = null;
            if (cam != null)
            {
                if (createdCamera) Destroy(cam.gameObject);
                else if (controlCamera) cam.transform.SetPositionAndRotation(savedCameraPosition, savedCameraRotation);
            }
            cam = null;
            createdCamera = false;
            held.Clear();
        }

        void OnApplicationFocus(bool focus)
        {
            if (!focus)
                held.Clear(); // a key released while the window had no focus never sends its KeyUp
        }

        void Update()
        {
            if (bodies == null)
                return;

            frameMs = Mathf.Lerp(frameMs, Time.unscaledDeltaTime * 1000f, 0.05f);

            if (requestReset)
            {
                BuildSimulation();
                requestReset = false;
                requestShoot = false;
                requestClearShots = false;
                requestCountBroken = false;
            }
            if (requestClearShots)
            {
                ClearShots();
                requestClearShots = false;
            }
            if (requestShoot)
            {
                Shoot();
                requestShoot = false;
            }
            if (requestCountBroken)
            {
                brokenJoints = constraints.BrokenCount;
                requestCountBroken = false;
            }

            MoveCamera();
            tunnelRisk = shotSpeed * solver.Settings.dt > shotSize;

            // The settings take effect at the next step.
            AvbdSolverSettings settings = solver.Settings;
            if (settings.iterations != iterations)
            {
                settings.iterations = iterations;
                solver.Settings = settings;
            }
            solver.FuseDualUpdate = fuseDualUpdate;

            // Make spawns and removals visible to the readback below, also while paused.
            bodies.ApplyEdits();

            if (requestStep)
            {
                StepOnce();
                requestStep = false;
            }
            else if (!paused)
            {
                float dt = solver.Settings.dt;
                accumulator += Time.deltaTime;
                int steps = 0;
                while (accumulator >= dt && steps < maxStepsPerFrame)
                {
                    StepOnce();
                    accumulator -= dt;
                    steps++;
                }
                behindRealTime = accumulator >= dt;
                if (behindRealTime)
                    accumulator = 0f;
            }

            DrawBodies();
        }

        void LateUpdate()
        {
            if (!controlCamera || cam == null)
                return;
            cam.transform.SetPositionAndRotation(camPosition, Quaternion.Euler(pitch, yaw, 0f));
        }

        // ---- simulation --------------------------------------------------------------------------------

        // The skins of the three kinds of object, in voxels (see VoxelShape).
        VoxelShape[] MakeShapes()
        {
            var shapes = new VoxelShape[3];
            shapes[(int)Kind.Cube] = VoxelShape.Cube(CubeEdge).Shell(shellLayers);
            shapes[(int)Kind.Sphere] = VoxelShape.Sphere(SphereRadius).Shell(shellLayers);
            shapes[(int)Kind.Torus] = VoxelShape.Torus(TorusMajorRadius, TorusMinorRadius).Shell(shellLayers);
            return shapes;
        }

        void BuildSimulation()
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            DestroySimulation();

            shellLayers = Mathf.Clamp(shellLayers, 1, 4);
            VoxelShape[] shapes = MakeShapes();
            int voxelTotal = 0;
            int jointTotal = 0;
            for (int o = 0; o < ObjectCount; o++)
            {
                voxelTotal += shapes[(int)Ring[o]].Count;
                jointTotal += shapes[(int)Ring[o]].CountPairs();
            }

            // Bodies: the ground, the voxels and the shots.
            maxShots = Mathf.Max(maxShots, 1);
            int capacity = 1 + voxelTotal + maxShots + 8;
            bodies = new AvbdBodyBuffers(capacity);
            if (bodies.State.stride != UnsafeUtility.SizeOf<BodyStateRecord>())
                throw new System.InvalidOperationException("The package's body state record changed; update BodyStateRecord in VoxelImpactDemo.");
            staticInstances.Clear();

            AvbdSolverSettings settings = AvbdSolverSettings.Default; // 60 Hz, gravity (0, -10, 0): Unity's y-up
            settings.penaltyMin = PenaltyMin;
            settings.iterations = iterations;
            // Manifold slots as in the JSON demo: room for what the voxels can touch (the joined ones make no manifold), and what a shot can lie on.
            int manifoldSlots = Mathf.Min(voxelTotal * 8 + maxShots * 16 + 256, AvbdConstraintBuffers.MaxContactManifolds);
            // The plastic array only exists if the joints can be plastic: without it the solver runs the joint kernels with no plastic code at all.
            int jointCapacity = jointTotal + 8;
            constraints = new AvbdConstraintBuffers(bodies, jointCapacity, null, manifoldSlots, (plasticJoints || linearYield) ? jointCapacity : 0);
            constraints.ColoringReductionPasses = 4;
            solver = new AvbdSolver(bodies, settings, null, constraints);
            solver.FuseDualUpdate = fuseDualUpdate;

            // The ground and every voxel go in one batch (the ground first, then the objects one after the other, their voxels in the order of the shape),
            // placed in Z-curve order. The handles come back in the order of the list; the joints and the instances use them.
            var descriptors = new List<BodyDescriptor>(1 + voxelTotal);
            var colors = new List<Color>(1 + voxelTotal);
            var ground = new float3(GroundHalfExtent * 2f, 1f, GroundHalfExtent * 2f);
            descriptors.Add(BodyDescriptor.Box(ground, 0f, new float3(0f, -0.5f, 0f)));
            colors.Add(new Color(0.30f, 0.33f, 0.36f));

            // The box is a little smaller than the cell: neighbors that are not joined (those that touch along an edge or at a corner) start apart.
            float edge = VoxelSize - Mathf.Min(VoxelGap, 0.05f * VoxelSize);
            var voxelBox = new float3(edge);
            float density = VoxelMass / (edge * edge * edge);
            var positions = new float3[voxelTotal];
            int first = 0; // index in positions of the first voxel of the object
            for (int o = 0; o < ObjectCount; o++)
            {
                VoxelShape shape = shapes[(int)Ring[o]];
                float angle = o * (2f * Mathf.PI / ObjectCount);
                var toObject = new float3(Mathf.Cos(angle), 0f, Mathf.Sin(angle)); // from the meeting point to the object
                float3 center = new float3(0f, MeetingHeight, 0f) + toObject * RingRadius;
                float3 velocity = -toObject * approachSpeed;
                for (int n = 0; n < shape.Count; n++)
                {
                    float3 position = center + shape.CellOffset(n) * VoxelSize;
                    positions[first + n] = position;
                    BodyDescriptor voxel = BodyDescriptor.Box(voxelBox, density, position);
                    voxel.linearVelocity = velocity;
                    descriptors.Add(voxel);
                    int3 cell = shape.cells[n];
                    colors.Add(VoxelColor(o, cell.x, cell.y, cell.z));
                }
                first += shape.Count;
            }
            var spawned = new BodyHandle[descriptors.Count];
            bodies.SpawnBatch(descriptors, spawned, true);
            voxelCount = 0;
            for (int i = 0; i < spawned.Length; i++)
            {
                if (!spawned[i].IsValid)
                    continue;
                staticInstances.Add(MakeInstance(spawned[i], i == 0 ? ground : voxelBox, colors[i]));
                if (i > 0)
                    voxelCount++;
            }

            // Joints: one for every pair. The records are added in the order of the bodies' slots (the lower slot of the pair first, then the higher), not in the
            // order of the shapes: see the JSON Mesh Import Demo. The solver computes the same thing in either order.
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
            System.Array.Sort(pairKeys, pairIds);
            jointCount = 0;
            for (int m = 0; m < jointTotal; m++)
            {
                int a = pairA[pairIds[m]];
                int b = pairB[pairIds[m]];
                Join(spawned[1 + a], positions[a], spawned[1 + b], positions[b]);
            }
            bodies.ApplyEdits();

            CreateInstanceBuffer();

            accumulator = 0f;
            stepMs = 0f;
            stepCount = 0;
            brokenJoints = -1;
            watch.Stop();
            buildMs = (float)watch.Elapsed.TotalMilliseconds;
        }

        // A ball-socket joint at the middle between the two voxels, with a rigid angle lock. Both anchors are that point, so the joint starts satisfied.
        // The objects are not rotated, so a local anchor is the offset from the voxel's center to that point.
        void Join(BodyHandle a, float3 positionA, BodyHandle b, float3 positionB)
        {
            if (!a.IsValid || !b.IsValid)
                return;
            float3 anchor = (positionA + positionB) * 0.5f;
            float yield = plasticJoints ? yieldThreshold : float.PositiveInfinity;
            float bendLimit = plasticJoints && bendBreak ? Mathf.Min(bendBreakDegrees * Mathf.Deg2Rad, Mathf.PI) : float.PositiveInfinity;
            float fractureLinear = linearFracture ? linearFractureThreshold : float.PositiveInfinity;
            float yieldLinear = linearYield ? linearYieldThreshold : float.PositiveInfinity;
            float stretchLimit = linearYield && stretchBreak ? stretchBreakMeters : float.PositiveInfinity;
            if (constraints.AddJoint(a, b, anchor - positionA, anchor - positionB, float.PositiveInfinity, float.PositiveInfinity, fractureThreshold, yield,
                    bendLimit, fractureLinear, yieldLinear, stretchLimit) >= 0)
                jointCount++;
        }

        // Each object has its own hue, and the voxels alternate a little in brightness so that you can see them.
        static Color VoxelColor(int objectIndex, int i, int j, int k)
        {
            float hue = Mathf.Repeat(0.02f + 0.17f * objectIndex, 1f);
            return Color.HSVToRGB(hue, 0.55f, ((i + j + k) & 1) == 0 ? 0.88f : 0.78f);
        }

        static InstanceData MakeInstance(BodyHandle handle, float3 size, Color color)
        {
            // The shader outputs the color as it is, so a project in the linear color space needs the linear value.
            if (QualitySettings.activeColorSpace == ColorSpace.Linear)
                color = color.linear;
            return new InstanceData { slot = (uint)handle.Slot, sizeX = size.x, sizeY = size.y, sizeZ = size.z, color = new float4(color.r, color.g, color.b, 1f) };
        }

        // The buffer holds the static instances (written now) and room for the most shots that can exist; the shots are written when they change.
        void CreateInstanceBuffer()
        {
            staticCount = staticInstances.Count;
            instances = new InstanceData[staticCount + maxShots];
            staticInstances.CopyTo(instances);
            instanceBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, instances.Length, UnsafeUtility.SizeOf<InstanceData>());
            instanceBuffer.SetData(instances, 0, 0, staticCount);
            shotsChanged = false;

            boxMaterial.SetBuffer("_Poses", bodies.State);
            boxMaterial.SetBuffer("_Instances", instanceBuffer);
        }

        void DestroySimulation()
        {
            shots.Clear();
            staticInstances.Clear();
            if (instanceBuffer != null) instanceBuffer.Release();
            instanceBuffer = null;
            instances = null;
            staticCount = 0;
            if (solver != null) solver.Dispose();
            if (constraints != null) constraints.Dispose();
            if (bodies != null) bodies.Dispose();
            solver = null;
            constraints = null;
            bodies = null;
        }

        void Shoot()
        {
            if (cam == null)
                return;
            int shotLimit = Mathf.Min(maxShots, instances.Length - staticCount); // what the instance buffer was made for
            while (shots.Count >= shotLimit)
                RemoveShot(0);

            Transform t = cam.transform;
            float3 direction = t.forward;
            float size = shotSize;

            // Start just in front of the camera, clear of the near plane, and fly the way the camera looks.
            BodyDescriptor descriptor = BodyDescriptor.Box(new float3(size), shotMass / (size * size * size), (float3)t.position + direction * (size + 0.6f));
            descriptor.rotation = t.rotation;
            descriptor.linearVelocity = direction * shotSpeed;

            BodyHandle handle = bodies.Spawn(descriptor);
            if (!handle.IsValid)
                return;

            shots.Add(new Shot { handle = handle, size = size });
            shotsChanged = true;
        }

        void RemoveShot(int index)
        {
            bodies.Remove(shots[index].handle);
            shots.RemoveAt(index);
            shotsChanged = true;
        }

        void ClearShots()
        {
            while (shots.Count > 0)
                RemoveShot(shots.Count - 1);
        }

        void StepOnce()
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            solver.Step();
            watch.Stop();
            stepCount++;
            float ms = (float)watch.Elapsed.TotalMilliseconds;
            stepMs = stepMs <= 0f ? ms : Mathf.Lerp(stepMs, ms, 0.1f);
        }

        // One instanced draw of every body. The poses stay in the solver's buffer; the only data that goes to the GPU is the list of shots, and only
        // when a shot was fired or removed.
        void DrawBodies()
        {
            if (shotsChanged)
            {
                for (int i = 0; i < shots.Count; i++)
                    instances[staticCount + i] = MakeInstance(shots[i].handle, new float3(shots[i].size), ShotColor);
                if (shots.Count > 0)
                    instanceBuffer.SetData(instances, staticCount, staticCount, shots.Count);
                shotsChanged = false;
            }

            // The light is the scene's sun if there is one (the direction towards it), else a fixed direction.
            Vector3 toLight = new Vector3(0.45f, 0.75f, -0.45f).normalized;
            Color lightColor = new Color(0.85f, 0.85f, 0.85f);
            Light sun = RenderSettings.sun != null ? RenderSettings.sun : createdLight != null ? createdLight.GetComponent<Light>() : null;
            if (sun != null && sun.isActiveAndEnabled)
            {
                toLight = -sun.transform.forward;
                lightColor = sun.color * sun.intensity;
            }
            boxMaterial.SetVector("_LightDirection", toLight);
            boxMaterial.SetColor("_LightColor", lightColor);
            boxMaterial.SetColor("_AmbientColor", new Color(0.35f, 0.37f, 0.40f));

            var renderParams = new RenderParams(boxMaterial)
            {
                // The bodies move on the GPU, so the CPU does not know where they are: a bounds that contains the scene keeps them from being culled.
                worldBounds = new Bounds(Vector3.zero, new Vector3(100000f, 100000f, 100000f)),
                shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off,
                receiveShadows = false,
                layer = gameObject.layer,
            };
            Graphics.RenderMeshPrimitives(renderParams, cubeMesh, 0, staticCount + shots.Count);
        }

        // ---- camera and light --------------------------------------------------------------------------

        void SetUpCameraAndLight()
        {
            cam = Camera.main;
            if (cam == null)
                cam = FindFirstObjectByType<Camera>();
            if (cam == null)
            {
                var go = new GameObject("AVBD Demo Camera");
                go.tag = "MainCamera";
                cam = go.AddComponent<Camera>();
                cam.nearClipPlane = 0.1f;
                cam.farClipPlane = 500f;
                cam.clearFlags = CameraClearFlags.SolidColor;
                cam.backgroundColor = new Color(0.55f, 0.70f, 0.85f);
                createdCamera = true;
            }
            else
            {
                savedCameraPosition = cam.transform.position;
                savedCameraRotation = cam.transform.rotation;
            }

            if (FindFirstObjectByType<Light>() == null)
            {
                createdLight = new GameObject("AVBD Demo Light");
                Light light = createdLight.AddComponent<Light>();
                light.type = LightType.Directional;
                light.shadows = LightShadows.Soft;
                light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
            }
        }

        void MoveCamera()
        {
            if (!controlCamera || cam == null)
                return;

            Quaternion rotation = Quaternion.Euler(pitch, yaw, 0f);
            Vector3 move = Vector3.zero;
            if (held.Contains(KeyCode.W)) move += rotation * Vector3.forward;
            if (held.Contains(KeyCode.S)) move -= rotation * Vector3.forward;
            if (held.Contains(KeyCode.D)) move += rotation * Vector3.right;
            if (held.Contains(KeyCode.A)) move -= rotation * Vector3.right;
            if (held.Contains(KeyCode.E)) move += Vector3.up;
            if (held.Contains(KeyCode.Q)) move -= Vector3.up;

            float speed = moveSpeed * (held.Contains(KeyCode.LeftShift) || held.Contains(KeyCode.RightShift) ? 3f : 1f);
            camPosition += move * (speed * Time.unscaledDeltaTime);
        }

        // ---- IMGUI -------------------------------------------------------------------------------------

        void OnGUI()
        {
            if (bodies == null)
                return;

            if (boldLabel == null)
                boldLabel = new GUIStyle(GUI.skin.label) { fontStyle = FontStyle.Bold };

            Event e = Event.current;
            if (e.type == EventType.Repaint)
                DrawCrosshair();

            DrawPanel();
            HandleInput(e);
        }

        static void DrawCrosshair()
        {
            Color old = GUI.color;
            GUI.color = new Color(1f, 1f, 1f, 0.9f);
            float cx = Screen.width * 0.5f;
            float cy = Screen.height * 0.5f;
            GUI.DrawTexture(new Rect(cx - 9f, cy - 1f, 18f, 2f), Texture2D.whiteTexture);
            GUI.DrawTexture(new Rect(cx - 1f, cy - 9f, 2f, 18f), Texture2D.whiteTexture);
            GUI.color = old;
        }

        void DrawPanel()
        {
            const float margin = 8f;
            const float width = 340f;
            float scale = Mathf.Clamp(Screen.height / 800f, 1f, 2.5f);

            Matrix4x4 oldMatrix = GUI.matrix;
            GUI.matrix = Matrix4x4.Scale(new Vector3(scale, scale, 1f));

            GUILayout.BeginArea(new Rect(margin, margin, width, Screen.height / scale - 2f * margin));
            GUILayout.BeginVertical(GUI.skin.box);

            GUILayout.Label("Shot", boldLabel);
            GUILayout.Label("Speed: " + shotSpeed.ToString("0") + " m/s");
            shotSpeed = GUILayout.HorizontalSlider(shotSpeed, 5f, 80f);

            GUILayout.Label("Size: " + shotSize.ToString("0.00") + " m");
            shotSize = GUILayout.HorizontalSlider(shotSize, 0.2f, 4f);

            // Mass on a log scale: 0.1 to 2000.
            GUILayout.Label("Mass: " + shotMass.ToString(shotMass < 10f ? "0.0" : "0") + " kg   (density " + (shotMass / (shotSize * shotSize * shotSize)).ToString("0.#") + ")");
            float logMass = GUILayout.HorizontalSlider(Mathf.Log10(shotMass), -1f, Mathf.Log10(2000f));
            shotMass = Mathf.Clamp(Mathf.Pow(10f, logMass), 0.1f, 2000f);

            if (tunnelRisk)
                GUILayout.Label("A shot moves farther than its size per step: it may pass through thin boxes.");

            GUILayout.Space(4f);
            GUILayout.Label("Solver", boldLabel);
            GUILayout.Label("Iterations: " + iterations);
            iterations = Mathf.RoundToInt(GUILayout.HorizontalSlider(iterations, 1, 30));
            fuseDualUpdate = GUILayout.Toggle(fuseDualUpdate, "Fuse the dual update into the next primal sweep");

            GUILayout.Space(4f);
            GUILayout.Label("Impact and joints (Reset to apply)", boldLabel);
            GUILayout.Label("Speed of each object: " + approachSpeed.ToString("0") + " m/s   (head-on: " + (2f * approachSpeed).ToString("0") + ")");
            approachSpeed = Mathf.Round(GUILayout.HorizontalSlider(approachSpeed, 0f, 60f));
            GUILayout.Label("Skin: " + shellLayers + " voxels (" + (shellLayers * VoxelSize).ToString("0.0") + " m)");
            shellLayers = Mathf.RoundToInt(GUILayout.HorizontalSlider(shellLayers, 1, 4));
            plasticJoints = GUILayout.Toggle(plasticJoints, "Plastic yield (force and angle) and bend break");
            linearFracture = GUILayout.Toggle(linearFracture, "Linear fracture");
            linearYield = GUILayout.Toggle(linearYield, "Linear yield and stretch break");

            GUILayout.BeginHorizontal();
            if (GUILayout.Button(paused ? "Play" : "Pause")) paused = !paused;
            if (GUILayout.Button("Step")) { paused = true; requestStep = true; }
            if (GUILayout.Button("Clear shots")) requestClearShots = true;
            if (GUILayout.Button("Reset")) requestReset = true;
            GUILayout.EndHorizontal();

            GUILayout.Space(4f);
            GUILayout.Label("voxels " + voxelCount + "   joints " + jointCount + "   shots " + shots.Count + "/" + maxShots);
            GUILayout.BeginHorizontal();
            GUILayout.Label("broken " + (brokenJoints < 0 ? "-" : brokenJoints.ToString()), GUILayout.Width(120f));
            if (GUILayout.Button("Count broken (waits for the GPU)")) requestCountBroken = true;
            GUILayout.EndHorizontal();
            GUILayout.Label("step " + stepCount + "   time " + (stepCount * solver.Settings.dt).ToString("0.00") + " s");
            GUILayout.Label("build " + buildMs.ToString("0") + " ms (CPU side, with the batch spawn and the joints)");
            GUILayout.Label("step " + stepMs.ToString("0.00") + " ms (CPU side)   frame " + frameMs.ToString("0.0") + " ms");
            if (behindRealTime)
                GUILayout.Label("running slower than real time");

            GUILayout.Space(4f);
            GUILayout.Label("Left click: shoot   Space: pause   R: reset   N: step\nRight drag: look   W A S D Q E: move (Shift: faster)\nMouse wheel: move forward and back");

            GUILayout.EndVertical();
            Rect used = GUILayoutUtility.GetLastRect();
            GUILayout.EndArea();

            if (Event.current.type == EventType.Repaint)
                panelScreenRect = new Rect(0f, 0f, (width + 2f * margin) * scale, (used.yMax + 2f * margin) * scale);

            GUI.matrix = oldMatrix;
        }

        void HandleInput(Event e)
        {
            switch (e.type)
            {
                case EventType.KeyDown:
                    held.Add(e.keyCode);
                    if (e.keyCode == KeyCode.Space) { paused = !paused; e.Use(); }
                    else if (e.keyCode == KeyCode.R) { requestReset = true; e.Use(); }
                    else if (e.keyCode == KeyCode.N) { paused = true; requestStep = true; e.Use(); }
                    return;

                case EventType.KeyUp:
                    held.Remove(e.keyCode);
                    return;
            }

            // Mouse events a panel control already used have the type Used and do not get here; the rest outside the panel are the scene's.
            if (panelScreenRect.Contains(e.mousePosition))
                return;

            if (e.type == EventType.MouseDown && e.button == 0)
            {
                requestShoot = true;
                e.Use();
            }
            else if (controlCamera && e.type == EventType.MouseDrag && e.button == 1)
            {
                yaw += e.delta.x * lookSensitivity;
                pitch = Mathf.Clamp(pitch + e.delta.y * lookSensitivity, -89f, 89f);
                e.Use();
            }
            else if (controlCamera && e.type == EventType.ScrollWheel)
            {
                camPosition -= Quaternion.Euler(pitch, yaw, 0f) * Vector3.forward * (e.delta.y * 0.5f);
                e.Use();
            }
        }

        // ---- helpers -----------------------------------------------------------------------------------

        static readonly Color ShotColor = new Color(1f, 0.25f, 0.15f);
    }

#if UNITY_EDITOR
    static class VoxelImpactDemoMenu
    {
        [UnityEditor.MenuItem("GameObject/McKrill AVBD/Voxel Impact Demo", false, 13)]
        static void Create(UnityEditor.MenuCommand command)
        {
            var go = new GameObject("AVBD Voxel Impact Demo");
            UnityEditor.GameObjectUtility.SetParentAndAlign(go, command.context as GameObject);
            go.AddComponent<VoxelImpactDemo>();
            UnityEditor.Undo.RegisterCreatedObjectUndo(go, "Create AVBD Voxel Impact Demo");
            UnityEditor.Selection.activeObject = go;
        }
    }
#endif
}
