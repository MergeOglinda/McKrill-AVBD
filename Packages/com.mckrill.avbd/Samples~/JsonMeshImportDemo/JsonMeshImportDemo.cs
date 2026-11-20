using System.Collections.Generic;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd.Samples
{
    /// <summary>
    /// Spawns the objects of a list of voxel files (JSON written by <c>Tools/Voxelize/voxelize_stl.py</c> from an STL mesh) on the GPU solver. Every voxel is a box
    /// body and every joint of the file is a ball-socket joint with a hard angle lock between two neighboring boxes, at the middle between their centers, that
    /// fractures like the joints of the GPU Fracture Wall Demo. The objects stand on the ground 5 m apart (center to center, along x); you shoot boxes at them
    /// or blow them up.
    ///
    /// The list of files is <see cref="VoxelFiles"/>, in the code. Reset reads the files again.
    ///
    /// The simulation runs entirely through the package's public API (<see cref="AvbdBodyBuffers"/>, <see cref="AvbdConstraintBuffers"/>,
    /// <see cref="AvbdSolver.Step"/>), and drawing is one instanced draw call that reads the poses from the solver's buffer, as in the Fracture Wall Demo.
    /// The scene is y-up, so solver positions and rotations are used as Unity world positions and rotations. Input is read through IMGUI events, so it works
    /// with the Input Manager and with the Input System package.
    /// </summary>
    [AddComponentMenu("McKrill AVBD/JSON Mesh Import Demo")]
    [DisallowMultipleComponent]
    public sealed class JsonMeshImportDemo : MonoBehaviour
    {
        /// <summary>
        /// The voxel files to spawn, one object each, in this order. An entry that is a full path (for example <c>C:/models/engine.voxels.json</c>) is read from disk.
        /// Any other entry is the name of a text asset in a <c>Resources</c> folder, without the extension: the three objects below ship with the sample.
        /// </summary>
        static readonly string[] VoxelFiles =
        {
            "JsonMeshImportDemo/Sphere",
            "JsonMeshImportDemo/Torus",
            "JsonMeshImportDemo/Cube",
        };

        /// <summary>Distance between the centers of two neighboring objects, along x, in meters.</summary>
        const float ObjectSpacing = 5f;

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

        // A voxel file that was read, with where it stands in the scene.
        sealed class VoxelObject
        {
            public string name;
            public VoxelMeshData data;
            public float3 offset;   // added to the file's coordinates: puts the object on the ground at its place in the row
            public float3 extent;   // size of the object's bounding box in meters
        }

        [Header("Objects (applied on Reset)")]
        [Tooltip("Mass of one voxel in kg. The density of an object follows from it and the size of its voxels, so a finer file has the same mass per voxel. A block of the Fracture Wall Demo weighs 0.25 kg.")]
        [SerializeField, Min(0.001f)] float voxelMass = 0.25f;
        [Tooltip("Joints are rigid (infinite stiffness) instead of springs with the stiffness below.")]
        [SerializeField] bool rigidJoints = false;
        [Tooltip("Stiffness of the joints' linear part when they are not rigid. The angle lock is always rigid: only a rigid angle lock stores the multiplier that fracture tests.")]
        [SerializeField, Min(1f)] float jointStiffness = 5e3f;
        [Tooltip("A joint breaks when the norm of its angular multiplier is above this.")]
        [SerializeField, Min(0.01f)] float fractureThreshold = 40f;

        [Header("Plastic joints (applied on Reset)")]
        [Tooltip("Joints yield instead of holding: above the yield threshold the angle lock gives way and the joint keeps the bend. Off: the joints are rigid up to the fracture threshold, and the package runs without any plastic code or buffer.")]
        [SerializeField] bool plasticJoints = false;
        [Tooltip("A plastic joint yields when the norm of its angular multiplier is above this. Keep it below the fracture threshold: a joint breaks before it can yield otherwise.")]
        [SerializeField, Min(0.1f)] float yieldThreshold = 8f;
        [Tooltip("A plastic joint that has yielded breaks once it is bent further than the angle below from where it started.")]
        [SerializeField] bool bendBreak = true;
        [Tooltip("The bend, in degrees, above which a plastic joint breaks (with bend break on).")]
        [SerializeField, Range(1f, 180f)] float bendBreakDegrees = 30f;

        [Header("Linear fracture and plasticity (applied on Reset; need rigid joints)")]
        [Tooltip("A joint breaks when the norm of its linear multiplier (the force it carries) is above the threshold below. Needs rigid joints.")]
        [SerializeField] bool linearFracture = false;
        [SerializeField, Min(0.01f)] float linearFractureThreshold = 400f;
        [Tooltip("Joints yield in the linear row too: above the yield threshold the joint gives way and keeps the stretch. Needs rigid joints.")]
        [SerializeField] bool linearYield = false;
        [SerializeField, Min(0.1f)] float linearYieldThreshold = 100f;
        [Tooltip("A joint that has yielded linearly breaks once its anchors have moved apart further than the distance below.")]
        [SerializeField] bool stretchBreak = true;
        [SerializeField, Range(0.01f, 2f)] float stretchBreakMeters = 0.25f;

        [Header("Solver")]
        [SerializeField, Range(1, 30)] int iterations = 10;
        [Tooltip("Upper bound on solver steps per rendered frame. If the GPU cannot keep up, the simulation runs slower than real time instead of stalling.")]
        [SerializeField, Range(1, 4)] int maxStepsPerFrame = 2;

        [Header("Shots")]
        [Tooltip("Speed of a shot in meters per second.")]
        [SerializeField, Range(5f, 80f)] float shotSpeed = 30f;
        [Tooltip("Edge length of the shot, a cube.")]
        [SerializeField, Range(0.2f, 2f)] float shotSize = 1f;
        [Tooltip("Mass of the shot. A voxel weighs 0.25 kg by default.")]
        [SerializeField, Range(0.1f, 200f)] float shotMass = 50f;
        [Tooltip("Shots alive at once. Shooting more removes the oldest.")]
        [SerializeField, Min(1)] int maxShots = 100;

        [Header("Explosion")]
        [Tooltip("Reach of an explosion in meters: a voxel whose center is farther from the explosion is not touched, and the push falls off to zero at this distance.")]
        [SerializeField, Range(1f, 50f)] float explosionRadius = 5f;
        [Tooltip("Impulse in N s that a body at the center of the explosion would receive. A body's velocity changes by this times a falloff (1 at the center, 0 at the radius, squared) divided by its mass.")]
        [SerializeField, Range(0.1f, 1000f)] float explosionForce = 10f;
        [Tooltip("Meters behind the surface that the camera ray hits, along the ray, where the explosion goes off: 0 on the surface, more to go inside.")]
        [SerializeField, Range(0f, 20f)] float explosionDepth = 0.5f;

        [Header("View")]
        [Tooltip("Fly camera: right drag to look, W A S D to move, Q and E for down and up, Shift for faster, mouse wheel to dolly.")]
        [SerializeField] bool controlCamera = true;
        [SerializeField, Min(0.1f)] float moveSpeed = 8f;
        [SerializeField, Min(0.01f)] float lookSensitivity = 0.2f;

        const float GroundHalfExtent = 100f;
        const float VoxelGap = 0.01f;       // meters between the faces of neighboring boxes (less for tiny voxels): the objects start almost at rest, and
                                            // voxels that touch only along an edge or at a corner do not start inside each other
        const float MaxGapFraction = 0.05f; // of the voxel size
        const bool zCurveOrder = true;

        // What a left click does.
        enum ClickMode
        {
            Shoot,
            Explode,
        }

        AvbdBodyBuffers bodies;
        AvbdConstraintBuffers constraints;
        AvbdSolver solver;
        // Drawing: one instance per body. The first staticCount instances (the ground and the voxels) are written once per Reset, the shots follow.
        InstanceData[] instances;
        GraphicsBuffer instanceBuffer;
        int staticCount;
        bool shotsChanged;
        int jointCount;
        int voxelTotal;

        readonly List<InstanceData> staticInstances = new List<InstanceData>(); // ground and voxels
        readonly List<Shot> shots = new List<Shot>();                            // oldest first
        readonly List<VoxelObject> objects = new List<VoxelObject>();            // the files that were read
        readonly List<string> objectLines = new List<string>();                  // one line per entry of VoxelFiles for the panel: what was read, or why not
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
        bool requestExplode;
        bool requestToggleMode;
        bool requestReset;
        bool requestClearShots;
        bool requestStep;

        ClickMode clickMode = ClickMode.Shoot;
        string explosionStatus = "";
        AvbdRaycaster raycaster;       // the ray of an explosion, cast on the GPU
        int simulationVersion;         // counts the simulations built, so that the answer of a ray cast of an old one is dropped
        const float RayMaxDistance = 10000f;

        Rect panelScreenRect;
        GUIStyle boldLabel;

        // ---- lifecycle ---------------------------------------------------------------------------------

        void OnEnable()
        {
            Shader boxShader = AvbdShaderSet.Load()?.InstancedBoxes;
            if (boxShader == null || !boxShader.isSupported)
            {
                Debug.LogError("JSON Mesh Import Demo: the package's instanced box shader is missing or not supported on this device (it needs compute buffers in the vertex shader, shader model 4.5).");
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
            PlaceCameraInFrontOfObjects();
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
                requestExplode = false;
                requestClearShots = false;
            }
            if (requestToggleMode)
            {
                clickMode = clickMode == ClickMode.Shoot ? ClickMode.Explode : ClickMode.Shoot;
                requestToggleMode = false;
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
            if (requestExplode)
            {
                Explode();
                requestExplode = false;
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

        // ---- loading -----------------------------------------------------------------------------------

        // Reads every file of VoxelFiles. A file that cannot be read is reported (log and panel) and left out; the rest is still spawned.
        void LoadObjects()
        {
            objects.Clear();
            objectLines.Clear();
            voxelTotal = 0;

            foreach (string entry in VoxelFiles)
            {
                string name = DisplayName(entry);
                if (!VoxelMeshData.TryLoad(entry, out VoxelMeshData data, out string error))
                {
                    Debug.LogError("JSON Mesh Import Demo: \"" + entry + "\" was not loaded: " + error);
                    objectLines.Add(name + ": not loaded, " + error);
                    continue;
                }

                // The bounding box of the voxels that are there (the grid of the file can be larger), in the file's frame.
                int3 min = new int3(int.MaxValue), max = new int3(int.MinValue);
                for (int n = 0; n < data.voxelCount; n++)
                {
                    var index = new int3(data.voxels[3 * n], data.voxels[3 * n + 1], data.voxels[3 * n + 2]);
                    min = math.min(min, index);
                    max = math.max(max, index);
                }
                float3 origin = new float3(data.origin[0], data.origin[1], data.origin[2]);
                float3 boxMin = origin + (float3)min * data.voxelSize;
                float3 boxMax = origin + (float3)(max + 1) * data.voxelSize;

                // Object i of n stands at x = (i - (n - 1) / 2) * spacing, centered on x = 0, with the middle of its bounding box on that spot in x and z and its
                // lowest face on the ground. The count n is not known yet, so the offsets get their x in the pass below.
                var item = new VoxelObject
                {
                    name = name,
                    data = data,
                    offset = new float3(-(boxMin.x + boxMax.x) * 0.5f, -boxMin.y, -(boxMin.z + boxMax.z) * 0.5f),
                    extent = boxMax - boxMin,
                };
                objects.Add(item);
                voxelTotal += data.voxelCount;
                objectLines.Add(name + ": " + data.voxelCount + " voxels, " + data.jointCount + " joints, "
                    + item.extent.x.ToString("0.0") + " x " + item.extent.y.ToString("0.0") + " x " + item.extent.z.ToString("0.0") + " m");
                if (item.extent.x > ObjectSpacing || item.extent.z > ObjectSpacing)
                    Debug.LogWarning("JSON Mesh Import Demo: \"" + name + "\" is wider than the " + ObjectSpacing + " m between the objects and will overlap its neighbor.");
            }

            for (int i = 0; i < objects.Count; i++)
                objects[i].offset.x += (i - (objects.Count - 1) * 0.5f) * ObjectSpacing;
        }

        // The name shown in the panel: the file name without the folder and without ".json" or ".voxels.json".
        static string DisplayName(string entry)
        {
            string name = entry.Replace('\\', '/');
            name = name.Substring(name.LastIndexOf('/') + 1);
            if (name.EndsWith(".json", System.StringComparison.OrdinalIgnoreCase))
                name = name.Substring(0, name.Length - ".json".Length);
            if (name.EndsWith(".voxels", System.StringComparison.OrdinalIgnoreCase))
                name = name.Substring(0, name.Length - ".voxels".Length);
            return name;
        }

        // ---- simulation --------------------------------------------------------------------------------

        void BuildSimulation()
        {
            DestroySimulation();
            LoadObjects();

            int jointsInFiles = 0;
            foreach (VoxelObject o in objects)
                jointsInFiles += o.data.jointCount;

            // Bodies: the ground, the voxels and the shots.
            maxShots = Mathf.Max(maxShots, 1);
            int capacity = 1 + voxelTotal + maxShots + 8;
            bodies = new AvbdBodyBuffers(capacity);
            if (bodies.State.stride != UnsafeUtility.SizeOf<BodyStateRecord>())
                throw new System.InvalidOperationException("The package's body state record changed; update BodyStateRecord in JsonMeshImportDemo.");
            staticInstances.Clear();

            AvbdSolverSettings settings = AvbdSolverSettings.Default; // 60 Hz, gravity (0, -10, 0): Unity's y-up
            settings.iterations = iterations;
            // Manifold slots: a voxel touches up to 26 others, but at rest only the joined ones (which make no manifold) and a few more; a shot can lie on or
            // inside a dozen. A pair that finds no slot just gets no contacts until one is free.
            int manifoldSlots = Mathf.Min(voxelTotal * 8 + maxShots * 16 + 256, AvbdConstraintBuffers.MaxContactManifolds);
            // The plastic array only exists if the joints can be plastic: without it the solver runs the joint kernels with no plastic code at all.
            int jointCapacity = jointsInFiles + 8;
            constraints = new AvbdConstraintBuffers(bodies, jointCapacity, null, manifoldSlots, (plasticJoints || (linearYield && rigidJoints)) ? jointCapacity : 0);
            solver = new AvbdSolver(bodies, settings, null, constraints);
            raycaster = new AvbdRaycaster(bodies);
            simulationVersion++;

            // The ground and every voxel go in one batch (the ground is element 0, then the voxels object by object, in the order of the file), so that the bodies
            // can be placed in Z-curve order. The handles come back in the order of the list; the joints and the instances use them.
            var descriptors = new List<BodyDescriptor>(1 + voxelTotal);
            var instanceSizes = new List<float3>(1 + voxelTotal);
            var instanceColors = new List<Color>(1 + voxelTotal);
            var positions = new float3[voxelTotal]; // world position of every voxel, in the same order

            // Ground: a static slab whose top face is the plane y = 0.
            var ground = new float3(GroundHalfExtent * 2f, 1f, GroundHalfExtent * 2f);
            descriptors.Add(BodyDescriptor.Box(ground, 0f, new float3(0f, -0.5f, 0f)));
            instanceSizes.Add(ground);
            instanceColors.Add(new Color(0.30f, 0.33f, 0.36f));

            int first = 0; // index in positions of the first voxel of the object
            for (int i = 0; i < objects.Count; i++)
            {
                VoxelObject o = objects[i];
                VoxelMeshData data = o.data;
                // The box is a little smaller than the cell: neighbors that are not joined (those that touch along an edge or at a corner) start apart.
                float edge = data.voxelSize - Mathf.Min(VoxelGap, MaxGapFraction * data.voxelSize);
                var size = new float3(edge);
                float density = voxelMass / (edge * edge * edge);
                for (int n = 0; n < data.voxelCount; n++)
                {
                    float3 position = data.VoxelCenter(n) + o.offset;
                    positions[first + n] = position;
                    descriptors.Add(BodyDescriptor.Box(size, density, position));
                    instanceSizes.Add(size);
                    instanceColors.Add(VoxelColor(i, data.voxels[3 * n], data.voxels[3 * n + 1], data.voxels[3 * n + 2]));
                }
                first += data.voxelCount;
            }
            var spawned = new BodyHandle[descriptors.Count];
            bodies.SpawnBatch(descriptors, spawned, zCurveOrder);
            for (int i = 0; i < spawned.Length; i++)
                AddStaticInstance(spawned[i], instanceSizes[i], instanceColors[i]);

            // Joints: one for every pair of the file.
            jointCount = 0;
            first = 0;
            foreach (VoxelObject o in objects)
            {
                int[] pairs = o.data.joints;
                for (int m = 0; m < o.data.jointCount; m++)
                {
                    int a = first + pairs[2 * m];
                    int b = first + pairs[2 * m + 1];
                    Join(spawned[1 + a], positions[a], spawned[1 + b], positions[b]);
                }
                first += o.data.voxelCount;
            }
            bodies.ApplyEdits();

            CreateInstanceBuffer();

            accumulator = 0f;
            stepMs = 0f;
        }

        // A ball-socket joint at the middle between the two voxels (the middle of the face they share, for neighbors by a face), with a rigid angle lock.
        // Both anchors are that point, so the joint starts satisfied.
        void Join(BodyHandle a, float3 positionA, BodyHandle b, float3 positionB)
        {
            if (!a.IsValid || !b.IsValid)
                return;
            float3 anchor = (positionA + positionB) * 0.5f;
            float linear = rigidJoints ? float.PositiveInfinity : jointStiffness;
            float yield = plasticJoints ? yieldThreshold : float.PositiveInfinity;
            float bendLimit = plasticJoints && bendBreak ? Mathf.Min(bendBreakDegrees * Mathf.Deg2Rad, Mathf.PI) : float.PositiveInfinity;
            // The linear options need a hard linear row (AddJoint throws otherwise), so they apply to rigid joints only.
            float fractureLinear = rigidJoints && linearFracture ? linearFractureThreshold : float.PositiveInfinity;
            float yieldLinear = rigidJoints && linearYield ? linearYieldThreshold : float.PositiveInfinity;
            float stretchLimit = rigidJoints && linearYield && stretchBreak ? stretchBreakMeters : float.PositiveInfinity;
            if (constraints.AddJoint(a, b, anchor - positionA, anchor - positionB, linear, float.PositiveInfinity, fractureThreshold, yield, bendLimit,
                    fractureLinear, yieldLinear, stretchLimit) >= 0)
                jointCount++;
        }

        // Each object has its own hue, and the voxels alternate a little in brightness so that you can see them.
        static Color VoxelColor(int objectIndex, int i, int j, int k)
        {
            float hue = Mathf.Repeat(0.02f + 0.17f * objectIndex, 1f);
            return Color.HSVToRGB(hue, 0.55f, ((i + j + k) & 1) == 0 ? 0.88f : 0.78f);
        }

        void AddStaticInstance(BodyHandle handle, float3 size, Color color)
        {
            if (!handle.IsValid)
                return;
            staticInstances.Add(MakeInstance(handle, size, color));
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
            if (raycaster != null) raycaster.Dispose();
            raycaster = null;
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

        // An explosion where the ray from the center of the camera meets a body, pushed explosionDepth meters further along the ray. The ray is cast on the GPU
        // (AvbdRaycaster) against the poses where they are, so the CPU never reads them: only the distance of the nearest hit comes back, a few frames later, and
        // then the explosion is queued (AvbdSolver.Explode). The package applies it in the predict stage of the next step, so while the simulation is paused
        // it waits for the next Step. The radius, force and depth of the click are the ones in the panel when it is clicked.
        void Explode()
        {
            if (cam == null)
                return;

            Transform t = cam.transform;
            float3 origin = t.position;
            float3 direction = t.forward;
            float radius = explosionRadius;
            float force = explosionForce;
            float depth = explosionDepth;
            int version = simulationVersion; // a Reset before the answer arrives makes it stale

            explosionStatus = "Casting the ray...";
            raycaster.Cast(origin, direction, RayMaxDistance, (hit, distance) =>
            {
                if (version != simulationVersion || solver == null)
                    return;
                if (!hit)
                {
                    explosionStatus = "No explosion: the ray from the camera hit nothing.";
                    return;
                }

                float3 center = origin + math.normalizesafe(direction) * (distance + depth);
                explosionStatus = solver.Explode(center, radius, force)
                    ? "Explosion at (" + center.x.ToString("0.0") + ", " + center.y.ToString("0.0") + ", " + center.z.ToString("0.0") + "), "
                        + distance.ToString("0.0") + " m to the surface" + (paused ? " (waits for the next step)" : "")
                    : "Not queued: too many explosions for one step.";
            });
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

        // In front of the row of objects, far enough back to see all of it, at about half the height of the tallest. Reset leaves the camera where it is.
        void PlaceCameraInFrontOfObjects()
        {
            float width = ObjectSpacing, height = 2f, depth = ObjectSpacing;
            foreach (VoxelObject o in objects)
            {
                width = Mathf.Max(width, o.extent.x);
                height = Mathf.Max(height, o.extent.y);
                depth = Mathf.Max(depth, o.extent.z);
            }
            width += Mathf.Max(objects.Count - 1, 0) * ObjectSpacing;
            camPosition = new Vector3(0f, 1.5f + 0.4f * height, -(0.6f * width + 0.5f * depth + 3f));
            yaw = 0f;
            pitch = 5f;
        }

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

            // What a left click does. The button only asks: the mode changes in Update, so the controls below never change in the middle of an event.
            if (GUILayout.Button(clickMode == ClickMode.Shoot ? "Left click: shoot boxes   (switch to explosions)" : "Left click: explosions   (switch to shooting boxes)"))
                requestToggleMode = true;

            if (clickMode == ClickMode.Shoot)
            {
                GUILayout.Label("Speed: " + shotSpeed.ToString("0") + " m/s");
                shotSpeed = GUILayout.HorizontalSlider(shotSpeed, 5f, 80f);

                GUILayout.Label("Size: " + shotSize.ToString("0.00") + " m");
                shotSize = GUILayout.HorizontalSlider(shotSize, 0.2f, 2f);

                // Mass on a log scale: 0.1 to 200.
                GUILayout.Label("Mass: " + shotMass.ToString(shotMass < 10f ? "0.0" : "0") + " kg   (density " + (shotMass / (shotSize * shotSize * shotSize)).ToString("0.#") + ")");
                float logMass = GUILayout.HorizontalSlider(Mathf.Log10(shotMass), -1f, Mathf.Log10(200f));
                shotMass = Mathf.Clamp(Mathf.Pow(10f, logMass), 0.1f, 200f);

                if (tunnelRisk)
                    GUILayout.Label("A shot moves farther than its size per step: it may pass through thin boxes.");
            }
            else
            {
                GUILayout.Label("Radius: " + explosionRadius.ToString("0.0") + " m");
                explosionRadius = GUILayout.HorizontalSlider(explosionRadius, 1f, 50f);

                // Force on a log scale: 0.1 to 1000 N s. A voxel at the center gets force / mass.
                GUILayout.Label("Force: " + explosionForce.ToString(explosionForce < 10f ? "0.0" : "0") + " N s   (" + (explosionForce / voxelMass).ToString("0") + " m/s for a voxel at the center)");
                explosionForce = Mathf.Clamp(Mathf.Pow(10f, GUILayout.HorizontalSlider(Mathf.Log10(explosionForce), -1f, 3f)), 0.1f, 1000f);

                GUILayout.Label("Depth: " + explosionDepth.ToString("0.0") + " m   (behind the surface the camera ray hits)");
                explosionDepth = GUILayout.HorizontalSlider(explosionDepth, 0f, 20f);

                GUILayout.Label(explosionStatus);
            }

            GUILayout.Space(4f);
            GUILayout.Label("Iterations: " + iterations);
            iterations = Mathf.RoundToInt(GUILayout.HorizontalSlider(iterations, 1, 30));

            GUILayout.Space(4f);
            GUILayout.Label("Objects (Reset reads the files again)", boldLabel);
            foreach (string line in objectLines)
                GUILayout.Label(line);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button(paused ? "Play" : "Pause")) paused = !paused;
            if (GUILayout.Button("Step")) { paused = true; requestStep = true; }
            if (GUILayout.Button("Clear shots")) requestClearShots = true;
            if (GUILayout.Button("Reset")) requestReset = true;
            GUILayout.EndHorizontal();

            // Mass, stiffness and the threshold on log scales.
            GUILayout.Label("Voxel mass: " + voxelMass.ToString(voxelMass < 1f ? "0.000" : "0.0") + " kg");
            voxelMass = Mathf.Clamp(Mathf.Pow(10f, GUILayout.HorizontalSlider(Mathf.Log10(voxelMass), -2f, 1f)), 0.01f, 10f);

            rigidJoints = GUILayout.Toggle(rigidJoints, "Rigid joints");
            GUILayout.Label("Joint stiffness: " + (rigidJoints ? "rigid (the slider is not used)" : jointStiffness.ToString("0.0e+0")));
            jointStiffness = Mathf.Clamp(Mathf.Pow(10f, GUILayout.HorizontalSlider(Mathf.Log10(jointStiffness), 2f, 8f)), 1e2f, 1e8f);

            GUILayout.Label("Fracture threshold: " + fractureThreshold.ToString(fractureThreshold < 10f ? "0.0" : "0"));
            fractureThreshold = Mathf.Clamp(Mathf.Pow(10f, GUILayout.HorizontalSlider(Mathf.Log10(fractureThreshold), 0f, 4f)), 1f, 1e4f);

            // Plastic joints. The controls are always drawn (greyed out when plastic joints are off): a control that comes and goes while the panel
            // is being laid out is an IMGUI error, and a click on the toggle changes the state in the middle of an event.
            GUILayout.Space(4f);
            plasticJoints = GUILayout.Toggle(plasticJoints, "Plastic yield: joints give way and keep the bend");
            bool wasEnabled = GUI.enabled;
            GUI.enabled = wasEnabled && plasticJoints;
            GUILayout.Label("Yield threshold: " + yieldThreshold.ToString(yieldThreshold < 10f ? "0.0" : "0")
                + (yieldThreshold >= fractureThreshold ? "   (not below the fracture threshold: joints break first)" : ""));
            yieldThreshold = Mathf.Clamp(Mathf.Pow(10f, GUILayout.HorizontalSlider(Mathf.Log10(yieldThreshold), -1f, 4f)), 0.1f, 1e4f);
            bendBreak = GUILayout.Toggle(bendBreak, "Bend break: a joint breaks when bent too far");
            GUI.enabled = wasEnabled && plasticJoints && bendBreak;
            GUILayout.Label("Bend limit: " + bendBreakDegrees.ToString("0") + " degrees");
            bendBreakDegrees = Mathf.Round(GUILayout.HorizontalSlider(bendBreakDegrees, 1f, 180f));
            GUI.enabled = wasEnabled;

            // Linear fracture and plasticity: only for rigid joints (greyed out otherwise).
            GUILayout.Space(4f);
            bool linearEnabled = wasEnabled && rigidJoints;
            GUI.enabled = linearEnabled;
            linearFracture = GUILayout.Toggle(linearFracture, "Linear fracture: a joint breaks under force" + (rigidJoints ? "" : "   (needs rigid joints)"));
            GUI.enabled = linearEnabled && linearFracture;
            GUILayout.Label("Linear fracture threshold: " + linearFractureThreshold.ToString("0"));
            linearFractureThreshold = Mathf.Clamp(Mathf.Pow(10f, GUILayout.HorizontalSlider(Mathf.Log10(linearFractureThreshold), 0f, 5f)), 1f, 1e5f);
            GUI.enabled = linearEnabled;
            linearYield = GUILayout.Toggle(linearYield, "Linear yield: joints give way and keep the stretch");
            GUI.enabled = linearEnabled && linearYield;
            GUILayout.Label("Linear yield threshold: " + linearYieldThreshold.ToString("0")
                + (linearFracture && linearYieldThreshold >= linearFractureThreshold ? "   (not below the fracture threshold: joints break first)" : ""));
            linearYieldThreshold = Mathf.Clamp(Mathf.Pow(10f, GUILayout.HorizontalSlider(Mathf.Log10(linearYieldThreshold), -1f, 5f)), 0.1f, 1e5f);
            stretchBreak = GUILayout.Toggle(stretchBreak, "Stretch break: a joint breaks when stretched too far");
            GUI.enabled = linearEnabled && linearYield && stretchBreak;
            GUILayout.Label("Stretch limit: " + stretchBreakMeters.ToString("0.00") + " m");
            stretchBreakMeters = Mathf.Round(GUILayout.HorizontalSlider(stretchBreakMeters, 0.01f, 2f) * 100f) / 100f;
            GUI.enabled = wasEnabled;
            GUILayout.Space(4f);
            GUILayout.Label("bodies " + (staticCount + shots.Count) + "   shots " + shots.Count + "/" + maxShots);
            GUILayout.Label("joints broken " + (constraints != null ? constraints.BrokenCount : 0) + " / " + jointCount);
            GUILayout.Label("step " + stepMs.ToString("0.00") + " ms (CPU side)   frame " + frameMs.ToString("0.0") + " ms");
            if (behindRealTime)
                GUILayout.Label("running slower than real time");

            GUILayout.Space(4f);
            GUILayout.Label("Left click: shoot or explode (the mode button)\nSpace: pause   R: reset\nRight drag: look   W A S D Q E: move (Shift: faster)\nMouse wheel: move forward and back");

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
                if (clickMode == ClickMode.Explode)
                    requestExplode = true;
                else
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
    static class JsonMeshImportDemoMenu
    {
        [UnityEditor.MenuItem("GameObject/McKrill AVBD/JSON Mesh Import Demo", false, 13)]
        static void Create(UnityEditor.MenuCommand command)
        {
            var go = new GameObject("AVBD JSON Mesh Import Demo");
            UnityEditor.GameObjectUtility.SetParentAndAlign(go, command.context as GameObject);
            go.AddComponent<JsonMeshImportDemo>();
            UnityEditor.Undo.RegisterCreatedObjectUndo(go, "Create AVBD JSON Mesh Import Demo");
            UnityEditor.Selection.activeObject = go;
        }
    }
#endif
}
