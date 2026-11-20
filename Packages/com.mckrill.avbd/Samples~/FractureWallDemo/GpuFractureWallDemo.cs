using System.Collections.Generic;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd.Samples
{
    /// <summary>
    /// A brick wall (75 x 75 blocks by default) with an immobile bottom row, on the GPU solver, held together by joints that fracture, and a camera you can shoot boxes from.
    /// The panel can set up several walls, one behind the other, 4 m apart (ten by default).
    ///
    /// Every block is joined to its neighbors (the one beside it and the two under it in a running bond) by a ball-socket joint at the
    /// middle of the shared face, with a hard angle lock. A joint breaks when its angular multiplier exceeds the fracture threshold
    /// (stage D). The panel sets the properties of the joints and the blocks; they take effect on Reset, because the package cannot change a
    /// joint after it was added. A fractured joint leaves the solver, and its two blocks collide again like any other pair.
    ///
    /// The joints can also be plastic (off by default). A plastic joint yields when its angular multiplier is above the yield threshold: the angle
    /// lock gives way and the joint keeps the bend it ends up with, instead of springing back. A bend limit breaks a joint that has bent too far.
    /// The yield threshold should be below the fracture threshold, which then only catches the loads that jump past it within one step.
    ///
    /// The simulation runs entirely through the package's public API: <see cref="AvbdBodyBuffers"/> holds the bodies,
    /// <see cref="AvbdSolver.Step"/> advances them. Collisions are part of that step: the broadphase finds the touching pairs, the narrow
    /// phase keeps a manifold for each, and the solver solves the contacts, so this demo contains no collision code.
    ///
    /// Drawing never leaves the GPU: one instanced draw call (<c>Graphics.RenderMeshPrimitives</c>) draws every box, and the vertex shader
    /// (<c>AvbdInstancedBoxes</c>) reads each box's pose from the solver's own body state buffer. There is no readback and no object per body,
    /// so the cost of drawing does not grow with the number of bodies the way a GameObject per body does. The shader has no shadows.
    ///
    /// The scene is y-up (the package default), so solver positions and rotations are used as Unity world positions and rotations.
    /// Input is read through IMGUI events, so it works with the Input Manager and with the Input System package.
    /// </summary>
    [AddComponentMenu("McKrill AVBD/GPU Fracture Wall Demo")]
    [DisallowMultipleComponent]
    public sealed class GpuFractureWallDemo : MonoBehaviour
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

        [Header("Wall (applied on Reset)")]
        [Tooltip("Blocks per row. 75 columns and 75 rows are 5625 blocks per wall.")]
        [SerializeField, Range(MinColumns, MaxColumns)] int columns = 75;
        [Tooltip("Rows of blocks. The bottom row is immobile.")]
        [SerializeField, Range(MinRows, MaxRows)] int rows = 75;
        [Tooltip("Walls one behind the other, 4 meters apart along the camera's viewing direction (z). Each has the size above.")]
        [SerializeField, Range(MinWalls, MaxWalls)] int walls = 10;
        [SerializeField] Vector3 blockSize = new Vector3(1f, 0.5f, 0.5f);
        [SerializeField, Min(0.01f)] float blockDensity = 1f;
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

        [Header("Solver")]
        [SerializeField, Range(1, 30)] int iterations = 10;
        [Tooltip("Upper bound on solver steps per rendered frame. If the GPU cannot keep up, the simulation runs slower than real time instead of stalling.")]
        [SerializeField, Range(1, 4)] int maxStepsPerFrame = 2;

        [Header("Shots")]
        [Tooltip("Speed of a shot in meters per second.")]
        [SerializeField, Range(5f, 80f)] float shotSpeed = 30f;
        [Tooltip("Edge length of the shot, a cube.")]
        [SerializeField, Range(0.2f, 2f)] float shotSize = 2f;
        [Tooltip("Mass of the shot. A wall block weighs its volume times the block density (0.25 by default).")]
        [SerializeField, Range(0.1f, 200f)] float shotMass = 200f;
        [Tooltip("Shots alive at once. Shooting more removes the oldest.")]
        [SerializeField, Min(1)] int maxShots = 100;

        [Header("Explosion")]
        [Tooltip("Reach of an explosion in meters: a block whose center is farther from the explosion is not touched, and the push falls off to zero at this distance.")]
        [SerializeField, Range(1f, 50f)] float explosionRadius = 10f;
        [Tooltip("Impulse in N s that a body at the center of the explosion would receive. A body's velocity changes by this times a falloff (1 at the center, 0 at the radius, squared) divided by its mass, so the light blocks fly (0.25 kg by default) and a heavy shot hardly moves.")]
        [SerializeField, Range(0.1f, 1000f)] float explosionForce = 10f;
        [Tooltip("Meters behind the surface that the camera ray hits, along the ray, where the explosion goes off: 0 on the surface, more to go inside.")]
        [SerializeField, Range(0f, 20f)] float explosionDepth = 1f;

        [Header("Grab")]
        [Tooltip("Meters around the point the camera ray hits: every body whose box is this close to the point is grabbed (the nearest block alone for a small radius, a group of blocks for a large one).")]
        [SerializeField, Range(0.1f, 10f)] float grabRadius = 1.5f;
        [Tooltip("Stiffness in N/m of the spring that pulls the grabbed blocks (as one group, by their centroid) to the point at the center of the screen: each block gets this times how far the centroid is from that point, in newtons. It grows with the distance of the blocks from the center of the screen (see Offset gain).")]
        [SerializeField, Range(10f, 2000f)] float grabForce = 100f;
        [Tooltip("Damping ratio of the velocity of each grabbed block. 0 is a spring that swings, 1 is critical damping.")]
        [SerializeField, Range(0f, 3f)] float grabDamping = 1f;
        [Tooltip("How much the force grows with the distance of the grabbed blocks from the center of the screen, per meter: the stiffness is Force times (1 + gain times the distance in meters of the blocks from the line of sight through the held point). 0 is a constant stiffness.")]
        [SerializeField, Range(0f, 5f)] float grabOffsetGain = 0.5f;

        [Header("View")]
        [Tooltip("Fly camera: right drag to look, W A S D to move, Q and E for down and up, Shift for faster, mouse wheel to dolly.")]
        [SerializeField] bool controlCamera = true;
        [SerializeField, Min(0.1f)] float moveSpeed = 8f;
        [SerializeField, Min(0.01f)] float lookSensitivity = 0.2f;

        // The wall sizes: 15 x 15 blocks at least, 75 x 75 at most, in 1 to 10 walls. The limit is what was tried: the constraint lists are built on
        // the GPU (stage G), but the graph coloring still reads two small things back every step.
        const int MinColumns = 15;
        const int MaxColumns = 75;
        const int MinRows = 15;
        const int MaxRows = 75;
        const int MinWalls = 1;
        const int MaxWalls = 10;
        const float WallSpacing = 4f; // meters between the walls (from the middle of one to the middle of the next, along z)
        const float GroundHalfExtent = 100f;
        const float BlockGap = 0.01f; // between neighbors; the wall starts almost at rest
        const bool zCurveOrder = true;

        // What a left click does.
        enum ClickMode
        {
            Explode,
            Shoot,
            Grab,
        }

        // Every body the demo spawns is in DemoGroup. The group mask of a new body has all bits set, which would put it in every group, so the bodies get
        // this one bit and the grab (a spring force field, see Grab below) labels the bodies it holds with GrabGroup and nothing else.
        const uint DemoGroup = 1u;
        const uint GrabGroup = 2u;

        AvbdBodyBuffers bodies;
        AvbdConstraintBuffers constraints;
        AvbdSolver solver;
        // Drawing: one instance per body. The first staticCount instances (the ground and the walls) are written once per Reset, the shots follow.
        InstanceData[] instances;
        GraphicsBuffer instanceBuffer;
        int staticCount;
        bool shotsChanged;
        int jointCount;

        readonly List<InstanceData> staticInstances = new List<InstanceData>(); // ground and walls
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
        bool requestExplode;
        int requestMode = -1; // a ClickMode to switch to, or -1
        bool requestGrabStart;
        bool requestVolley;
        bool requestReset;
        bool requestClearShots;
        bool requestStep;

        ClickMode clickMode = ClickMode.Shoot;
        string explosionStatus = "";
        AvbdRaycaster raycaster;       // the ray of an explosion, cast on the GPU
        int simulationVersion;         // counts the simulations built, so that the answer of a ray cast of an old one is dropped
        const float RayMaxDistance = 10000f;

        // Grab: hold the left mouse button in Grab mode. The ray of the click is cast on the GPU like the one of an explosion; when the hit is back, the
        // bodies within the radius of the hit get the group mask bit GrabGroup (AvbdGroupLabeler, on the GPU) and a spring force field for GrabGroup pulls their
        // centroid to a point in front of the camera, at the distance the ray hit. The blocks move together and keep their arrangement; the joints between
        // them and the rest of the wall stretch and break. Letting go removes the field and the labels.
        AvbdGroupLabeler labeler;
        bool grabHeld;                 // the button is down (set in OnGUI)
        bool grabActive;               // the bodies are labelled and the field exists
        int grabToken;                 // changes at every start and end, so that the answer of a ray cast of a grab that is over is dropped
        float grabDistance;            // meters from the camera to the point that is held
        ForceFieldHandle grabField;
        string grabStatus = "";

        Rect panelScreenRect;
        GUIStyle boldLabel;

        // ---- lifecycle ---------------------------------------------------------------------------------

        void OnEnable()
        {
            Shader boxShader = AvbdShaderSet.Load()?.InstancedBoxes;
            if (boxShader == null || !boxShader.isSupported)
            {
                Debug.LogError("GPU Fracture Wall Demo: the package's instanced box shader is missing or not supported on this device (it needs compute buffers in the vertex shader, shader model 4.5).");
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

            // Camera: in front of the wall, at about half its height, looking at it. Reset leaves the camera where it is.
            camPosition = new Vector3(0f, 1.5f + 0.4f * rows * (blockSize.y + BlockGap), -(columns * (blockSize.x + BlockGap) * 0.9f + 4f));
            yaw = 0f;
            pitch = 5f;
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
            {
                held.Clear(); // a key released while the window had no focus never sends its KeyUp
                grabHeld = false; // nor does the button
            }
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
                requestGrabStart = false;
                requestVolley = false;
                requestClearShots = false;
            }
            if (requestMode >= 0)
            {
                if ((ClickMode)requestMode != ClickMode.Grab)
                    EndGrab();
                clickMode = (ClickMode)requestMode;
                requestMode = -1;
            }
            if (requestGrabStart)
            {
                if (grabHeld && clickMode == ClickMode.Grab)
                    StartGrab();
                requestGrabStart = false;
            }
            if (!grabHeld)
                EndGrab();
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
            if (requestVolley)
            {
                FireVolley();
                requestVolley = false;
            }

            MoveCamera();
            UpdateGrab();
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

        // ---- simulation --------------------------------------------------------------------------------

        /// <summary>Number of blocks of a wall.</summary>
        public static int BlockCount(int columns, int rows) => columns * rows;

        /// <summary>Number of blocks of <paramref name="walls"/> walls.</summary>
        public static int BlockCount(int columns, int rows, int walls) => columns * rows * walls;

        void BuildSimulation()
        {
            DestroySimulation();

            columns = Mathf.Clamp(columns, MinColumns, MaxColumns);
            rows = Mathf.Clamp(rows, MinRows, MaxRows);
            walls = Mathf.Clamp(walls, MinWalls, MaxWalls);
            int blocks = BlockCount(columns, rows, walls); // of all the walls
            // Bodies: the ground, the blocks and the shots.
            maxShots = Mathf.Max(maxShots, 1);
            int capacity = 1 + blocks + maxShots + 8;
            bodies = new AvbdBodyBuffers(capacity);
            if (bodies.State.stride != UnsafeUtility.SizeOf<BodyStateRecord>())
                throw new System.InvalidOperationException("The package's body state record changed; update BodyStateRecord in GpuFractureWallDemo.");
            staticInstances.Clear();

            AvbdSolverSettings settings = AvbdSolverSettings.Default; // 60 Hz, gravity (0, -10, 0): Unity's y-up
            settings.iterations = iterations;
            // Up to three joints per block (the one beside it, two under it). Manifold slots: a block touches about six others and a shot can
            // lie on or inside a dozen, so this has room to spare (pairs a joint joins make no manifold); a pair that finds no slot just gets
            // no contacts until one is free.
            int manifoldSlots = Mathf.Min(blocks * 8 + maxShots * 16 + 256, AvbdConstraintBuffers.MaxContactManifolds);
            // The plastic array only exists if the joints can be plastic: without it the solver runs the joint kernels with no plastic code at all.
            int jointCapacity = blocks * 3 + 8;
            constraints = new AvbdConstraintBuffers(bodies, jointCapacity, null, manifoldSlots, plasticJoints ? jointCapacity : 0);
            solver = new AvbdSolver(bodies, settings, null, constraints);
            raycaster = new AvbdRaycaster(bodies);
            labeler = new AvbdGroupLabeler(bodies);
            simulationVersion++;

            // The ground and every block go in one batch (the ground is element 0, then the blocks wall by wall, row by row, column by column), so that
            // the bodies can be placed in Z-curve order. The handles come back in the order of the list; the joints and the instances use them.
            // Walls: wall w stands at z = w * WallSpacing, the first one in front. Row r is shifted by half a block when r is odd (a running bond),
            // so every block rests on two blocks of the row under it.
            int perWall = rows * columns;
            var descriptors = new List<BodyDescriptor>(1 + blocks);
            var instanceSizes = new List<float3>(1 + blocks);
            var instanceColors = new List<Color>(1 + blocks);
            var positions = new float3[walls * perWall];

            // Ground: a static slab whose top face is the plane y = 0.
            var ground = new float3(GroundHalfExtent * 2f, 1f, GroundHalfExtent * 2f);
            descriptors.Add(DemoBox(ground, 0f, new float3(0f, -0.5f, 0f)));
            instanceSizes.Add(ground);
            instanceColors.Add(new Color(0.30f, 0.33f, 0.36f));

            float3 size = new float3(blockSize.x, blockSize.y, blockSize.z);
            float stepX = size.x + BlockGap;
            float stepY = size.y + BlockGap;
            float centerColumn = (columns - 1) * 0.5f + 0.25f;
            for (int w = 0; w < walls; w++)
            {
                float z = w * WallSpacing;
                for (int r = 0; r < rows; r++)
                {
                    float y = size.y * 0.5f + BlockGap + r * stepY;
                    Color color = Color.HSVToRGB(0.02f + 0.08f * (r % 2), 0.55f + 0.1f * (r % 3) / 2f, 0.85f);
                    for (int c = 0; c < columns; c++)
                    {
                        var position = new float3((c + 0.5f * (r & 1) - centerColumn) * stepX, y, z);
                        positions[w * perWall + r * columns + c] = position;
                        // The bottom row is immobile: a body without mass is static.
                        descriptors.Add(DemoBox(size, r == 0 ? 0f : blockDensity, position));
                        instanceSizes.Add(size);
                        instanceColors.Add(color);
                    }
                }
            }
            var spawned = new BodyHandle[descriptors.Count];
            bodies.SpawnBatch(descriptors, spawned, zCurveOrder);
            for (int i = 0; i < spawned.Length; i++)
                AddStaticInstance(spawned[i], instanceSizes[i], instanceColors[i]);

            // Joints. The bottom row is not joined to itself, only to the row above it. A block is joined to the block on its left, and to the two blocks under it: with a shift of half a block, those are
            // (c, c + 1) in an odd row and (c - 1, c) in an even one.
            jointCount = 0;
            for (int w = 0; w < walls; w++)
            {
                int first = w * perWall;
                for (int r = 1; r < rows; r++)   // the bottom row is static: its blocks need no joints to each other
                    for (int c = 0; c < columns; c++)
                    {
                        BodyHandle self = spawned[1 + first + r * columns + c];
                        float3 selfPosition = positions[first + r * columns + c];
                        if (c > 0)
                            Join(self, selfPosition, spawned[1 + first + r * columns + c - 1], positions[first + r * columns + c - 1]);
                        int firstBelow = (r & 1) == 1 ? c : c - 1;
                        for (int k = 0; k < 2; k++)
                        {
                            int below = firstBelow + k;
                            if (below >= 0 && below < columns)
                                Join(self, selfPosition, spawned[1 + first + (r - 1) * columns + below], positions[first + (r - 1) * columns + below]);
                        }
                    }
            }
            bodies.ApplyEdits();

            CreateInstanceBuffer();

            accumulator = 0f;
            stepMs = 0f;
        }

        // A ball-socket joint at the middle between the two blocks (the middle of the face or of the part of it they share), with a rigid
        // angle lock. Both anchors are that point, so the joint starts satisfied.
        void Join(BodyHandle a, float3 positionA, BodyHandle b, float3 positionB)
        {
            if (!a.IsValid || !b.IsValid)
                return;
            float3 anchor = (positionA + positionB) * 0.5f;
            float linear = rigidJoints ? float.PositiveInfinity : jointStiffness;
            float yield = plasticJoints ? yieldThreshold : float.PositiveInfinity;
            float bendLimit = plasticJoints && bendBreak ? Mathf.Min(bendBreakDegrees * Mathf.Deg2Rad, Mathf.PI) : float.PositiveInfinity;
            if (constraints.AddJoint(a, b, anchor - positionA, anchor - positionB, linear, float.PositiveInfinity, fractureThreshold, yield, bendLimit) >= 0)
                jointCount++;
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

        // A box of the demo: in DemoGroup only.
        static BodyDescriptor DemoBox(float3 size, float density, float3 position)
        {
            BodyDescriptor d = BodyDescriptor.Box(size, density, position);
            d.groupMask = DemoGroup;
            return d;
        }

        void DestroySimulation()
        {
            // The solver and the buffers go with their fields and labels: nothing to remove one by one.
            grabActive = false;
            grabHeld = false;
            grabToken++;
            grabField = ForceFieldHandle.Invalid;
            grabStatus = "";
            if (labeler != null) labeler.Dispose();
            labeler = null;
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
            BodyDescriptor descriptor = DemoBox(new float3(size), shotMass / (size * size * size), (float3)t.position + direction * (size + 0.6f));
            descriptor.rotation = t.rotation;
            descriptor.linearVelocity = direction * shotSpeed;

            BodyHandle handle = bodies.Spawn(descriptor);
            if (!handle.IsValid)
                return;

            shots.Add(new Shot { handle = handle, size = size });
            shotsChanged = true;
        }

        // The volley of the performance tests (FractureWallScene.ShootVolley): 15 shots from a grid of 5 columns by 3 rows in front of the first wall, 10 m
        // away, all fired at once along +z with the panel's speed, size and mass. At the default wall (75 x 75 blocks) the grid is the test's (x = -30 to 30 m,
        // y = 6, 19 and 32 m); a smaller wall gets a proportionally smaller grid.
        static readonly float[] VolleyX = { -30f, -15f, 0f, 15f, 30f };
        static readonly float[] VolleyY = { 6f, 19f, 32f };
        const float VolleyZ = -10f;

        void FireVolley()
        {
            int volley = VolleyX.Length * VolleyY.Length;
            int shotLimit = Mathf.Min(maxShots, instances.Length - staticCount);
            while (shots.Count > 0 && shots.Count + volley > shotLimit)
                RemoveShot(0);

            float size = shotSize;
            float density = shotMass / (size * size * size);
            for (int i = 0; i < volley && shots.Count < shotLimit; i++)
            {
                float x = VolleyX[i % VolleyX.Length] * (columns / 75f);
                float y = VolleyY[i / VolleyX.Length] * (rows / 75f);
                BodyDescriptor descriptor = DemoBox(new float3(size), density, new float3(x, y, VolleyZ));
                descriptor.linearVelocity = new float3(0f, 0f, shotSpeed);
                BodyHandle handle = bodies.Spawn(descriptor);
                if (!handle.IsValid)
                    break;
                shots.Add(new Shot { handle = handle, size = size });
            }
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

        // Where the camera is and which way it looks: the pose of this frame (the camera object gets it in LateUpdate).
        void GetCameraRay(out float3 origin, out float3 direction)
        {
            if (controlCamera || cam == null)
            {
                origin = camPosition;
                direction = Quaternion.Euler(pitch, yaw, 0f) * Vector3.forward;
            }
            else
            {
                origin = cam.transform.position;
                direction = cam.transform.forward;
            }
        }

        // The spring of the grab: the held point is on the line of sight at grabDistance, and the stiffness grows with how far the grabbed blocks are from that line,
        // that is, from the center of the screen (the field does it on the GPU, from the centroid of the blocks).
        ForceFieldDescriptor GrabSpring(float3 origin, float3 direction)
        {
            float3 forward = math.normalizesafe(direction);
            return ForceFieldDescriptor.Spring(GrabGroup, origin + forward * grabDistance, grabForce, grabDamping, forward, grabOffsetGain);
        }

        void StartGrab()
        {
            if (cam == null || grabActive)
                return;

            GetCameraRay(out float3 origin, out float3 direction);
            float radius = grabRadius;
            int version = simulationVersion;
            int token = ++grabToken;

            grabStatus = "Casting the ray...";
            raycaster.Cast(origin, direction, RayMaxDistance, (hit, distance) =>
            {
                if (token != grabToken || version != simulationVersion || solver == null || !grabHeld || clickMode != ClickMode.Grab)
                    return; // the button was released, the mode changed or the simulation was rebuilt while the ray was on its way
                if (!hit)
                {
                    grabStatus = "Nothing to grab: the ray from the camera hit nothing.";
                    return;
                }

                float3 point = origin + math.normalizesafe(direction) * distance;
                labeler.Label(point, radius, GrabGroup, 0u);

                grabDistance = distance;
                ForceFieldHandle field = solver.AddForceField(GrabSpring(origin, direction));
                if (!field.IsValid)
                {
                    labeler.ClearAll(GrabGroup);
                    grabStatus = "Not grabbed: no force field left.";
                    return;
                }
                grabField = field;
                grabActive = true;
                grabStatus = "";
            });
        }

        // Every frame while a grab is held: the target is the point the camera looks at, at the distance of the grab, and the sliders of the panel apply.
        void UpdateGrab()
        {
            if (!grabActive || solver == null)
                return;

            GetCameraRay(out float3 origin, out float3 direction);
            solver.UpdateForceField(grabField, GrabSpring(origin, direction));
        }

        void EndGrab()
        {
            if (!grabActive)
                return;
            grabToken++;
            grabActive = false;
            if (solver != null)
                solver.RemoveForceField(grabField);
            grabField = ForceFieldHandle.Invalid;
            if (labeler != null)
                labeler.ClearAll(GrabGroup);
            grabStatus = "Released.";
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
            // The release of the button ends a grab wherever it happens, also over the panel, where a control may use the event (rawType is the original).
            if (e.rawType == EventType.MouseUp && e.button == 0)
                grabHeld = false;

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

            // What a left click does. The buttons only ask: the mode changes in Update, so the controls below never change in the middle of an event.
            GUILayout.Label("Left click:", boldLabel);
            GUILayout.BeginHorizontal();
            for (int i = 0; i < ClickModeNames.Length; i++)
            {
                if (GUILayout.Toggle(clickMode == (ClickMode)i, ClickModeNames[i], GUI.skin.button) && clickMode != (ClickMode)i)
                    requestMode = i;
            }
            GUILayout.EndHorizontal();

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
            else if (clickMode == ClickMode.Explode)
            {
                GUILayout.Label("Radius: " + explosionRadius.ToString("0.0") + " m");
                explosionRadius = GUILayout.HorizontalSlider(explosionRadius, 1f, 50f);

                // Force on a log scale: 0.1 to 1000 N s. A block (0.25 kg by default) at the center gets force / mass.
                float blockMass = blockDensity * blockSize.x * blockSize.y * blockSize.z;
                GUILayout.Label("Force: " + explosionForce.ToString(explosionForce < 10f ? "0.0" : "0") + " N s   (" + (explosionForce / blockMass).ToString("0") + " m/s for a block at the center)");
                explosionForce = Mathf.Clamp(Mathf.Pow(10f, GUILayout.HorizontalSlider(Mathf.Log10(explosionForce), -1f, 3f)), 0.1f, 1000f);

                GUILayout.Label("Depth: " + explosionDepth.ToString("0.0") + " m   (behind the surface the camera ray hits)");
                explosionDepth = GUILayout.HorizontalSlider(explosionDepth, 0f, 20f);

                GUILayout.Label(explosionStatus);
            }
            else
            {
                GUILayout.Label("Radius: " + grabRadius.ToString("0.0") + " m   (blocks around the point the ray hits)");
                grabRadius = GUILayout.HorizontalSlider(grabRadius, 0.1f, 10f);

                // Stiffness on a log scale: 10 to 2000 N/m. Each block gets this times the distance of the group from the held point, in newtons.
                GUILayout.Label("Force: " + grabForce.ToString("0") + " N/m   (a pull of this many N per meter the blocks are from the held point)");
                grabForce = Mathf.Clamp(Mathf.Pow(10f, GUILayout.HorizontalSlider(Mathf.Log10(grabForce), 1f, Mathf.Log10(2000f))), 10f, 2000f);

                GUILayout.Label("Damping ratio: " + grabDamping.ToString("0.00") + "   (0 swings, 1 is critical)");
                grabDamping = GUILayout.HorizontalSlider(grabDamping, 0f, 3f);

                GUILayout.Label("Offset gain: " + grabOffsetGain.ToString("0.00") + " per m   (the force grows with the distance of the blocks from the center of the screen)");
                grabOffsetGain = GUILayout.HorizontalSlider(grabOffsetGain, 0f, 5f);

                GUILayout.Label(grabActive ? "Holding at " + grabDistance.ToString("0.0") + " m from the camera" : grabStatus);
            }

            GUILayout.Space(4f);
            GUILayout.Label("Iterations: " + iterations);
            iterations = Mathf.RoundToInt(GUILayout.HorizontalSlider(iterations, 1, 30));

            GUILayout.Space(4f);
            GUILayout.Label("Wall size (Reset to apply)", boldLabel);
            GUILayout.BeginHorizontal();
            GUILayout.Label("C " + columns, GUILayout.Width(38f));
            columns = Mathf.RoundToInt(GUILayout.HorizontalSlider(columns, MinColumns, MaxColumns));
            GUILayout.Label("R " + rows, GUILayout.Width(38f));
            rows = Mathf.RoundToInt(GUILayout.HorizontalSlider(rows, MinRows, MaxRows));
            GUILayout.Label("W " + walls, GUILayout.Width(38f));
            walls = Mathf.RoundToInt(GUILayout.HorizontalSlider(walls, MinWalls, MaxWalls));
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            if (GUILayout.Button(paused ? "Play" : "Pause")) paused = !paused;
            if (GUILayout.Button("Step")) { paused = true; requestStep = true; }
            if (GUILayout.Button("Clear shots")) requestClearShots = true;
            if (GUILayout.Button("Reset")) requestReset = true;
            GUILayout.EndHorizontal();

            // Density, stiffness and the threshold on log scales.
            GUILayout.Label("Block density: " + blockDensity.ToString("0.00") + "   (" + (blockDensity * blockSize.x * blockSize.y * blockSize.z).ToString("0.00") + " kg per block)");
            blockDensity = Mathf.Clamp(Mathf.Pow(10f, GUILayout.HorizontalSlider(Mathf.Log10(blockDensity), -1f, 1f)), 0.1f, 10f);

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
            GUILayout.Space(4f);
            GUILayout.Space(4f);
            GUILayout.Label("bodies " + (staticCount + shots.Count) + "   shots " + shots.Count + "/" + maxShots);
            GUILayout.Label("joints broken " + (constraints != null ? constraints.BrokenCount : 0) + " / " + jointCount);
            GUILayout.Label("step " + stepMs.ToString("0.00") + " ms (CPU side)   frame " + frameMs.ToString("0.0") + " ms");
            if (behindRealTime)
                GUILayout.Label("running slower than real time");

            GUILayout.Space(4f);
            GUILayout.Label("Left click: explosion, shot, or hold to grab (the mode buttons)   V: volley\nSpace: pause   R: reset\nRight drag: look   W A S D Q E: move (Shift: faster)\nMouse wheel: move forward and back");

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
                    else if (e.keyCode == KeyCode.V) { requestVolley = true; e.Use(); }
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
                {
                    requestExplode = true;
                }
                else if (clickMode == ClickMode.Shoot)
                {
                    requestShoot = true;
                }
                else
                {
                    grabHeld = true;
                    requestGrabStart = true;
                }
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

        static readonly string[] ClickModeNames = { "Explosion", "Shot", "Grab" };

        static readonly Color ShotColor = new Color(1f, 0.25f, 0.15f);
    }

#if UNITY_EDITOR
    static class GpuFractureWallDemoMenu
    {
        [UnityEditor.MenuItem("GameObject/McKrill AVBD/GPU Fracture Wall Demo", false, 11)]
        static void Create(UnityEditor.MenuCommand command)
        {
            var go = new GameObject("AVBD GPU Fracture Wall Demo");
            UnityEditor.GameObjectUtility.SetParentAndAlign(go, command.context as GameObject);
            go.AddComponent<GpuFractureWallDemo>();
            UnityEditor.Undo.RegisterCreatedObjectUndo(go, "Create AVBD GPU Fracture Wall Demo");
            UnityEditor.Selection.activeObject = go;
        }
    }
#endif
}
