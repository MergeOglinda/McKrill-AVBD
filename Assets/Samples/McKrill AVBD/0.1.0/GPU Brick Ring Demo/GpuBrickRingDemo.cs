using System.Collections.Generic;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd.Samples
{
    /// <summary>
    /// The brick ring of three-avbd (paper Fig. 1) on the GPU solver, the scene of the performance test <c>Scenario5_BrickRing110k</c> (see
    /// <c>Tests/Performance/BrickRingScene.cs</c>): 40 courses of bricks laid tangentially in running bond on rings from a radius of 80 m outwards, 10 bricks
    /// deep at the bottom and one fewer every 4 courses, so the outside is stepped and the inside sheer; 110 440 bricks with the defaults. The bricks rest exactly on
    /// each other, so it stands from the first step. A cube with the volume and density of an 8 m ball starts just outside the stepped face, on its way in at
    /// 30 m/s. You can shoot boxes from the camera as well.
    ///
    /// The scene is the performance test's, so what you see is what it times: the same bodies, the same capacities (the contact pool is the largest the package
    /// allows with the defaults, which needs a lot of GPU memory) and the same 4 solver iterations. The panel's ring settings take effect on Reset.
    /// The bodies can be spawned in the order the ring is built (course by course) or in Z-curve order of their positions (<see cref="AvbdBodyBuffers.SpawnBatch"/>),
    /// to compare the two.
    ///
    /// The simulation runs entirely through the package's public API; drawing never leaves the GPU: one instanced draw call draws every box, and the vertex
    /// shader (<c>AvbdInstancedBoxes</c>) reads each box's pose from the solver's own body state buffer, as in the GPU Brick Ring Demo, which this demo
    /// follows. The shader has no shadows.
    ///
    /// The scene is y-up (the package default), so solver positions and rotations are used as Unity world positions and rotations.
    /// Input is read through IMGUI events, so it works with the Input Manager and with the Input System package.
    /// </summary>
    [AddComponentMenu("McKrill AVBD/GPU Brick Ring Demo")]
    [DisallowMultipleComponent]
    public sealed class GpuBrickRingDemo : MonoBehaviour
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

        [Header("Ring (applied on Reset)")]
        [Tooltip("Courses (layers) of bricks. 40 courses with 10 rows are 110 440 bricks.")]
        [SerializeField, Range(MinCourses, MaxCourses)] int courses = 40;
        [Tooltip("Bricks deep in the lowest course; one fewer every 4 courses, so the outside of the ring is stepped.")]
        [SerializeField, Range(1, MaxDepth)] int depth = 10;
        [Tooltip("Inner radius of the ring in meters.")]
        [SerializeField, Range(MinRadius, MaxRadius)] float ringRadius = 80f;
        [Tooltip("Place the bodies in the GPU arrays in Z-curve order of their positions (AvbdBodyBuffers.SpawnBatch), so that bricks that touch are near each other in memory. Off: in the order the ring is built, course by course.")]
        [SerializeField] bool zCurveOrder = false;
        [Tooltip("The cube that starts outside the stepped face and flies into the ring.")]
        [SerializeField] bool ball = true;
        [SerializeField, Range(0f, 80f)] float ballSpeed = 30f;

        [Header("Solver")]
        [Tooltip("Solver iterations per step: 4 in the performance test.")]
        [SerializeField, Range(1, 30)] int iterations = 4;
        // The solver runs this many steps per rendered frame, always, whatever the frame time: the simulation advances StepsPerFrame time steps
        // every frame. It does not wait for the time of a step to pass and does not skip steps when a frame is slow.
        const int StepsPerFrame = 2;

        [Header("Shots")]
        [Tooltip("Speed of a shot in meters per second.")]
        [SerializeField, Range(5f, 80f)] float shotSpeed = 30f;
        [Tooltip("Edge length of the shot, a cube.")]
        [SerializeField, Range(0.2f, 4f)] float shotSize = 2f;
        [Tooltip("Mass of the shot. A brick weighs 0.25 kg.")]
        [SerializeField, Range(0.1f, 2000f)] float shotMass = 200f;
        [Tooltip("Shots alive at once. Shooting more removes the oldest.")]
        [SerializeField, Min(1)] int maxShots = 100;

        [Header("View")]
        [Tooltip("Fly camera: right drag to look, W A S D to move, Q and E for down and up, Shift for faster, mouse wheel to dolly.")]
        [SerializeField] bool controlCamera = true;
        [SerializeField, Min(0.1f)] float moveSpeed = 30f;
        [SerializeField, Min(0.01f)] float lookSensitivity = 0.2f;

        const int MinCourses = 4;
        const int MaxCourses = 40;
        const int MaxDepth = 10;
        const float MinRadius = 20f;
        const float MaxRadius = 100f;
        const int Tier = 4;
        const float BrickLength = 1f;
        const float BrickHeight = 0.5f;
        const float BrickDensity = 1f;
        const float BallDensity = 10f;

        AvbdBodyBuffers bodies;
        AvbdConstraintBuffers constraints;
        AvbdSolver solver;
        // Drawing: one instance per body. The first staticCount instances (the ground, the bricks, the ball) are written once per Reset, the shots follow.
        InstanceData[] instances;
        GraphicsBuffer instanceBuffer;
        int staticCount;
        int brickCount;
        bool shotsChanged;
        float buildMs;

        readonly List<InstanceData> staticInstances = new List<InstanceData>(); // ground, bricks, ball
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

        bool paused;
        bool tunnelRisk; // set in Update, not in OnGUI: a control that appears or disappears between Layout and Repaint is an IMGUI error
        float stepMs;
        float frameMs;

        // Requests made from OnGUI, applied in Update so the IMGUI control count never changes between Layout and Repaint.
        bool requestShoot;
        bool requestReset;
        bool requestClearShots;
        bool requestStep;

        Rect panelScreenRect;
        GUIStyle boldLabel;

        // ---- lifecycle ---------------------------------------------------------------------------------

        void OnEnable()
        {
            Shader boxShader = AvbdShaderSet.Load()?.InstancedBoxes;
            if (boxShader == null || !boxShader.isSupported)
            {
                Debug.LogError("GPU Brick Ring Demo: the package's instanced box shader is missing or not supported on this device (it needs compute buffers in the vertex shader, shader model 4.5).");
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

            // Camera: outside the stepped face, a little to the side and above, looking at the middle of the face the ball will hit. Reset leaves it where it is.
            float outside = ringRadius + 0.52f * depth;
            camPosition = new Vector3(-0.9f * outside, 0.9f * courses * BrickHeight + 6f, 1.35f * outside);
            Vector3 target = new Vector3(0f, 0.4f * courses * BrickHeight, outside - 30f);
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
                for (int i = 0; i < StepsPerFrame; i++)
                    StepOnce();
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

        /// <summary>Bricks on a ring of radius r: as many as fit on its inner edge with 2% gaps.</summary>
        static int RingCount(float r) => Mathf.FloorToInt(2f * Mathf.PI * (r - 0.25f) / 1.02f);

        /// <summary>Number of bricks of a ring (110 440 for 40 courses, 10 rows and radius 80).</summary>
        public static int BrickCount(int courses, int depth, float radius)
        {
            int total = 0;
            for (int c = 0; c < courses; c++)
            {
                int rowsInCourse = Mathf.Max(1, depth - c / Tier);
                for (int j = 0; j < rowsInCourse; j++)
                    total += RingCount(radius + 0.25f + 0.52f * j);
            }
            return total;
        }

        void BuildSimulation()
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            DestroySimulation();

            courses = Mathf.Clamp(courses, MinCourses, MaxCourses);
            depth = Mathf.Clamp(depth, 1, MaxDepth);
            ringRadius = Mathf.Clamp(ringRadius, MinRadius, MaxRadius);
            int bricks = BrickCount(courses, depth, ringRadius);
            // Bodies: the ground, the bricks, the ball and the shots.
            maxShots = Mathf.Max(maxShots, 1);
            int capacity = 1 + bricks + 1 + maxShots + 8;
            bodies = new AvbdBodyBuffers(capacity);
            if (bodies.State.stride != UnsafeUtility.SizeOf<BodyStateRecord>())
                throw new System.InvalidOperationException("The package's body state record changed; update BodyStateRecord in GpuBrickRingDemo.");
            staticInstances.Clear();

            AvbdSolverSettings settings = AvbdSolverSettings.Default; // 60 Hz, gravity (0, -10, 0): Unity's y-up
            settings.iterations = iterations;
            // The contact pool is the largest the package allows when the ring is large, like the performance test: a ring this size has more touching
            // pairs than it can hold, and a pair that finds no slot gets no contacts. Smaller rings get less.
            int manifoldSlots = Mathf.Min(bricks * 8 + maxShots * 16 + 256, AvbdConstraintBuffers.MaxContactManifolds);
            constraints = new AvbdConstraintBuffers(bodies, 1, null, manifoldSlots); // no joints
            solver = new AvbdSolver(bodies, settings, null, constraints);

            // All the bodies in one batch: the ground, the bricks course by course and ring by ring, the ball. The colors go with them.
            var descriptors = new List<BodyDescriptor>(capacity);
            var sizes = new List<float3>(capacity);
            var colors = new List<Color>(capacity);

            float extent = ringRadius + depth;
            float groundSize = Mathf.Max(200f, 4f * extent); // room for the debris to scatter
            var ground = new float3(groundSize, 1f, groundSize);
            descriptors.Add(BodyDescriptor.Box(ground, 0f, new float3(0f, -0.5f, 0f)));
            sizes.Add(ground);
            colors.Add(new Color(0.30f, 0.33f, 0.36f));

            var brickSize = new float3(BrickLength, BrickHeight, BrickHeight);
            for (int c = 0; c < courses; c++)
            {
                int rowsInCourse = Mathf.Max(1, depth - c / Tier);
                for (int j = 0; j < rowsInCourse; j++)
                {
                    float r = ringRadius + 0.25f + 0.52f * j; // rings 2 cm apart: a straight brick's corners reach past its ring's outer radius
                    int n = RingCount(r);
                    Color color = Color.HSVToRGB(0.02f + 0.10f * (c % 2) + 0.004f * j, 0.55f + 0.1f * (c % 3) / 2f, 0.80f + 0.015f * (j % 3));
                    for (int i = 0; i < n; i++)
                    {
                        float a = (i + (c % 2) * 0.5f) / n * 2f * Mathf.PI; // odd courses are offset by half a brick
                        BodyDescriptor brick = BodyDescriptor.Box(brickSize, BrickDensity,
                            new float3(r * Mathf.Cos(a), BrickHeight * 0.5f + BrickHeight * c, -r * Mathf.Sin(a)));
                        brick.rotation = quaternion.AxisAngle(new float3(0f, 1f, 0f), a + Mathf.PI * 0.5f);
                        descriptors.Add(brick);
                        sizes.Add(brickSize);
                        colors.Add(color);
                    }
                }
            }

            // The ball: a cube of the volume of a sphere of radius courses / 5 (8 m for 40 courses), just outside the stepped face, moving in.
            if (ball)
            {
                float ballRadius = courses / 5f;
                float side = Mathf.Pow(4f / 3f * Mathf.PI, 1f / 3f) * ballRadius;
                float outside = ringRadius + 0.52f * depth + ballRadius + 2f;
                BodyDescriptor ballBody = BodyDescriptor.Box(new float3(side), BallDensity, new float3(0f, side * 0.5f, outside));
                ballBody.linearVelocity = new float3(0f, 0f, -ballSpeed);
                descriptors.Add(ballBody);
                sizes.Add(new float3(side));
                colors.Add(new Color(1f, 0.45f, 0.1f));
            }

            var handles = new BodyHandle[descriptors.Count];
            bodies.SpawnBatch(descriptors, handles, zCurveOrder);
            brickCount = 0;
            for (int i = 0; i < handles.Length; i++)
            {
                if (!handles[i].IsValid)
                    continue;
                staticInstances.Add(MakeInstance(handles[i], sizes[i], colors[i]));
                if (i > 0 && !(ball && i == handles.Length - 1))
                    brickCount++;
            }
            bodies.ApplyEdits();

            CreateInstanceBuffer();
            stepMs = 0f;
            watch.Stop();
            buildMs = (float)watch.Elapsed.TotalMilliseconds;
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

            GUILayout.Space(4f);
            GUILayout.Label("Ring (Reset to apply)", boldLabel);
            GUILayout.Label("Courses: " + courses + "   deep: " + depth + "   radius: " + ringRadius.ToString("0") + " m");
            GUILayout.BeginHorizontal();
            GUILayout.Label("C " + courses, GUILayout.Width(38f));
            courses = Mathf.RoundToInt(GUILayout.HorizontalSlider(courses, MinCourses, MaxCourses));
            GUILayout.Label("D " + depth, GUILayout.Width(38f));
            depth = Mathf.RoundToInt(GUILayout.HorizontalSlider(depth, 1, MaxDepth));
            GUILayout.Label("R " + ringRadius.ToString("0"), GUILayout.Width(44f));
            ringRadius = Mathf.Round(GUILayout.HorizontalSlider(ringRadius, MinRadius, MaxRadius));
            GUILayout.EndHorizontal();
            GUILayout.Label("Bricks after Reset: " + BrickCount(courses, depth, ringRadius));
            zCurveOrder = GUILayout.Toggle(zCurveOrder, "Z-curve body order (off: the order the ring is built)");
            ball = GUILayout.Toggle(ball, "Ball");
            GUILayout.Label("Ball speed: " + ballSpeed.ToString("0") + " m/s");
            ballSpeed = Mathf.Round(GUILayout.HorizontalSlider(ballSpeed, 0f, 80f));

            GUILayout.BeginHorizontal();
            if (GUILayout.Button(paused ? "Play" : "Pause")) paused = !paused;
            if (GUILayout.Button("Step")) { paused = true; requestStep = true; }
            if (GUILayout.Button("Clear shots")) requestClearShots = true;
            if (GUILayout.Button("Reset")) requestReset = true;
            GUILayout.EndHorizontal();

            GUILayout.Space(4f);
            GUILayout.Label("bodies " + (staticCount + shots.Count) + "   bricks " + brickCount + "   shots " + shots.Count + "/" + maxShots);
            GUILayout.Label("build " + buildMs.ToString("0") + " ms (CPU side, with the batch spawn)");
            GUILayout.Label("step " + stepMs.ToString("0.00") + " ms (CPU side)   frame " + frameMs.ToString("0.0") + " ms");

            GUILayout.Space(4f);
            GUILayout.Label("Left click: shoot   Space: pause   R: reset\nRight drag: look   W A S D Q E: move (Shift: faster)\nMouse wheel: move forward and back");

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
    static class GpuBrickRingDemoMenu
    {
        [UnityEditor.MenuItem("GameObject/McKrill AVBD/GPU Brick Ring Demo", false, 12)]
        static void Create(UnityEditor.MenuCommand command)
        {
            var go = new GameObject("AVBD GPU Brick Ring Demo");
            UnityEditor.GameObjectUtility.SetParentAndAlign(go, command.context as GameObject);
            go.AddComponent<GpuBrickRingDemo>();
            UnityEditor.Undo.RegisterCreatedObjectUndo(go, "Create AVBD GPU Brick Ring Demo");
            UnityEditor.Selection.activeObject = go;
        }
    }
#endif
}
