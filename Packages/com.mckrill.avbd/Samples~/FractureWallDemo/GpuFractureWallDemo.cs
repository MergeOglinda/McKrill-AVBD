using System.Collections.Generic;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd.Samples
{
    /// <summary>
    /// A brick wall of 225 or more blocks (15 x 15 by default) with an immobile bottom row, on the GPU solver, held together by joints that fracture, and a camera you can shoot boxes from.
    ///
    /// Every block is joined to its neighbors (the one beside it and the two under it in a running bond) by a ball-socket joint at the
    /// middle of the shared face, with a hard angle lock. A joint breaks when its angular multiplier exceeds the fracture threshold
    /// (stage D). The panel sets the properties of the joints and the blocks; they take effect on Reset, because the package cannot change a
    /// joint after it was added. A fractured joint leaves the solver, and its two blocks collide again like any other pair.
    ///
    /// The simulation runs entirely through the package's public API: <see cref="AvbdBodyBuffers"/> holds the bodies,
    /// <see cref="AvbdSolver.Step"/> advances them. Collisions are part of that step: the broadphase finds the touching pairs, the narrow
    /// phase keeps a manifold for each, and the solver solves the contacts, so this demo contains no collision code.
    ///
    /// Drawing is the simple part: each frame the poses are read back from the GPU and copied to one cube per body. That is a
    /// stall and a per-body GameObject, fine for a few hundred boxes and not what the package's own renderer will do.
    ///
    /// The scene is y-up (the package default), so solver positions and rotations are used as Unity world positions and rotations.
    /// Input is read through IMGUI events, so it works with the Input Manager and with the Input System package.
    /// </summary>
    [AddComponentMenu("McKrill AVBD/GPU Fracture Wall Demo")]
    [DisallowMultipleComponent]
    public sealed class GpuFractureWallDemo : MonoBehaviour
    {
        // Mirror of the package's state record (design doc, section 3): position, rotation, start-of-step position and rotation.
        // The package keeps its own struct internal; the demo reads the public buffer with this copy and checks the stride.
        struct BodyStateRecord
        {
            public float4 position;
            public float4 rotation;
            public float4 startPosition;
            public float4 startRotation;
        }

        struct Visual
        {
            public BodyHandle handle;
            public Transform transform;
            public Material material;
        }

        [Header("Wall (applied on Reset)")]
        [Tooltip("Blocks per row. 15 columns and 15 rows are 225 blocks.")]
        [SerializeField, Range(MinColumns, MaxColumns)] int columns = 15;
        [Tooltip("Rows of blocks. The bottom row is immobile.")]
        [SerializeField, Range(MinRows, MaxRows)] int rows = 15;
        [SerializeField] Vector3 blockSize = new Vector3(1f, 0.5f, 0.5f);
        [SerializeField, Min(0.01f)] float blockDensity = 1f;
        [Tooltip("Joints are rigid (infinite stiffness) instead of springs with the stiffness below.")]
        [SerializeField] bool rigidJoints = true;
        [Tooltip("Stiffness of the joints' linear part when they are not rigid. The angle lock is always rigid: only a rigid angle lock stores the multiplier that fracture tests.")]
        [SerializeField, Min(1f)] float jointStiffness = 1e5f;
        [Tooltip("A joint breaks when the norm of its angular multiplier is above this.")]
        [SerializeField, Min(0.01f)] float fractureThreshold = 100f;

        [Header("Solver")]
        [SerializeField, Range(1, 30)] int iterations = 10;
        [Tooltip("Upper bound on solver steps per rendered frame. If the GPU cannot keep up, the simulation runs slower than real time instead of stalling.")]
        [SerializeField, Range(1, 4)] int maxStepsPerFrame = 2;

        [Header("Shots")]
        [Tooltip("Speed of a shot in meters per second.")]
        [SerializeField, Range(5f, 80f)] float shotSpeed = 30f;
        [Tooltip("Edge length of the shot, a cube.")]
        [SerializeField, Range(0.2f, 2f)] float shotSize = 0.8f;
        [Tooltip("Mass of the shot. A wall block weighs its volume times the block density (0.25 by default).")]
        [SerializeField, Range(0.1f, 200f)] float shotMass = 5f;
        [Tooltip("Shots alive at once. Shooting more removes the oldest.")]
        [SerializeField, Min(1)] int maxShots = 100;

        [Header("View")]
        [Tooltip("Fly camera: right drag to look, W A S D to move, Q and E for down and up, Shift for faster, mouse wheel to dolly.")]
        [SerializeField] bool controlCamera = true;
        [SerializeField, Min(0.1f)] float moveSpeed = 8f;
        [SerializeField, Min(0.01f)] float lookSensitivity = 0.2f;

        // The wall sizes: 15 x 15 = 225 blocks at least, 22 x 18 = 396 at most. The limit is the cost of the constraint lists, which are
        // still built on the CPU every step (stage G moves them to the GPU).
        const int MinColumns = 15;
        const int MaxColumns = 22;
        const int MinRows = 15;
        const int MaxRows = 18;
        const float GroundHalfExtent = 100f;
        const float BlockGap = 0.01f; // between neighbors; the wall starts almost at rest

        AvbdBodyBuffers bodies;
        AvbdConstraintBuffers constraints;
        AvbdSolver solver;
        BodyStateRecord[] poses;
        int jointCount;

        readonly List<Visual> staticVisuals = new List<Visual>(); // ground and wall
        readonly List<Visual> shots = new List<Visual>();          // oldest first
        Shader boxShader;
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

        Rect panelScreenRect;
        GUIStyle boldLabel;

        // ---- lifecycle ---------------------------------------------------------------------------------

        void OnEnable()
        {
            boxShader = FindShader("Universal Render Pipeline/Lit", "Standard", "Sprites/Default", "Hidden/Internal-Colored");

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

            SyncVisuals();
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

        void BuildSimulation()
        {
            DestroySimulation();

            columns = Mathf.Clamp(columns, MinColumns, MaxColumns);
            rows = Mathf.Clamp(rows, MinRows, MaxRows);
            int blocks = BlockCount(columns, rows);
            // Bodies: the ground, the blocks and the shots.
            maxShots = Mathf.Max(maxShots, 1);
            int capacity = 1 + blocks + maxShots + 8;
            bodies = new AvbdBodyBuffers(capacity);
            if (bodies.State.stride != UnsafeUtility.SizeOf<BodyStateRecord>())
                throw new System.InvalidOperationException("The package's body state record changed; update BodyStateRecord in GpuFractureWallDemo.");
            poses = new BodyStateRecord[capacity];

            AvbdSolverSettings settings = AvbdSolverSettings.Default; // 60 Hz, gravity (0, -10, 0): Unity's y-up
            settings.iterations = iterations;
            // Up to three joints per block (the one beside it, two under it). Manifold slots: a block touches about six others and a shot can
            // lie on or inside a dozen, so this has room to spare (pairs a joint joins make no manifold); a pair that finds no slot just gets
            // no contacts until one is free.
            int manifoldSlots = Mathf.Min(blocks * 8 + maxShots * 16 + 256, AvbdConstraintBuffers.MaxContactManifolds);
            constraints = new AvbdConstraintBuffers(bodies, blocks * 3 + 8, null, manifoldSlots);
            solver = new AvbdSolver(bodies, settings, null, constraints);

            // Ground: a static slab whose top face is the plane y = 0.
            var ground = new float3(GroundHalfExtent * 2f, 1f, GroundHalfExtent * 2f);
            AddStaticVisual(bodies.Spawn(BodyDescriptor.Box(ground, 0f, new float3(0f, -0.5f, 0f))), ground, new Color(0.30f, 0.33f, 0.36f));

            // Wall: row r is shifted by half a block when r is odd (a running bond), so every block rests on two blocks of the row under it.
            float3 size = new float3(blockSize.x, blockSize.y, blockSize.z);
            float stepX = size.x + BlockGap;
            float stepY = size.y + BlockGap;
            float centerColumn = (columns - 1) * 0.5f + 0.25f;
            var handles = new BodyHandle[rows, columns];
            var positions = new float3[rows, columns];
            for (int r = 0; r < rows; r++)
            {
                float y = size.y * 0.5f + BlockGap + r * stepY;
                Color color = Color.HSVToRGB(0.02f + 0.08f * (r % 2), 0.55f + 0.1f * (r % 3) / 2f, 0.85f);
                for (int c = 0; c < columns; c++)
                {
                    var position = new float3((c + 0.5f * (r & 1) - centerColumn) * stepX, y, 0f);
                    // The bottom row is immobile: a body without mass is static.
                    handles[r, c] = bodies.Spawn(BodyDescriptor.Box(size, r == 0 ? 0f : blockDensity, position));
                    positions[r, c] = position;
                    AddStaticVisual(handles[r, c], size, color);
                }
            }

            // Joints. The bottom row is not joined to itself, only to the row above it. A block is joined to the block on its left, and to the two blocks under it: with a shift of half a block, those are
            // (c, c + 1) in an odd row and (c - 1, c) in an even one.
            jointCount = 0;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < columns; c++)
                {
                    if (r == 0)
                        continue; // the bottom row is static: its blocks need no joints to each other
                    if (c > 0)
                        Join(handles[r, c], positions[r, c], handles[r, c - 1], positions[r, c - 1]);
                    int firstBelow = (r & 1) == 1 ? c : c - 1;
                    for (int k = 0; k < 2; k++)
                    {
                        int below = firstBelow + k;
                        if (below >= 0 && below < columns)
                            Join(handles[r, c], positions[r, c], handles[r - 1, below], positions[r - 1, below]);
                    }
                }
            bodies.ApplyEdits();

            accumulator = 0f;
            stepMs = 0f;
            SyncVisuals();
        }

        // A ball-socket joint at the middle between the two blocks (the middle of the face or of the part of it they share), with a rigid
        // angle lock. Both anchors are that point, so the joint starts satisfied.
        void Join(BodyHandle a, float3 positionA, BodyHandle b, float3 positionB)
        {
            if (!a.IsValid || !b.IsValid)
                return;
            float3 anchor = (positionA + positionB) * 0.5f;
            float linear = rigidJoints ? float.PositiveInfinity : jointStiffness;
            if (constraints.AddJoint(a, b, anchor - positionA, anchor - positionB, linear, float.PositiveInfinity, fractureThreshold) >= 0)
                jointCount++;
        }

        void AddStaticVisual(BodyHandle handle, float3 size, Color color)
        {
            if (!handle.IsValid)
                return;
            staticVisuals.Add(CreateVisual(handle, size, color));
        }

        Visual CreateVisual(BodyHandle handle, float3 size, Color color)
        {
            var go = new GameObject("Body " + handle.Slot);
            go.transform.SetParent(transform, false);
            go.AddComponent<MeshFilter>().sharedMesh = cubeMesh;
            var material = new Material(boxShader) { color = color };
            go.AddComponent<MeshRenderer>().sharedMaterial = material;
            go.transform.localScale = new Vector3(size.x, size.y, size.z);
            return new Visual { handle = handle, transform = go.transform, material = material };
        }

        void DestroySimulation()
        {
            DestroyVisuals(staticVisuals);
            DestroyVisuals(shots);
            if (solver != null) solver.Dispose();
            if (constraints != null) constraints.Dispose();
            if (bodies != null) bodies.Dispose();
            solver = null;
            constraints = null;
            bodies = null;
            poses = null;
        }

        static void DestroyVisuals(List<Visual> visuals)
        {
            for (int i = 0; i < visuals.Count; i++)
            {
                if (visuals[i].material != null) Destroy(visuals[i].material);
                if (visuals[i].transform != null) Destroy(visuals[i].transform.gameObject);
            }
            visuals.Clear();
        }

        void Shoot()
        {
            if (cam == null)
                return;
            while (shots.Count >= maxShots)
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

            Visual visual = CreateVisual(handle, new float3(size), new Color(1f, 0.25f, 0.15f));
            visual.transform.SetPositionAndRotation(t.position + t.forward * (size + 0.6f), t.rotation);
            shots.Add(visual);
        }

        void RemoveShot(int index)
        {
            Visual visual = shots[index];
            bodies.Remove(visual.handle);
            if (visual.material != null) Destroy(visual.material);
            if (visual.transform != null) Destroy(visual.transform.gameObject);
            shots.RemoveAt(index);
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

        // One readback of the used slots, then every visual takes the pose of its slot.
        void SyncVisuals()
        {
            int count = Mathf.Min(bodies.HighWaterMark, poses.Length);
            if (count == 0)
                return;
            bodies.State.GetData(poses, 0, 0, count);
            ApplyPoses(staticVisuals, count);
            ApplyPoses(shots, count);
        }

        void ApplyPoses(List<Visual> visuals, int count)
        {
            for (int i = 0; i < visuals.Count; i++)
            {
                int slot = visuals[i].handle.Slot;
                if (slot >= count)
                    continue;
                BodyStateRecord s = poses[slot];
                visuals[i].transform.SetPositionAndRotation(
                    new Vector3(s.position.x, s.position.y, s.position.z),
                    new Quaternion(s.rotation.x, s.rotation.y, s.rotation.z, s.rotation.w));
            }
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
            const float width = 300f;
            float scale = Mathf.Clamp(Screen.height / 800f, 1f, 2.5f);

            Matrix4x4 oldMatrix = GUI.matrix;
            GUI.matrix = Matrix4x4.Scale(new Vector3(scale, scale, 1f));

            GUILayout.BeginArea(new Rect(margin, margin, width, Screen.height / scale - 2f * margin));
            GUILayout.BeginVertical(GUI.skin.box);

            GUILayout.Label("AVBD GPU solver: fracturing wall", boldLabel);

            GUILayout.Label("Shot", boldLabel);
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

            GUILayout.Space(4f);
            GUILayout.Label("Solver", boldLabel);
            GUILayout.Label("Iterations: " + iterations);
            iterations = Mathf.RoundToInt(GUILayout.HorizontalSlider(iterations, 1, 30));

            GUILayout.Space(4f);
            GUILayout.Label("Wall (applied on Reset)", boldLabel);
            GUILayout.Label("Columns: " + columns + "   Rows: " + rows + "   (" + BlockCount(columns, rows) + " blocks)");
            columns = Mathf.RoundToInt(GUILayout.HorizontalSlider(columns, MinColumns, MaxColumns));
            rows = Mathf.RoundToInt(GUILayout.HorizontalSlider(rows, MinRows, MaxRows));

            // Density, stiffness and the threshold on log scales.
            GUILayout.Label("Block density: " + blockDensity.ToString("0.00") + "   (" + (blockDensity * blockSize.x * blockSize.y * blockSize.z).ToString("0.00") + " kg per block)");
            blockDensity = Mathf.Clamp(Mathf.Pow(10f, GUILayout.HorizontalSlider(Mathf.Log10(blockDensity), -1f, 1f)), 0.1f, 10f);

            rigidJoints = GUILayout.Toggle(rigidJoints, "Rigid joints");
            GUILayout.Label("Joint stiffness: " + (rigidJoints ? "rigid (the slider is not used)" : jointStiffness.ToString("0.0e+0")));
            jointStiffness = Mathf.Clamp(Mathf.Pow(10f, GUILayout.HorizontalSlider(Mathf.Log10(jointStiffness), 2f, 8f)), 1e2f, 1e8f);

            GUILayout.Label("Fracture threshold: " + fractureThreshold.ToString(fractureThreshold < 10f ? "0.0" : "0"));
            fractureThreshold = Mathf.Clamp(Mathf.Pow(10f, GUILayout.HorizontalSlider(Mathf.Log10(fractureThreshold), 0f, 4f)), 1f, 1e4f);

            GUILayout.BeginHorizontal();
            if (GUILayout.Button(paused ? "Play" : "Pause")) paused = !paused;
            if (GUILayout.Button("Step")) { paused = true; requestStep = true; }
            if (GUILayout.Button("Clear shots")) requestClearShots = true;
            if (GUILayout.Button("Reset")) requestReset = true;
            GUILayout.EndHorizontal();

            GUILayout.Space(4f);
            GUILayout.Label("bodies " + (staticVisuals.Count + shots.Count) + "   shots " + shots.Count + "/" + maxShots);
            GUILayout.Label("joints broken " + (constraints != null ? constraints.BrokenCount : 0) + " / " + jointCount);
            GUILayout.Label("step " + stepMs.ToString("0.00") + " ms (CPU side)   frame " + frameMs.ToString("0.0") + " ms");
            if (behindRealTime)
                GUILayout.Label("running slower than real time");

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

        static Shader FindShader(params string[] names)
        {
            for (int i = 0; i < names.Length; i++)
            {
                Shader shader = Shader.Find(names[i]);
                if (shader != null)
                    return shader;
            }
            return null;
        }
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
