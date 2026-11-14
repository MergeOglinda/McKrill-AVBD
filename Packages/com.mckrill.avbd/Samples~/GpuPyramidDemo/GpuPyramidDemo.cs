using System.Collections.Generic;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd.Samples
{
    /// <summary>
    /// A pyramid of boxes on the GPU solver, and a camera you can shoot boxes from.
    ///
    /// The simulation runs entirely through the package's public API: <see cref="AvbdBodyBuffers"/> holds the bodies,
    /// <see cref="AvbdSolver.Step"/> advances them. Collisions are part of that step (broadphase, narrowphase and contact solve),
    /// so this demo contains no collision code. Until those stages are inside <see cref="AvbdSolver.Step"/> the boxes fall through
    /// each other and the ground.
    ///
    /// Drawing is the simple part: each frame the poses are read back from the GPU and copied to one cube per body. That is a
    /// stall and a per-body GameObject, fine for a few hundred boxes and not what the package's own renderer will do.
    ///
    /// The scene is y-up (the package default), so solver positions and rotations are used as Unity world positions and rotations.
    /// Input is read through IMGUI events, so it works with the Input Manager and with the Input System package.
    /// </summary>
    [AddComponentMenu("McKrill AVBD/GPU Pyramid Demo")]
    [DisallowMultipleComponent]
    public sealed class GpuPyramidDemo : MonoBehaviour
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

        [Header("Pyramid")]
        [Tooltip("Layers of the pyramid. Layer k holds (layers - k)^2 blocks: 7 layers are 140 blocks. Applied on Reset.")]
        [SerializeField, Range(7, 12)] int layers = 7;
        [SerializeField] Vector3 blockSize = new Vector3(1f, 0.5f, 1f);
        [SerializeField, Min(0.01f)] float blockDensity = 1f;

        [Header("Solver")]
        [SerializeField, Range(1, 30)] int iterations = 10;
        [Tooltip("Upper bound on solver steps per rendered frame. If the GPU cannot keep up, the simulation runs slower than real time instead of stalling.")]
        [SerializeField, Range(1, 4)] int maxStepsPerFrame = 2;

        [Header("Shots")]
        [Tooltip("Speed of a shot in meters per second.")]
        [SerializeField, Range(5f, 80f)] float shotSpeed = 30f;
        [Tooltip("Edge length of the shot, a cube.")]
        [SerializeField, Range(0.2f, 2f)] float shotSize = 0.8f;
        [Tooltip("Mass of the shot. A pyramid block weighs blockSize volume times blockDensity (0.5 by default).")]
        [SerializeField, Range(0.1f, 200f)] float shotMass = 5f;
        [Tooltip("Shots alive at once. Shooting more removes the oldest.")]
        [SerializeField, Min(1)] int maxShots = 100;

        [Header("View")]
        [Tooltip("Fly camera: right drag to look, W A S D to move, Q and E for down and up, Shift for faster, mouse wheel to dolly.")]
        [SerializeField] bool controlCamera = true;
        [SerializeField, Min(0.1f)] float moveSpeed = 8f;
        [SerializeField, Min(0.01f)] float lookSensitivity = 0.2f;

        const float GroundHalfExtent = 100f;
        const float BlockGap = 0.01f; // between neighbors; the pyramid starts almost at rest

        AvbdBodyBuffers bodies;
        AvbdSolver solver;
        BodyStateRecord[] poses;

        readonly List<Visual> staticVisuals = new List<Visual>(); // ground and pyramid
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

            // Camera: in front of the pyramid, a little above the ground, looking at its middle. Reset leaves the camera where it is.
            camPosition = new Vector3(0f, 1.5f + 0.3f * layers * (blockSize.y + BlockGap), -(layers * (blockSize.x + BlockGap) * 0.5f + 12f));
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

        /// <summary>Number of blocks of a pyramid with the given layers: 1 + 4 + 9 + ... (layers^2).</summary>
        public static int BlockCount(int layers) => layers * (layers + 1) * (2 * layers + 1) / 6;

        void BuildSimulation()
        {
            DestroySimulation();

            int blocks = BlockCount(layers);
            int capacity = 1 + blocks + maxShots + 8;
            bodies = new AvbdBodyBuffers(capacity);
            if (bodies.State.stride != UnsafeUtility.SizeOf<BodyStateRecord>())
                throw new System.InvalidOperationException("The package's body state record changed; update BodyStateRecord in GpuPyramidDemo.");
            poses = new BodyStateRecord[capacity];

            AvbdSolverSettings settings = AvbdSolverSettings.Default; // 60 Hz, gravity (0, -10, 0): Unity's y-up
            settings.iterations = iterations;
            solver = new AvbdSolver(bodies, settings);

            // Ground: a static slab whose top face is the plane y = 0.
            var ground = new float3(GroundHalfExtent * 2f, 1f, GroundHalfExtent * 2f);
            AddStaticVisual(bodies.Spawn(BodyDescriptor.Box(ground, 0f, new float3(0f, -0.5f, 0f))), ground, new Color(0.30f, 0.33f, 0.36f));

            // Pyramid: layer k is a square grid of (layers - k) blocks per side, centered over the layer below, so every block rests
            // on four blocks of the layer under it.
            float3 size = new float3(blockSize.x, blockSize.y, blockSize.z);
            float stepX = size.x + BlockGap;
            float stepZ = size.z + BlockGap;
            float stepY = size.y + BlockGap;
            for (int k = 0; k < layers; k++)
            {
                int n = layers - k;
                float y = size.y * 0.5f + BlockGap + k * stepY;
                Color color = Color.HSVToRGB(0.08f + 0.5f * k / Mathf.Max(1, layers - 1), 0.55f, 0.95f);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                    {
                        var position = new float3((i - (n - 1) * 0.5f) * stepX, y, (j - (n - 1) * 0.5f) * stepZ);
                        AddStaticVisual(bodies.Spawn(BodyDescriptor.Box(size, blockDensity, position)), size, color);
                    }
            }
            bodies.ApplyEdits();

            accumulator = 0f;
            stepMs = 0f;
            SyncVisuals();
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
            if (bodies != null) bodies.Dispose();
            solver = null;
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

            GUILayout.Label("AVBD GPU solver: pyramid", boldLabel);

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

            GUILayout.Label("Pyramid layers: " + layers + "  (" + BlockCount(layers) + " blocks, applied on Reset)");
            layers = Mathf.RoundToInt(GUILayout.HorizontalSlider(layers, 7, 12));

            GUILayout.BeginHorizontal();
            if (GUILayout.Button(paused ? "Play" : "Pause")) paused = !paused;
            if (GUILayout.Button("Step")) { paused = true; requestStep = true; }
            if (GUILayout.Button("Clear shots")) requestClearShots = true;
            if (GUILayout.Button("Reset")) requestReset = true;
            GUILayout.EndHorizontal();

            GUILayout.Space(4f);
            GUILayout.Label("bodies " + (staticVisuals.Count + shots.Count) + "   shots " + shots.Count + "/" + maxShots);
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
    static class GpuPyramidDemoMenu
    {
        [UnityEditor.MenuItem("GameObject/McKrill AVBD/GPU Pyramid Demo", false, 11)]
        static void Create(UnityEditor.MenuCommand command)
        {
            var go = new GameObject("AVBD GPU Pyramid Demo");
            UnityEditor.GameObjectUtility.SetParentAndAlign(go, command.context as GameObject);
            go.AddComponent<GpuPyramidDemo>();
            UnityEditor.Undo.RegisterCreatedObjectUndo(go, "Create AVBD GPU Pyramid Demo");
            UnityEditor.Selection.activeObject = go;
        }
    }
#endif
}
