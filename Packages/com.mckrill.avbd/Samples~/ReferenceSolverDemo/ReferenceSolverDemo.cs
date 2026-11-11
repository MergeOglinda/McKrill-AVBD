using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Samples
{
    /// <summary>
    /// Runs the CPU reference solver live and draws it with one cube per body.
    ///
    /// This is a visualization of the reference (the test oracle), not of the GPU solver: it is
    /// managed code with an O(n^2) broadphase and one GameObject per body, so it is meant for looking
    /// at behavior, not for performance. The GPU package will not use GameObjects for bodies.
    ///
    /// The solver works in upstream's z-up frame. A child transform rotated -90 degrees about X turns
    /// that into Unity's y-up world, so solver positions and rotations are copied over unchanged.
    ///
    /// Input is read through IMGUI events only, so it works whether the project uses the old Input
    /// Manager or the Input System package.
    /// </summary>
    [AddComponentMenu("McKrill AVBD/Reference Solver Demo")]
    [DisallowMultipleComponent]
    public sealed class ReferenceSolverDemo : MonoBehaviour
    {
        public enum SceneKind { Ground, Stack, Pyramid, Rope, Spring }
        public enum TintMode { Plain, GraphColor, Speed }

        [Header("Simulation")]
        [SerializeField] SceneKind scene = SceneKind.Stack;
        [SerializeField] GaussSeidelMode sweep = GaussSeidelMode.Colored;
        [SerializeField, Range(1, 30)] int iterations = 10;
        [Tooltip("Boxes along the pyramid base. Upstream uses 16; the managed reference is happier with fewer.")]
        [SerializeField, Range(4, 16)] int pyramidSize = 10;
        [Tooltip("Upper bound on solver steps per rendered frame. If the solver cannot keep up, the simulation runs slower than real time instead of stalling.")]
        [SerializeField, Range(1, 4)] int maxStepsPerFrame = 2;

        [Header("View")]
        [SerializeField] TintMode tint = TintMode.GraphColor;
        [SerializeField] bool showContacts = true;
        [SerializeField] bool showJoints = true;
        [Tooltip("Orbit camera: left or right drag to rotate, mouse wheel to zoom. Takes over the main camera while playing.")]
        [SerializeField] bool controlCamera = true;

        static readonly string[] SceneNames = { "Ground", "Stack", "Pyramid", "Rope", "Spring" };
        static readonly string[] SweepNames = { "Sequential", "Colored" };
        static readonly string[] TintNames = { "Plain", "Graph color", "Speed" };

        ReferenceSolver solver;
        Transform root;
        readonly List<GameObject> bodyObjects = new List<GameObject>();
        readonly List<Material> bodyMaterials = new List<Material>();
        readonly List<LineRenderer> lines = new List<LineRenderer>();
        Material jointMaterial;
        Material springMaterial;
        Shader boxShader;
        Shader lineShader;

        Camera cam;
        bool createdCamera;
        Vector3 savedCameraPosition;
        Quaternion savedCameraRotation;
        GameObject createdLight;
        Vector3 orbitTarget;
        float orbitDistance = 20f;
        float yaw = 25f;
        float pitch = 18f;

        float accumulator;
        bool paused;
        bool behindRealTime;
        float stepMs;

        // Requests made from OnGUI. They are applied in Update so the IMGUI control count never
        // changes between the Layout and Repaint passes of one frame.
        bool requestStep;
        bool requestDrop;
        bool requestReload;
        int requestScene = -1;

        Rect panelScreenRect;
        GUIStyle boldLabel;
        readonly System.Random rng = new System.Random(12345);

        // ---- lifecycle ---------------------------------------------------------------------------------

        void OnEnable()
        {
            solver = new ReferenceSolver();

            root = new GameObject("AVBD Reference Simulation").transform;
            root.SetParent(transform, false);
            root.localRotation = Quaternion.Euler(-90f, 0f, 0f); // solver z-up -> Unity y-up

            boxShader = FindShader("Universal Render Pipeline/Lit", "Standard", "Sprites/Default", "Hidden/Internal-Colored");
            lineShader = FindShader("Universal Render Pipeline/Unlit", "Sprites/Default", "Hidden/Internal-Colored");
            jointMaterial = MakeMaterial(lineShader, new Color(0.2f, 0.9f, 1f));
            springMaterial = MakeMaterial(lineShader, new Color(1f, 0.6f, 0.15f));

            SetUpCameraAndLight();
            LoadScene(scene);
        }

        void OnDisable()
        {
            ClearBodyObjects();
            for (int i = 0; i < lines.Count; i++)
                if (lines[i] != null) Destroy(lines[i].gameObject);
            lines.Clear();
            if (jointMaterial != null) Destroy(jointMaterial);
            if (springMaterial != null) Destroy(springMaterial);
            if (root != null) Destroy(root.gameObject);

            if (createdLight != null) Destroy(createdLight);
            createdLight = null;
            if (cam != null)
            {
                if (createdCamera) Destroy(cam.gameObject);
                else if (controlCamera) cam.transform.SetPositionAndRotation(savedCameraPosition, savedCameraRotation);
            }
            cam = null;
            createdCamera = false;
            solver = null;
        }

        void Update()
        {
            if (solver == null)
                return;

            if (requestScene >= 0)
            {
                LoadScene((SceneKind)requestScene);
                requestScene = -1;
                requestReload = false;
            }
            if (requestReload)
            {
                LoadScene(scene);
                requestReload = false;
            }
            if (requestDrop)
            {
                DropBox();
                requestDrop = false;
            }

            solver.Params.mode = sweep;
            solver.Params.iterations = iterations;

            if (requestStep)
            {
                StepOnce();
                requestStep = false;
            }
            else if (!paused)
            {
                float dt = solver.Params.dt;
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
            Quaternion rot = Quaternion.Euler(pitch, yaw, 0f);
            cam.transform.SetPositionAndRotation(orbitTarget + rot * new Vector3(0f, 0f, -orbitDistance), rot);
        }

        // ---- scene management --------------------------------------------------------------------------

        void LoadScene(SceneKind kind)
        {
            scene = kind;
            ClearBodyObjects();

            switch (kind)
            {
                case SceneKind.Ground: AvbdScenes.Ground(solver); break;
                case SceneKind.Stack: AvbdScenes.Stack(solver); break;
                case SceneKind.Pyramid: AvbdScenes.Pyramid(solver, pyramidSize); break;
                case SceneKind.Rope: AvbdScenes.Rope(solver); break;
                case SceneKind.Spring: AvbdScenes.Spring(solver); break;
            }

            for (int i = 0; i < solver.Bodies.Count; i++)
                CreateBodyObject(solver.Bodies[i]);

            accumulator = 0f;
            stepMs = 0f;
            FrameCamera();
            SyncVisuals();
        }

        void ClearBodyObjects()
        {
            for (int i = 0; i < bodyObjects.Count; i++)
                if (bodyObjects[i] != null) Destroy(bodyObjects[i]);
            for (int i = 0; i < bodyMaterials.Count; i++)
                if (bodyMaterials[i] != null) Destroy(bodyMaterials[i]);
            bodyObjects.Clear();
            bodyMaterials.Clear();
        }

        void CreateBodyObject(AvbdBody body)
        {
            GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = "Body " + body.Index;
            Collider collider = go.GetComponent<Collider>();
            if (collider != null) Destroy(collider);

            Transform t = go.transform;
            t.SetParent(root, false);
            t.localScale = ToVector3(body.Size);

            Material material = MakeMaterial(boxShader, Color.white);
            go.GetComponent<Renderer>().sharedMaterial = material;

            bodyObjects.Add(go);
            bodyMaterials.Add(material);
        }

        void DropBox()
        {
            ComputeBounds(out float3 lo, out float3 hi);
            float3 center = (lo + hi) * 0.5f;
            float3 size = new float3(RandomRange(0.5f, 1.2f), RandomRange(0.5f, 1.2f), RandomRange(0.5f, 1.2f));
            float3 position = new float3(center.x + RandomRange(-1.5f, 1.5f), center.y + RandomRange(-1.5f, 1.5f), hi.z + 3f + size.z);

            AvbdBody body = solver.AddBody(size, 1f, 0.5f, position, float3.zero);
            body.PositionAng = quaternion.EulerXYZ(new float3(RandomRange(0f, 6.28f), RandomRange(0f, 6.28f), RandomRange(0f, 6.28f)));
            CreateBodyObject(body);
        }

        float RandomRange(float min, float max) => min + (float)rng.NextDouble() * (max - min);

        // Bounds of everything except the huge ground slabs, in solver space.
        void ComputeBounds(out float3 lo, out float3 hi)
        {
            lo = new float3(float.MaxValue);
            hi = new float3(float.MinValue);
            bool any = false;
            for (int i = 0; i < solver.Bodies.Count; i++)
            {
                AvbdBody b = solver.Bodies[i];
                if (math.cmax(b.Size) > 50f)
                    continue;
                lo = math.min(lo, b.PositionLin - b.Radius);
                hi = math.max(hi, b.PositionLin + b.Radius);
                any = true;
            }
            if (!any)
            {
                lo = new float3(-5f);
                hi = new float3(5f);
            }
        }

        void FrameCamera()
        {
            ComputeBounds(out float3 lo, out float3 hi);
            float3 center = (lo + hi) * 0.5f;
            orbitTarget = root.TransformPoint(ToVector3(center));
            float radius = 0.5f * math.length(hi - lo) + 1f;
            float fov = cam != null ? cam.fieldOfView : 60f;
            orbitDistance = radius / Mathf.Sin(0.5f * fov * Mathf.Deg2Rad) * 1.05f;
        }

        // ---- stepping and visuals ----------------------------------------------------------------------

        void StepOnce()
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            solver.Step();
            watch.Stop();
            float ms = (float)watch.Elapsed.TotalMilliseconds;
            stepMs = stepMs <= 0f ? ms : Mathf.Lerp(stepMs, ms, 0.1f);
        }

        void SyncVisuals()
        {
            IReadOnlyList<AvbdBody> bodies = solver.Bodies;
            int count = Mathf.Min(bodies.Count, bodyObjects.Count);
            for (int i = 0; i < count; i++)
            {
                AvbdBody body = bodies[i];
                Transform t = bodyObjects[i].transform;
                t.localPosition = ToVector3(body.PositionLin);
                float4 q = body.PositionAng.value;
                t.localRotation = new Quaternion(q.x, q.y, q.z, q.w);
                bodyMaterials[i].color = BodyColor(body, i);
            }
            SyncLines();
        }

        Color BodyColor(AvbdBody body, int index)
        {
            if (!body.IsDynamic)
                return new Color(0.33f, 0.36f, 0.40f);

            switch (tint)
            {
                case TintMode.GraphColor:
                    if (sweep == GaussSeidelMode.Colored && body.Color >= 0)
                        return Color.HSVToRGB((body.Color * 0.61803f) % 1f, 0.65f, 0.95f);
                    return new Color(0.75f, 0.75f, 0.78f);

                case TintMode.Speed:
                    float t = Mathf.Clamp01(math.length(body.VelocityLin) / 8f);
                    return Color.Lerp(new Color(0.25f, 0.55f, 1f), new Color(1f, 0.35f, 0.25f), t);

                default:
                    return Color.HSVToRGB(0.58f, 0.35f, 0.75f + 0.2f * ((index * 0.37f) % 1f));
            }
        }

        void SyncLines()
        {
            int used = 0;
            if (showJoints)
            {
                IReadOnlyList<AvbdForce> forces = solver.Forces;
                for (int i = 0; i < forces.Count; i++)
                {
                    float3 a, b;
                    Material material;
                    if (forces[i] is AvbdJoint joint)
                    {
                        if (joint.broken) continue;
                        a = JointAnchorA(joint);
                        b = AvbdQuat.Transform(joint.BodyB.PositionLin, joint.BodyB.PositionAng, joint.rB);
                        material = jointMaterial;
                    }
                    else if (forces[i] is AvbdSpring spring)
                    {
                        a = spring.BodyA != null ? AvbdQuat.Transform(spring.BodyA.PositionLin, spring.BodyA.PositionAng, spring.rA) : spring.rA;
                        b = spring.BodyB != null ? AvbdQuat.Transform(spring.BodyB.PositionLin, spring.BodyB.PositionAng, spring.rB) : spring.rB;
                        material = springMaterial;
                    }
                    else
                    {
                        continue;
                    }

                    LineRenderer line = GetLine(used++);
                    line.sharedMaterial = material;
                    line.enabled = true;
                    line.SetPosition(0, root.TransformPoint(ToVector3(a)));
                    line.SetPosition(1, root.TransformPoint(ToVector3(b)));
                }
            }
            for (int i = used; i < lines.Count; i++)
                lines[i].enabled = false;
        }

        static float3 JointAnchorA(AvbdJoint joint)
        {
            return joint.BodyA != null
                ? AvbdQuat.Transform(joint.BodyA.PositionLin, joint.BodyA.PositionAng, joint.rA)
                : joint.rA;
        }

        LineRenderer GetLine(int index)
        {
            while (lines.Count <= index)
            {
                var go = new GameObject("Line " + lines.Count);
                go.transform.SetParent(root, false);
                LineRenderer line = go.AddComponent<LineRenderer>();
                line.useWorldSpace = true;
                line.positionCount = 2;
                line.widthMultiplier = 0.08f;
                line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                line.receiveShadows = false;
                lines.Add(line);
            }
            return lines[index];
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
                cam.farClipPlane = 1000f;
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
                light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
            }
        }

        // ---- IMGUI -------------------------------------------------------------------------------------

        void OnGUI()
        {
            if (solver == null)
                return;

            if (boldLabel == null)
                boldLabel = new GUIStyle(GUI.skin.label) { fontStyle = FontStyle.Bold };

            Event e = Event.current;
            if (e.type == EventType.Repaint)
                DrawOverlay();

            DrawPanel();
            HandleInput(e);
        }

        void DrawOverlay()
        {
            if (cam == null || (!showContacts && !showJoints))
                return;

            IReadOnlyList<AvbdForce> forces = solver.Forces;
            Color old = GUI.color;
            for (int i = 0; i < forces.Count; i++)
            {
                if (showContacts && forces[i] is AvbdManifold manifold)
                {
                    for (int k = 0; k < manifold.NumContacts; k++)
                    {
                        AvbdManifold.Contact contact = manifold.Contacts[k];
                        float3 p = AvbdQuat.Transform(manifold.BodyA.PositionLin, manifold.BodyA.PositionAng, contact.rA);
                        DrawDot(p, contact.stick ? new Color(0.2f, 1f, 0.3f) : new Color(1f, 0.9f, 0.1f), 5f);
                    }
                }
                else if (showJoints && forces[i] is AvbdJoint joint && !joint.broken)
                {
                    DrawDot(JointAnchorA(joint), new Color(0.2f, 0.9f, 1f), 8f);
                }
            }
            GUI.color = old;
        }

        void DrawDot(float3 solverPosition, Color color, float size)
        {
            Vector3 screen = cam.WorldToScreenPoint(root.TransformPoint(ToVector3(solverPosition)));
            if (screen.z <= 0f)
                return;
            GUI.color = color;
            GUI.DrawTexture(new Rect(screen.x - size * 0.5f, Screen.height - screen.y - size * 0.5f, size, size), Texture2D.whiteTexture);
        }

        void DrawPanel()
        {
            const float margin = 8f;
            const float width = 290f;
            float scale = Mathf.Clamp(Screen.height / 800f, 1f, 2.5f);

            Matrix4x4 oldMatrix = GUI.matrix;
            GUI.matrix = Matrix4x4.Scale(new Vector3(scale, scale, 1f));

            GUILayout.BeginArea(new Rect(margin, margin, width, Screen.height / scale - 2f * margin));
            GUILayout.BeginVertical(GUI.skin.box);

            GUILayout.Label("AVBD reference solver (CPU)", boldLabel);

            int sceneIndex = GUILayout.SelectionGrid((int)scene, SceneNames, 3);
            if (sceneIndex != (int)scene)
                requestScene = sceneIndex;

            GUILayout.Label("Primal sweep order");
            sweep = (GaussSeidelMode)GUILayout.Toolbar((int)sweep, SweepNames);

            GUILayout.Label("Iterations: " + iterations);
            iterations = Mathf.RoundToInt(GUILayout.HorizontalSlider(iterations, 1, 30));

            GUILayout.Label("Tint");
            tint = (TintMode)GUILayout.Toolbar((int)tint, TintNames);

            showContacts = GUILayout.Toggle(showContacts, "Contact points (green = stick)");
            showJoints = GUILayout.Toggle(showJoints, "Joints and springs");

            GUILayout.BeginHorizontal();
            if (GUILayout.Button(paused ? "Play" : "Pause")) paused = !paused;
            if (GUILayout.Button("Step")) { paused = true; requestStep = true; }
            if (GUILayout.Button("Drop box")) requestDrop = true;
            if (GUILayout.Button("Reset")) requestReload = true;
            GUILayout.EndHorizontal();

            AvbdStepStats stats = solver.Stats;
            GUILayout.Space(4f);
            GUILayout.Label("bodies " + solver.Bodies.Count + "   manifolds " + stats.manifolds + "   contacts " + stats.contacts);
            GUILayout.Label("colors " + stats.colors + "   step " + stepMs.ToString("0.00") + " ms");
            if (stats.solveFailures > 0)
            {
                Color oldContent = GUI.contentColor;
                GUI.contentColor = new Color(1f, 0.45f, 0.4f);
                GUILayout.Label("solve failures " + stats.solveFailures + " (skipped primal updates)");
                GUI.contentColor = oldContent;
            }
            if (behindRealTime)
                GUILayout.Label("running slower than real time");

            GUILayout.Space(4f);
            GUILayout.Label("Space pause   N step   B drop box   R reset\n1-5 scene   C sweep order\nDrag to orbit, wheel to zoom");

            GUILayout.EndVertical();
            Rect used = GUILayoutUtility.GetLastRect();
            GUILayout.EndArea();

            if (Event.current.type == EventType.Repaint)
                panelScreenRect = new Rect(0f, 0f, (width + 2f * margin) * scale, (used.yMax + 2f * margin) * scale);

            GUI.matrix = oldMatrix;
        }

        void HandleInput(Event e)
        {
            if (e.type == EventType.KeyDown)
            {
                switch (e.keyCode)
                {
                    case KeyCode.Space: paused = !paused; e.Use(); break;
                    case KeyCode.N: paused = true; requestStep = true; e.Use(); break;
                    case KeyCode.B: requestDrop = true; e.Use(); break;
                    case KeyCode.R: requestReload = true; e.Use(); break;
                    case KeyCode.C:
                        sweep = sweep == GaussSeidelMode.Sequential ? GaussSeidelMode.Colored : GaussSeidelMode.Sequential;
                        e.Use();
                        break;
                    case KeyCode.Alpha1: requestScene = 0; e.Use(); break;
                    case KeyCode.Alpha2: requestScene = 1; e.Use(); break;
                    case KeyCode.Alpha3: requestScene = 2; e.Use(); break;
                    case KeyCode.Alpha4: requestScene = 3; e.Use(); break;
                    case KeyCode.Alpha5: requestScene = 4; e.Use(); break;
                }
                return;
            }

            if (!controlCamera || panelScreenRect.Contains(e.mousePosition))
                return;

            if (e.type == EventType.MouseDrag && (e.button == 0 || e.button == 1))
            {
                yaw += e.delta.x * 0.25f;
                pitch = Mathf.Clamp(pitch + e.delta.y * 0.25f, -5f, 85f);
                e.Use();
            }
            else if (e.type == EventType.ScrollWheel)
            {
                orbitDistance = Mathf.Clamp(orbitDistance * (1f + e.delta.y * 0.05f), 2f, 400f);
                e.Use();
            }
        }

        // ---- helpers -----------------------------------------------------------------------------------

        static Vector3 ToVector3(float3 v) => new Vector3(v.x, v.y, v.z);

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

        static Material MakeMaterial(Shader shader, Color color)
        {
            var material = new Material(shader);
            material.color = color;
            return material;
        }
    }

#if UNITY_EDITOR
    static class ReferenceSolverDemoMenu
    {
        [UnityEditor.MenuItem("GameObject/McKrill AVBD/Reference Solver Demo", false, 10)]
        static void Create(UnityEditor.MenuCommand command)
        {
            var go = new GameObject("AVBD Reference Solver Demo");
            UnityEditor.GameObjectUtility.SetParentAndAlign(go, command.context as GameObject);
            go.AddComponent<ReferenceSolverDemo>();
            UnityEditor.Undo.RegisterCreatedObjectUndo(go, "Create AVBD Reference Solver Demo");
            UnityEditor.Selection.activeObject = go;
        }
    }
#endif
}
