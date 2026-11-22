using System.Collections.Generic;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace McKrill.Avbd.Samples
{
    /// <summary>
    /// Spawns the objects of every voxel file in the <c>JsonMeshImportDemo</c> folder under <c>Resources</c> (JSON written by <c>Tools/Voxelize/voxelize_stl.py</c> from an STL mesh) on the GPU solver. Every voxel is a box
    /// body and every joint of the file is a ball-socket joint with a hard angle lock between two neighboring boxes, at the middle between their centers, that
    /// fractures like the joints of the GPU Fracture Wall Demo. The objects stand on the ground 5 m apart (center to center, along x); you shoot boxes at them
    /// or blow them up.
    ///
    /// The files are not listed anywhere: every text asset in a <c>Resources/JsonMeshImportDemo</c> folder is spawned, in the order of their names. Reset reads them again.
    ///
    /// The simulation runs entirely through the package's public API (<see cref="AvbdBodyBuffers"/>, <see cref="AvbdConstraintBuffers"/>,
    /// <see cref="AvbdSolver.Step"/>), and drawing is one instanced draw call that reads the poses from the solver's buffer, as in the Fracture Wall Demo.
    ///
    /// With <c>smoothSurface</c> on, the blocks on the surface of an object are drawn as the pieces of the smooth surface that the voxelizer baked for them (a file
    /// written with <c>--smooth</c>): one mesh for all pieces, each moved rigidly with its block by the vertex shader, in one more draw call. Intact, the object
    /// looks smooth. A block that breaks off still is a block: its curved outside and its flat cut faces. The physics does not change, the bodies are the same boxes.
    /// The scene is y-up, so solver positions and rotations are used as Unity world positions and rotations. Input is read through IMGUI events, so it works
    /// with the Input Manager and with the Input System package.
    /// </summary>
    [AddComponentMenu("McKrill AVBD/JSON Mesh Import Demo")]
    [DisallowMultipleComponent]
    public sealed class JsonMeshImportDemo : MonoBehaviour
    {
        /// <summary>
        /// The folder, inside any <c>Resources</c> folder of the project, whose files are spawned: one object per text asset (<c>.json</c>), in the order of their names.
        /// The sample ships a sphere, a torus and a cube in it; add a file there and it is spawned the next time the demo builds the scene (Play or Reset).
        /// </summary>
        const string VoxelFolder = "JsonMeshImportDemo";

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

        // One vertex of the mesh of all chunks (mirrors the vertex shader of AvbdBlockChunks.shader): position and normal relative to the block, and the number
        // of the block's record in the chunk instance buffer.
        struct ChunkVertex
        {
            public float3 position;
            public float3 normal;
            public uint block;
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
        [SerializeField] bool smoothSurface = true;
        [Tooltip("Mass of one voxel in kg.")]
        [SerializeField, Min(0.001f)] float voxelMass = 0.25f;
        [SerializeField] bool rigidJoints = true;
        [Tooltip("Stiffness of the joints' linear part when they are not rigid. The angle lock is always rigid: only a rigid angle lock stores the multiplier that fracture tests.")]
        [SerializeField, Min(1f)] float jointStiffness = 5e3f;
        [Tooltip("A joint breaks when the norm of its angular multiplier is above this.")]
        [SerializeField, Min(0.01f)] float fractureThreshold = 330f;

        [Header("Plastic joints (applied on Reset)")]
        [SerializeField] bool plasticJoints = true;
        [Tooltip("A plastic joint yields when the norm of its angular multiplier is above this. Keep it below the fracture threshold: a joint breaks before it can yield otherwise.")]
        [SerializeField, Min(0.1f)] float yieldThreshold = 110f;
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
        [Tooltip("The solver's constants (AvbdSolverSettings). They are in the panel at the right of the screen and take effect at the next step, no Reset needed. The defaults are the package's, except the lowest penalty (2000, as the demo always used).")]
        [SerializeField, Range(15f, 240f)] float stepsPerSecond = 60f;
        [SerializeField] Vector3 gravity = new Vector3(0f, -10f, 0f);
        [SerializeField, Range(1, 30)] int iterations = 10;
        [SerializeField, Range(0f, 1f)] float alpha = 0.925f;
        [SerializeField, Min(0f)] float betaLin = 1e4f;
        [SerializeField, Min(0f)] float betaAng = 100f;
        [SerializeField, Range(0f, 1f)] float gamma = 0.999f;
        [SerializeField, Min(1e-3f)] float penaltyMin = 2000f;
        [SerializeField, Min(1e-3f)] float penaltyMax = 1e10f;
        [Tooltip("The most steps run in one frame; a frame that needs more runs slower than real time.")]
        [SerializeField, Range(1, 4)] int maxStepsPerFrame = 2;

        [Header("Throw")]
        [Tooltip("Speed in meters per second of the voxels that Throw lets fly.")]
        [SerializeField, Range(5f, 200f)] float throwSpeed = 30f;

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
        [SerializeField, Range(1f, 50f)] float explosionRadius = 5f;
        [SerializeField, Range(0.1f, 1000f)] float explosionForce = 10f;
        [SerializeField, Range(0f, 20f)] float explosionDepth = 0.5f;

        [Header("Grab")]
        [SerializeField, Range(0.1f, 10f)] float grabRadius = 1.5f;
        [SerializeField, Range(10f, 2000f)] float grabForce = 100f;
        [Tooltip("Damping ratio of the velocity of each grabbed block. 0 is a spring that swings, 1 is critical damping.")]
        [SerializeField, Range(0f, 3f)] float grabDamping = 1f;
        [Tooltip("How much the force grows with the distance of the grabbed blocks from the center of the screen, per meter: the stiffness is Force times (1 + gain times the distance in meters of the blocks from the line of sight through the held point). 0 is a constant stiffness.")]
        [SerializeField, Range(0f, 5f)] float grabOffsetGain = 0.5f;

        [Tooltip("Off: the grabbed blocks are pulled with a spring force (Force, Damping ratio and Offset gain). On: each block is placed directly, every frame, at its own place relative to the held point.")]
        [SerializeField] bool grabDirect = false;

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
            Throw,
            Grab,
        }

        // Every body the demo spawns is in DemoGroup. The group mask of a new body has all bits set, which would put it in every group, so the bodies get
        // this one bit and the grab (a spring force field, see Grab below) labels the bodies it holds with GrabGroup and nothing else.
        const uint DemoGroup = 1u;
        const uint GrabGroup = 2u;

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
        readonly List<string> objectLines = new List<string>();                  // one line per file of the folder for the panel: what was read, or why not
        Material boxMaterial;
        Material chunkMaterial;
        Mesh cubeMesh;
        // The smooth pieces: one record (slot, color) for each block that is drawn as a chunk, and one mesh of all the chunks.
        GraphicsBuffer chunkInstanceBuffer;
        InstanceData[] chunkInstances;      // what the buffer holds now (the selection changes colors)
        InstanceData[] chunkBaseInstances;  // the colors without a selection
        int[] chunkBlockOfSlot = new int[0]; // body slot -> number of its record in the chunk buffer, or -1
        Mesh chunkMesh;
        int chunkBlockCount;
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
        int requestMode = -1;          // the click mode asked for in the panel, or -1
        bool requestGrabStart;
        bool requestThrowClick;
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
        // them and the rest of the object stretch and break. Letting go removes the field and the labels.
        // With grabDirect no force field is made: once a frame, before the steps, every labelled body is placed on its own (AvbdGroupMover.Hold, on the GPU) at its offset from
        // the held point (remembered when the grab began) plus the held point, and gets the velocity of the held point. The place is absolute, so what the steps did to a body
        // (gravity, contacts, the joints to the rest of the object) is undone every frame: the group does not sink and keeps its arrangement, and it follows the camera. Let
        // go, the blocks keep the velocity of the point and fly on. Their mass and moments of inertia are not touched.
        AvbdGroupLabeler labeler;
        AvbdGroupMover mover;
        int requestGrabDirect = -1;    // 1 or 0 to switch the grab style, or -1 (applied in Update, like the mode)
        bool grabIsDirect;             // the style of the grab that is active (the setting can change while it is held: it ends the grab)
        float3 grabLastTarget;         // direct grab: where the held point was at the last frame
        bool grabFirstMove;            // direct grab: the next UpdateGrab only begins the hold (remembers where each block is relative to the held point)
        bool grabHeld;                 // the button is down (set in OnGUI)
        bool grabActive;               // the bodies are labelled and the field exists
        int grabToken;                 // changes at every start and end, so that the answer of a ray cast of a grab that is over is dropped
        float grabDistance;            // meters from the camera to the point that is held
        ForceFieldHandle grabField;
        string grabStatus = "";

        // Throw: in Throw mode the first left click casts a ray on the GPU (like an explosion) and selects the voxel it hits and every voxel that is joined to
        // it by a joint that has not fractured (the whole object while it is whole, one piece of it after it broke); the selection is drawn in SelectedColor. The
        // next click puts the selection in front of the camera, the way a shot is placed, with the velocity of the Throw speed. Which joints are intact is read
        // back once, at the first click (AvbdConstraintBuffers.ReadIntactJointLinks).
        static readonly Color SelectedColor = new Color(0.2f, 0.95f, 0.35f);
        int[] staticIndexOfSlot = new int[0];            // body slot -> index into the static instances, or -1
        readonly List<BodyHandle> staticHandleList = new List<BodyHandle>(); // the body of every static instance (the ground, the voxels)
        readonly List<float3> staticSizeList = new List<float3>();           // the size of the box of every static instance (the cube instance has size 0 for a block drawn as a chunk)
        readonly List<bool> staticImmobileList = new List<bool>();           // true for a static instance without mass (the ground)
        readonly List<int> selectedStatic = new List<int>(); // indices of the selected voxels
        BodyHandle selectedShot;                         // a selected shot (shots have no joints: it is selected alone)
        bool selectionAnchored;                          // the selection is joined to an immobile body (not with this demo's files; kept for the shared code)
        bool castingSelection;                           // the ray of a selection is on its way
        string throwStatus = "";

        Rect panelScreenRect;
        Rect objectsPanelScreenRect;   // the panel of the objects, top right
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

            // Optional: without the chunk shader (or where it is not supported) every block is a cube.
            Shader chunkShader = AvbdShaderSet.Load()?.BlockChunks;
            if (chunkShader != null && chunkShader.isSupported)
                chunkMaterial = new Material(chunkShader) { hideFlags = HideFlags.HideAndDontSave };
            else
                Debug.LogWarning("JSON Mesh Import Demo: the package's block chunk shader is missing or not supported on this device, so every block is drawn as a cube.");

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
            if (chunkMaterial != null) Destroy(chunkMaterial);
            chunkMaterial = null;

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
                requestThrowClick = false;
                requestClearShots = false;
            }
            if (requestMode >= 0)
            {
                if ((ClickMode)requestMode != ClickMode.Grab)
                    EndGrab();
                if ((ClickMode)requestMode != ClickMode.Throw)
                {
                    ClearSelection();
                    castingSelection = false;
                    throwStatus = "";
                }
                clickMode = (ClickMode)requestMode;
                requestMode = -1;
            }
            if (selectedShot.IsValid && !bodies.IsAlive(selectedShot))
                ClearSelection(); // the shot was removed
            if (requestThrowClick)
            {
                if (clickMode == ClickMode.Throw)
                    ThrowClick();
                requestThrowClick = false;
            }
            if (requestGrabDirect >= 0)
            {
                EndGrab(); // a grab keeps the style it started with; the next click uses the new one
                grabToken++; // and the answer of a ray cast that is on its way, made for the old style, is dropped
                grabDirect = requestGrabDirect == 1;
                requestGrabDirect = -1;
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

            MoveCamera();
            UpdateGrab();
            tunnelRisk = shotSpeed * solver.Settings.dt > shotSize;

            // The settings (the panel at the right) take effect at the next step.
            solver.Settings = CurrentSettings();

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

        // Reads every file of the Resources folder VoxelFolder, in the order of their names (Resources.LoadAll does not promise an order). A file that cannot be read
        // is reported (log and panel) and left out; the rest is still spawned.
        void LoadObjects()
        {
            objects.Clear();
            objectLines.Clear();
            voxelTotal = 0;

            TextAsset[] files = Resources.LoadAll<TextAsset>(VoxelFolder);
            System.Array.Sort(files, (a, b) => string.CompareOrdinal(a.name, b.name));
            if (files.Length == 0)
            {
                Debug.LogError("JSON Mesh Import Demo: there are no voxel files in a Resources/" + VoxelFolder + " folder, so there is nothing to spawn.");
                objectLines.Add("no voxel files in Resources/" + VoxelFolder);
            }

            foreach (TextAsset file in files)
            {
                string name = DisplayName(file.name);
                if (!VoxelMeshData.TryParse(file.text, out VoxelMeshData data, out string error))
                {
                    Debug.LogError("JSON Mesh Import Demo: \"" + file.name + "\" in Resources/" + VoxelFolder + " was not loaded: " + error);
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
                    + item.extent.x.ToString("0.0") + " x " + item.extent.y.ToString("0.0") + " x " + item.extent.z.ToString("0.0") + " m"
                    + (data.HasChunks ? ", " + data.chunkCount + " smooth blocks" : ", no smooth surface in the file"));
                if (item.extent.x > ObjectSpacing || item.extent.z > ObjectSpacing)
                    Debug.LogWarning("JSON Mesh Import Demo: \"" + name + "\" is wider than the " + ObjectSpacing + " m between the objects and will overlap its neighbor.");
            }

            for (int i = 0; i < objects.Count; i++)
                objects[i].offset.x += (i - (objects.Count - 1) * 0.5f) * ObjectSpacing;
        }

        // The name shown in the panel: the asset name (Unity leaves out the ".json" of the file) without a ".voxels" from "engine.voxels.json".
        static string DisplayName(string assetName)
        {
            string name = assetName;
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

            AvbdSolverSettings settings = CurrentSettings();
            // Manifold slots: a voxel touches up to 26 others, but at rest only the joined ones (which make no manifold) and a few more; a shot can lie on or
            // inside a dozen. A pair that finds no slot just gets no contacts until one is free.
            int manifoldSlots = Mathf.Min(voxelTotal * 8 + maxShots * 16 + 256, AvbdConstraintBuffers.MaxContactManifolds);
            // The plastic array only exists if the joints can be plastic: without it the solver runs the joint kernels with no plastic code at all.
            int jointCapacity = jointsInFiles + 8;
            constraints = new AvbdConstraintBuffers(bodies, jointCapacity, null, manifoldSlots, (plasticJoints || (linearYield && rigidJoints)) ? jointCapacity : 0);
            solver = new AvbdSolver(bodies, settings, null, constraints);
            raycaster = new AvbdRaycaster(bodies);
            labeler = new AvbdGroupLabeler(bodies);
            mover = new AvbdGroupMover(bodies);
            simulationVersion++;

            // The ground and every voxel go in one batch (the ground is element 0, then the voxels object by object, in the order of the file), so that the bodies
            // can be placed in Z-curve order. The handles come back in the order of the list; the joints and the instances use them.
            var descriptors = new List<BodyDescriptor>(1 + voxelTotal);
            var instanceSizes = new List<float3>(1 + voxelTotal);
            var instanceColors = new List<Color>(1 + voxelTotal);
            staticHandleList.Clear();
            staticImmobileList.Clear();
            var positions = new float3[voxelTotal]; // world position of every voxel, in the same order

            // Ground: a static slab whose top face is the plane y = 0.
            var ground = new float3(GroundHalfExtent * 2f, 1f, GroundHalfExtent * 2f);
            descriptors.Add(DemoBox(ground, 0f, new float3(0f, -0.5f, 0f)));
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
                    descriptors.Add(DemoBox(size, density, position));
                    instanceSizes.Add(size);
                    instanceColors.Add(VoxelColor(i, data.voxels[3 * n], data.voxels[3 * n + 1], data.voxels[3 * n + 2]));
                }
                first += data.voxelCount;
            }
            var spawned = new BodyHandle[descriptors.Count];
            bodies.SpawnBatch(descriptors, spawned, zCurveOrder);

            // The blocks that are drawn as chunks (those of a file with chunks, with the smooth surface on) are not cubes in the instance list: they get a record in
            // the chunk instance buffer instead. Index in the list of descriptors: 1 + the voxel's number in the whole scene.
            bool[] drawnAsChunk = new bool[spawned.Length];
            bool chunksWanted = smoothSurface && chunkMaterial != null;
            if (chunksWanted)
            {
                first = 0;
                foreach (VoxelObject o in objects)
                {
                    if (o.data.HasChunks)
                        for (int c = 0; c < o.data.chunkCount; c++)
                            drawnAsChunk[1 + first + o.data.chunkVoxels[c]] = spawned[1 + first + o.data.chunkVoxels[c]].IsValid;
                    first += o.data.voxelCount;
                }
            }
            for (int i = 0; i < spawned.Length; i++)
                AddStaticInstance(spawned[i], instanceSizes[i], instanceColors[i], i == 0, drawnAsChunk[i]); // element 0 is the ground
            if (chunksWanted)
                CreateChunkMesh(spawned, drawnAsChunk, instanceColors);

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

        // The mesh of all chunks, and the record of each (the body slot and the color of its block). The chunks of a file are stored one after the other, so the
        // vertices of chunk c of an object are the file's vertices chunkVertexStart[c] up to chunkVertexStart[c + 1] - 1, and the vertex number of a triangle's
        // corner is counted from the first of them. Positions and normals are relative to the block: the vertex shader moves them with the block's pose.
        void CreateChunkMesh(BodyHandle[] spawned, bool[] drawnAsChunk, List<Color> colors)
        {
            var blocks = new List<InstanceData>();
            var vertices = new List<ChunkVertex>();
            var indices = new List<int>();
            int first = 0;
            foreach (VoxelObject o in objects)
            {
                VoxelMeshData data = o.data;
                for (int c = 0; data.HasChunks && c < data.chunkCount; c++)
                {
                    int descriptorIndex = 1 + first + data.chunkVoxels[c];
                    if (!drawnAsChunk[descriptorIndex])
                        continue;
                    uint block = (uint)blocks.Count;
                    blocks.Add(MakeInstance(spawned[descriptorIndex], float3.zero, colors[descriptorIndex]));

                    int vertexBase = vertices.Count;
                    for (int v = data.chunkVertexStart[c]; v < data.chunkVertexStart[c + 1]; v++)
                        vertices.Add(new ChunkVertex
                        {
                            position = new float3(data.chunkPositions[3 * v], data.chunkPositions[3 * v + 1], data.chunkPositions[3 * v + 2]),
                            normal = new float3(data.chunkNormals[3 * v], data.chunkNormals[3 * v + 1], data.chunkNormals[3 * v + 2]),
                            block = block,
                        });
                    for (int m = 3 * data.chunkTriangleStart[c]; m < 3 * data.chunkTriangleStart[c + 1]; m++)
                        indices.Add(vertexBase + data.chunkTriangles[m]);
                }
                first += data.voxelCount;
            }

            chunkBlockCount = blocks.Count;
            if (chunkBlockCount == 0)
                return;

            chunkMesh = new Mesh { name = "AVBD block chunks", indexFormat = IndexFormat.UInt32, hideFlags = HideFlags.HideAndDontSave };
            chunkMesh.SetVertexBufferParams(vertices.Count,
                new VertexAttributeDescriptor(VertexAttribute.Position, VertexAttributeFormat.Float32, 3),
                new VertexAttributeDescriptor(VertexAttribute.Normal, VertexAttributeFormat.Float32, 3),
                new VertexAttributeDescriptor(VertexAttribute.TexCoord0, VertexAttributeFormat.UInt32, 1));
            chunkMesh.SetVertexBufferData(vertices, 0, 0, vertices.Count, 0, MeshUpdateFlags.DontRecalculateBounds);
            chunkMesh.SetIndexBufferParams(indices.Count, IndexFormat.UInt32);
            chunkMesh.SetIndexBufferData(indices, 0, 0, indices.Count, MeshUpdateFlags.DontRecalculateBounds);
            chunkMesh.SetSubMesh(0, new SubMeshDescriptor(0, indices.Count), MeshUpdateFlags.DontRecalculateBounds);
            chunkMesh.bounds = new Bounds(Vector3.zero, new Vector3(100000f, 100000f, 100000f)); // the blocks move on the GPU

            chunkBaseInstances = blocks.ToArray();
            chunkInstances = blocks.ToArray();
            chunkBlockOfSlot = new int[bodies.Capacity];
            for (int i = 0; i < chunkBlockOfSlot.Length; i++)
                chunkBlockOfSlot[i] = -1;
            for (int i = 0; i < chunkBaseInstances.Length; i++)
                chunkBlockOfSlot[chunkBaseInstances[i].slot] = i;
            chunkInstanceBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, blocks.Count, UnsafeUtility.SizeOf<InstanceData>());
            chunkInstanceBuffer.SetData(chunkInstances);
            chunkMaterial.SetBuffer("_Poses", bodies.State);
            chunkMaterial.SetBuffer("_Instances", chunkInstanceBuffer);
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

        // A block that is drawn as a chunk stays in the list of static instances (the selection of Throw works on it), but its cube has size 0: the cube shader
        // draws nothing for it. Its real size is in staticSizeList.
        void AddStaticInstance(BodyHandle handle, float3 size, Color color, bool immobile = false, bool drawnAsChunk = false)
        {
            if (!handle.IsValid)
                return;
            staticInstances.Add(MakeInstance(handle, drawnAsChunk ? float3.zero : size, color));
            staticSizeList.Add(size);
            staticHandleList.Add(handle);
            staticImmobileList.Add(immobile);
        }

        static InstanceData MakeInstance(BodyHandle handle, float3 size, Color color)
        {
            // The shader outputs the color as it is, so a project in the linear color space needs the linear value.
            if (QualitySettings.activeColorSpace == ColorSpace.Linear)
                color = color.linear;
            return new InstanceData { slot = (uint)handle.Slot, sizeX = size.x, sizeY = size.y, sizeZ = size.z, color = ShaderColor(color) };
        }

        // The shader outputs the color as it is, so a project in the linear color space needs the linear value.
        static float4 ShaderColor(Color color)
        {
            if (QualitySettings.activeColorSpace == ColorSpace.Linear)
                color = color.linear;
            return new float4(color.r, color.g, color.b, 1f);
        }

        // The buffer holds the static instances (written now) and room for the most shots that can exist; the shots are written when they change.
        void CreateInstanceBuffer()
        {
            staticCount = staticInstances.Count;
            staticIndexOfSlot = new int[bodies.Capacity];
            for (int i = 0; i < staticIndexOfSlot.Length; i++)
                staticIndexOfSlot[i] = -1;
            for (int i = 0; i < staticCount; i++)
                staticIndexOfSlot[staticHandleList[i].Slot] = i;
            instances = new InstanceData[staticCount + maxShots];
            staticInstances.CopyTo(instances);
            instanceBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, instances.Length, UnsafeUtility.SizeOf<InstanceData>());
            instanceBuffer.SetData(instances, 0, 0, staticCount);
            shotsChanged = false;

            boxMaterial.SetBuffer("_Poses", bodies.State);
            boxMaterial.SetBuffer("_Instances", instanceBuffer);
        }

        // The solver's constants as the panel has them. A value that would not pass the solver's check is brought into range (the highest penalty is not below the lowest).
        AvbdSolverSettings CurrentSettings()
        {
            float lowest = Mathf.Max(penaltyMin, 1e-3f);
            return new AvbdSolverSettings
            {
                dt = 1f / Mathf.Max(stepsPerSecond, 1f),
                gravity = gravity,
                iterations = iterations,
                alpha = Mathf.Clamp01(alpha),
                betaLin = Mathf.Max(betaLin, 0f),
                betaAng = Mathf.Max(betaAng, 0f),
                gamma = Mathf.Clamp01(gamma),
                penaltyMin = lowest,
                penaltyMax = Mathf.Max(penaltyMax, lowest),
            };
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
            selectedStatic.Clear();
            selectedShot = BodyHandle.Invalid;
            selectionAnchored = false;
            castingSelection = false;
            throwStatus = "";
            staticHandleList.Clear();
            staticSizeList.Clear();
            staticImmobileList.Clear();
            staticIndexOfSlot = new int[0];
            shots.Clear();
            staticInstances.Clear();
            if (instanceBuffer != null) instanceBuffer.Release();
            instanceBuffer = null;
            instances = null;
            staticCount = 0;
            if (chunkInstanceBuffer != null) chunkInstanceBuffer.Release();
            chunkInstanceBuffer = null;
            chunkInstances = null;
            chunkBaseInstances = null;
            chunkBlockOfSlot = new int[0];
            if (chunkMesh != null) Destroy(chunkMesh);
            chunkMesh = null;
            chunkBlockCount = 0;
            grabActive = false;
            grabHeld = false;
            grabToken++;
            grabField = ForceFieldHandle.Invalid;
            grabStatus = "";
            grabIsDirect = false;
            if (labeler != null) labeler.Dispose();
            labeler = null;
            if (mover != null) mover.Dispose();
            mover = null;
            if (raycaster != null) raycaster.Dispose();
            raycaster = null;
            if (solver != null) solver.Dispose();
            if (constraints != null) constraints.Dispose();
            if (bodies != null) bodies.Dispose();
            solver = null;
            constraints = null;
            bodies = null;
        }

        // ---- Throw -------------------------------------------------------------------------------------

        // The click of Throw mode: select what the ray meets, or, with a selection, throw it.
        void ThrowClick()
        {
            if (castingSelection || cam == null)
                return;
            if (selectedStatic.Count > 0 || selectedShot.IsValid)
                ThrowSelection();
            else
                SelectUnderCrosshair();
        }

        void SelectUnderCrosshair()
        {
            float3 origin = cam.transform.position;
            float3 direction = cam.transform.forward;
            direction = math.normalizesafe(direction);
            int version = simulationVersion; // a Reset before the answer arrives makes it stale

            castingSelection = true;
            throwStatus = "Casting the ray...";
            raycaster.Cast(origin, direction, RayMaxDistance, (hit, distance) =>
            {
                if (version != simulationVersion || solver == null)
                    return;
                castingSelection = false;
                if (clickMode != ClickMode.Throw)
                    return;
                if (!hit)
                {
                    throwStatus = "Nothing selected: the ray from the camera hit nothing.";
                    return;
                }
                SelectAt(origin + direction * distance);
            });
        }

        static float BoxDistance(in BodyStateRecord state, float3 size, float3 point)
        {
            quaternion rotation = new quaternion(state.rotation);
            float3 local = math.mul(math.conjugate(rotation), point - state.position.xyz);
            return math.length(math.max(math.abs(local) - size * 0.5f, 0f));
        }

        // Selects the body whose box contains the point where the ray met the scene (the point is on its surface, so the nearest box within a small distance),
        // and, for a block, every block that can be reached from it through intact joints without passing through an immobile block.
        void SelectAt(float3 point)
        {
            var state = new BodyStateRecord[bodies.Capacity];
            bodies.State.GetData(state); // the poses of the last step; waits for the GPU once

            const float tolerance = 0.1f;
            float best = tolerance;
            int bestStatic = -1;
            BodyHandle bestShot = BodyHandle.Invalid;
            for (int i = 0; i < staticCount; i++)
            {
                float d = BoxDistance(state[staticHandleList[i].Slot], staticSizeList[i], point);
                if (d < best)
                {
                    best = d;
                    bestStatic = i;
                    bestShot = BodyHandle.Invalid;
                }
            }
            for (int i = 0; i < shots.Count; i++)
            {
                float d = BoxDistance(state[shots[i].handle.Slot], new float3(shots[i].size), point);
                if (d < best)
                {
                    best = d;
                    bestStatic = -1;
                    bestShot = shots[i].handle;
                }
            }

            ClearSelection();
            if (bestShot.IsValid)
            {
                selectedShot = bestShot;
                shotsChanged = true; // the shot is drawn in the color of the selection
                selectionAnchored = false;
                throwStatus = "Selected a shot.";
                return;
            }
            if (bestStatic < 0)
            {
                throwStatus = "Nothing selected: no body at the place the ray hit.";
                return;
            }
            if (staticImmobileList[bestStatic])
            {
                throwStatus = "Nothing selected: that body is immobile.";
                return;
            }

            // The joints that are intact now (read back from the GPU), as a graph over the body slots.
            var links = new List<int2>();
            constraints.ReadIntactJointLinks(links);
            int capacity = bodies.Capacity;
            var first = new int[capacity + 1];
            foreach (int2 link in links)
            {
                first[link.x + 1]++;
                first[link.y + 1]++;
            }
            for (int i = 0; i < capacity; i++)
                first[i + 1] += first[i];
            var fill = new int[capacity];
            var neighbors = new int[2 * links.Count];
            foreach (int2 link in links)
            {
                neighbors[first[link.x] + fill[link.x]++] = link.y;
                neighbors[first[link.y] + fill[link.y]++] = link.x;
            }

            var visited = new bool[capacity];
            var pending = new Stack<int>();
            int seed = staticHandleList[bestStatic].Slot;
            visited[seed] = true;
            pending.Push(seed);
            int heldByImmobile = 0; // joints from the selection to immobile blocks
            while (pending.Count > 0)
            {
                int slot = pending.Pop();
                int index = staticIndexOfSlot[slot];
                if (index >= 0)
                    selectedStatic.Add(index);
                for (int k = first[slot]; k < first[slot + 1]; k++)
                {
                    int other = neighbors[k];
                    int otherIndex = staticIndexOfSlot[other];
                    if (otherIndex >= 0 && staticImmobileList[otherIndex])
                    {
                        heldByImmobile++;
                        continue;
                    }
                    if (visited[other])
                        continue;
                    visited[other] = true;
                    pending.Push(other);
                }
            }

            selectionAnchored = heldByImmobile > 0;
            for (int i = 0; i < selectedStatic.Count; i++)
            {
                InstanceData instance = instances[selectedStatic[i]];
                instance.color = ShaderColor(SelectedColor);
                instances[selectedStatic[i]] = instance;
            }
            instanceBuffer.SetData(instances, 0, 0, staticCount);
            ColorChunks(true);

            throwStatus = "Selected " + selectedStatic.Count + (selectedStatic.Count == 1 ? " voxel." : " voxels.")
                + (selectionAnchored ? " " + heldByImmobile + " joints still hold them to the ground: they cannot be thrown until those break." : "");
        }

        // The chunks of the selected blocks get the color of the selection (or their own again): the cube of such a block is not drawn.
        void ColorChunks(bool selected)
        {
            if (chunkInstanceBuffer == null || chunkInstances == null)
                return;
            foreach (int index in selectedStatic)
            {
                int block = chunkBlockOfSlot[staticHandleList[index].Slot];
                if (block < 0)
                    continue;
                chunkInstances[block] = chunkBaseInstances[block];
                if (selected)
                    chunkInstances[block].color = ShaderColor(SelectedColor);
            }
            chunkInstanceBuffer.SetData(chunkInstances);
        }

        void ClearSelection()
        {
            if (selectedStatic.Count > 0 && instanceBuffer != null && instances != null)
            {
                for (int i = 0; i < selectedStatic.Count; i++)
                    instances[selectedStatic[i]] = staticInstances[selectedStatic[i]];
                instanceBuffer.SetData(instances, 0, 0, staticCount);
                ColorChunks(false);
            }
            selectedStatic.Clear();
            if (selectedShot.IsValid)
                shotsChanged = true;
            selectedShot = BodyHandle.Invalid;
            selectionAnchored = false;
        }

        // Puts the selection in front of the camera, the way a shot is placed (clear of the near plane: its farthest block is 0.6 m ahead), keeping the arrangement
        // and the rotation of every block, and gives each block the velocity of the throw. The bodies are set from the CPU: the poses were read just now.
        void ThrowSelection()
        {
            if (selectionAnchored)
            {
                throwStatus = "Cannot throw: the selection is still held to the ground by joints.";
                return;
            }

            var handles = new List<BodyHandle>();
            var sizes = new List<float>(); // the radius of the sphere around each box
            if (selectedShot.IsValid)
            {
                for (int i = 0; i < shots.Count; i++)
                    if (shots[i].handle == selectedShot)
                    {
                        handles.Add(selectedShot);
                        sizes.Add(shots[i].size * 0.87f);
                    }
            }
            else
            {
                foreach (int index in selectedStatic)
                {
                    handles.Add(staticHandleList[index]);
                    sizes.Add(0.5f * math.length(staticSizeList[index]));
                }
            }
            if (handles.Count == 0)
            {
                ClearSelection();
                throwStatus = "Nothing to throw.";
                return;
            }

            var state = new BodyStateRecord[bodies.Capacity];
            bodies.State.GetData(state);
            float3 centroid = float3.zero;
            foreach (BodyHandle h in handles)
                centroid += state[h.Slot].position.xyz;
            centroid /= handles.Count;
            float radius = 0f;
            for (int i = 0; i < handles.Count; i++)
                radius = math.max(radius, math.length(state[handles[i].Slot].position.xyz - centroid) + sizes[i]);

            float3 origin = cam.transform.position;
            float3 direction = cam.transform.forward;
            direction = math.normalizesafe(direction);
            float3 target = origin + direction * (radius + 0.6f);
            float3 velocity = direction * throwSpeed;
            for (int i = 0; i < handles.Count; i++)
            {
                BodyStateRecord s = state[handles[i].Slot];
                bodies.SetPose(handles[i], s.position.xyz - centroid + target, new quaternion(s.rotation));
                bodies.SetVelocity(handles[i], velocity, float3.zero);
            }
            bodies.ApplyEdits();

            throwStatus = "Threw " + handles.Count + (handles.Count == 1 ? " body" : " bodies") + " at " + throwSpeed.ToString("0") + " m/s.";
            ClearSelection();
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
            bool direct = grabDirect;

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
                if (direct)
                {
                    // No field: the next UpdateGrab begins the hold and takes the point the camera holds then as the start, so the blocks do not jump by the way the
                    // camera moved while the ray was on its way.
                    grabIsDirect = true;
                    grabFirstMove = true;
                    grabActive = true;
                    grabStatus = "";
                    return;
                }
                ForceFieldHandle field = solver.AddForceField(GrabSpring(origin, direction));
                if (!field.IsValid)
                {
                    labeler.ClearAll(GrabGroup);
                    grabStatus = "Not grabbed: no force field left.";
                    return;
                }
                grabField = field;
                grabIsDirect = false;
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
            if (!grabIsDirect)
            {
                solver.UpdateForceField(grabField, GrabSpring(origin, direction));
                return;
            }

            // Direct: the held point is where the camera looks, at the distance of the grab. The first frame remembers where every block is relative to it. Every later
            // frame puts every block at the point plus its own offset, with the velocity of the point (its move over the time of the frame): the blocks follow the camera
            // exactly, and a block that is let go keeps that velocity. Paused, they follow the camera at rest. A very short frame is taken as 4 ms.
            float3 target = origin + math.normalizesafe(direction) * grabDistance;
            if (grabFirstMove)
            {
                grabLastTarget = target;
                grabFirstMove = false;
                mover.BeginHold(GrabGroup, target);
                return;
            }
            float3 velocity = paused ? float3.zero : (target - grabLastTarget) / Mathf.Max(Time.deltaTime, 0.004f);
            grabLastTarget = target;
            mover.Hold(GrabGroup, target, velocity);
        }

        void EndGrab()
        {
            if (!grabActive)
                return;
            grabToken++;
            grabActive = false;
            if (solver != null && !grabIsDirect)
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
                    instances[staticCount + i] = MakeInstance(shots[i].handle, new float3(shots[i].size), shots[i].handle == selectedShot ? SelectedColor : ShotColor);
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

            // The smooth pieces of the blocks on the surface: the same light, one more draw call for all of them.
            if (chunkMesh != null)
            {
                chunkMaterial.SetVector("_LightDirection", toLight);
                chunkMaterial.SetColor("_LightColor", lightColor);
                chunkMaterial.SetColor("_AmbientColor", new Color(0.35f, 0.37f, 0.40f));
                renderParams.material = chunkMaterial;
                Graphics.RenderMeshPrimitives(renderParams, chunkMesh, 0, 1);
            }
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
            // The release of the button ends a grab wherever it happens, also over a panel, where a control may use the event (rawType is the original).
            if (e.rawType == EventType.MouseUp && e.button == 0)
                grabHeld = false;

            DrawObjectsPanel();
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
            else if (clickMode == ClickMode.Throw)
            {
                GUILayout.Label("Speed: " + throwSpeed.ToString("0") + " m/s");
                throwSpeed = GUILayout.HorizontalSlider(throwSpeed, 5f, 200f);
                GUILayout.Label(selectedStatic.Count > 0 || selectedShot.IsValid
                    ? "Click again to throw the selection from the middle of the screen."
                    : "Click a voxel to select it and every voxel joined to it.");
                GUILayout.Label(throwStatus);
            }
            else if (clickMode == ClickMode.Explode)
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
            else
            {
                GUILayout.Label("Radius: " + grabRadius.ToString("0.0") + " m   (blocks around the point the ray hits)");
                grabRadius = GUILayout.HorizontalSlider(grabRadius, 0.1f, 10f);

                // How the blocks are grabbed. The toggle only asks (the style changes in Update). The sliders of the force are always drawn and greyed out in the
                // direct style: a control that comes and goes while the panel is drawn is an IMGUI error.
                bool grabEnabled = GUI.enabled;
                if (GUILayout.Toggle(grabDirect, "Direct: move the blocks, no force (mass and inertia unchanged)") != grabDirect)
                    requestGrabDirect = grabDirect ? 0 : 1;
                GUI.enabled = grabEnabled && !grabDirect;

                // Stiffness on a log scale: 10 to 2000 N/m. Each block gets this times the distance of the group from the held point, in newtons.
                GUILayout.Label("Force: " + grabForce.ToString("0") + " N/m   (a pull of this many N per meter the blocks are from the held point)");
                grabForce = Mathf.Clamp(Mathf.Pow(10f, GUILayout.HorizontalSlider(Mathf.Log10(grabForce), 1f, Mathf.Log10(2000f))), 10f, 2000f);

                GUILayout.Label("Damping ratio: " + grabDamping.ToString("0.00") + "   (0 swings, 1 is critical)");
                grabDamping = GUILayout.HorizontalSlider(grabDamping, 0f, 3f);

                GUILayout.Label("Offset gain: " + grabOffsetGain.ToString("0.00") + " per m   (the force grows with the distance of the blocks from the center of the screen)");
                grabOffsetGain = GUILayout.HorizontalSlider(grabOffsetGain, 0f, 5f);

                GUI.enabled = grabEnabled;

                GUILayout.Label(grabActive ? "Holding at " + grabDistance.ToString("0.0") + " m from the camera" : grabStatus);
            }

            GUILayout.Space(4f);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button(paused ? "Play" : "Pause")) paused = !paused;
            if (GUILayout.Button("Step")) { paused = true; requestStep = true; }
            if (GUILayout.Button("Clear shots")) requestClearShots = true;
            if (GUILayout.Button("Reset")) requestReset = true;
            GUILayout.EndHorizontal();

            // Mass, stiffness and the threshold on log scales.
            smoothSurface = GUILayout.Toggle(smoothSurface, "Smooth surface (needs chunks in the file, see the objects panel)");

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
            GUILayout.Label("bodies " + (staticCount + chunkBlockCount + shots.Count) + "   shots " + shots.Count + "/" + maxShots);
            GUILayout.Label("joints broken " + (constraints != null ? constraints.BrokenCount : 0) + " / " + jointCount);
            GUILayout.Label("step " + stepMs.ToString("0.00") + " ms (CPU side)   frame " + frameMs.ToString("0.0") + " ms");
            if (behindRealTime)
                GUILayout.Label("running slower than real time");

            GUILayout.Space(4f);
            GUILayout.Label("Left click: shoot, explode, click to select and click again to throw, or hold to grab (the mode buttons)\nSpace: pause   R: reset\nRight drag: look   W A S D Q E: move (Shift: faster)\nMouse wheel: move forward and back");

            GUILayout.EndVertical();
            Rect used = GUILayoutUtility.GetLastRect();
            GUILayout.EndArea();

            if (Event.current.type == EventType.Repaint)
                panelScreenRect = new Rect(0f, 0f, (width + 2f * margin) * scale, (used.yMax + 2f * margin) * scale);

            GUI.matrix = oldMatrix;
        }

        // The objects and, below them, the solver's settings, in a panel of their own at the top right of the screen.
        void DrawObjectsPanel()
        {
            const float margin = 8f;
            const float width = 340f;
            float scale = Mathf.Clamp(Screen.height / 800f, 1f, 2.5f);

            Matrix4x4 oldMatrix = GUI.matrix;
            GUI.matrix = Matrix4x4.Scale(new Vector3(scale, scale, 1f));

            GUILayout.BeginArea(new Rect(Screen.width / scale - width - margin, margin, width, Screen.height / scale - 2f * margin));
            GUILayout.BeginVertical(GUI.skin.box);
            GUILayout.Label("Objects (Reset reads the files again)", boldLabel);
            foreach (string line in objectLines)
                GUILayout.Label(line);
            GUILayout.EndVertical();

            // Below it, the solver's settings (AvbdSolverSettings). The controls never come and go, and none of them needs a Reset.
            GUILayout.Space(4f);
            GUILayout.BeginVertical(GUI.skin.box);
            GUILayout.Label("AVBD solver settings (apply at the next step)", boldLabel);

            GUILayout.Label("Time step: " + (1f / stepsPerSecond * 1000f).ToString("0.0") + " ms   (" + stepsPerSecond.ToString("0") + " steps per second)");
            stepsPerSecond = Mathf.Round(GUILayout.HorizontalSlider(stepsPerSecond, 15f, 240f));

            GUILayout.Label("Iterations: " + iterations + "   (a primal and a dual pass each)");
            iterations = Mathf.RoundToInt(GUILayout.HorizontalSlider(iterations, 1, 30));

            GUILayout.Label("Gravity: (" + gravity.x.ToString("0.0") + ", " + gravity.y.ToString("0.0") + ", " + gravity.z.ToString("0.0") + ") m/s^2");
            gravity.x = GUILayout.HorizontalSlider(gravity.x, -30f, 30f);
            gravity.y = GUILayout.HorizontalSlider(gravity.y, -30f, 30f);
            gravity.z = GUILayout.HorizontalSlider(gravity.z, -30f, 30f);

            GUILayout.Label("Alpha: " + alpha.ToString("0.000") + "   (stabilization: 1 none, 0 full)");
            alpha = GUILayout.HorizontalSlider(alpha, 0f, 1f);

            betaLin = LogSlider("Beta linear", betaLin, 1f, 1e7f, "   (penalty ramping, linear rows)");
            betaAng = LogSlider("Beta angular", betaAng, 1f, 1e6f, "   (penalty ramping, angular rows)");

            GUILayout.Label("Gamma: " + gamma.ToString("0.0000") + "   (warm start decay)");
            gamma = GUILayout.HorizontalSlider(gamma, 0f, 1f);

            penaltyMin = LogSlider("Penalty min", penaltyMin, 1f, 1e8f, "   (lowest penalty of a row)");
            penaltyMax = LogSlider("Penalty max", penaltyMax, 1f, 1e12f, "   (highest penalty of a row)");

            GUILayout.Label("Max steps per frame: " + maxStepsPerFrame + "   (more is slower than real time)");
            maxStepsPerFrame = Mathf.RoundToInt(GUILayout.HorizontalSlider(maxStepsPerFrame, 1, 4));

            if (GUILayout.Button("Default solver settings"))
            {
                AvbdSolverSettings d = AvbdSolverSettings.Default;
                stepsPerSecond = 1f / d.dt;
                gravity = d.gravity;
                iterations = d.iterations;
                alpha = d.alpha;
                betaLin = d.betaLin;
                betaAng = d.betaAng;
                gamma = d.gamma;
                penaltyMin = 2000f; // the demo's own lowest penalty
                penaltyMax = d.penaltyMax;
                maxStepsPerFrame = 2;
            }
            GUILayout.EndVertical();

            Rect used = GUILayoutUtility.GetLastRect();
            GUILayout.EndArea();

            if (Event.current.type == EventType.Repaint)
                objectsPanelScreenRect = new Rect(Screen.width - (width + 2f * margin) * scale, 0f, (width + 2f * margin) * scale, (used.yMax + 2f * margin) * scale);

            GUI.matrix = oldMatrix;
        }

        // A slider on a log scale (the value in [min, max]) with its label.
        static float LogSlider(string name, float value, float min, float max, string note)
        {
            GUILayout.Label(name + ": " + value.ToString("G4") + note);
            float t = GUILayout.HorizontalSlider(Mathf.Log10(Mathf.Clamp(value, min, max)), Mathf.Log10(min), Mathf.Log10(max));
            return Mathf.Clamp(Mathf.Pow(10f, t), min, max);
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
            if (panelScreenRect.Contains(e.mousePosition) || objectsPanelScreenRect.Contains(e.mousePosition))
                return;

            if (e.type == EventType.MouseDown && e.button == 0)
            {
                if (clickMode == ClickMode.Explode)
                    requestExplode = true;
                else if (clickMode == ClickMode.Throw)
                    requestThrowClick = true;
                else if (clickMode == ClickMode.Shoot)
                    requestShoot = true;
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

        static readonly string[] ClickModeNames = { "Shot", "Explosion", "Throw", "Grab" };

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
