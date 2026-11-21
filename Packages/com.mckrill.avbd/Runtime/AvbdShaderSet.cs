using UnityEngine;

namespace McKrill.Avbd
{
    /// <summary>
    /// Holds references to every compute shader the package dispatches.
    /// The single asset lives in a package Resources folder, so the shaders are always
    /// included in player builds and need no per-call loading.
    /// The final shape of this set is decided by the GPU data layout task.
    /// </summary>
    public sealed class AvbdShaderSet : ScriptableObject
    {
        /// <summary>Path of the shader set asset, relative to any Resources folder.</summary>
        public const string ResourcePath = "McKrill.Avbd/AvbdShaderSet";

        [SerializeField] ComputeShader smoke;
        [SerializeField] ComputeShader bodyEdits;
        [SerializeField] ComputeShader predict;
        [SerializeField] ComputeShader primal;
        [SerializeField] ComputeShader dual;
        [SerializeField] ComputeShader velocity;
        [SerializeField] ComputeShader narrowphase;
        [SerializeField] ComputeShader coloring;
        [SerializeField] ComputeShader broadphase;
        [SerializeField] ComputeShader lists;
        [SerializeField] ComputeShader raycast;
        [SerializeField] ComputeShader groupLabel;
        [SerializeField] Shader instancedBoxes;
        [SerializeField] Shader blockChunks;

        /// <summary>Trivial kernel that writes a known pattern into a buffer. Proves the toolchain.</summary>
        public ComputeShader Smoke => smoke;

        /// <summary>Stage E: clears the per-body arrays and applies edit commands.</summary>
        public ComputeShader BodyEdits => bodyEdits;

        /// <summary>Stage P: inertial target and adaptive warm start.</summary>
        public ComputeShader Predict => predict;

        /// <summary>Stage V: primal update of each body (6x6 LDL^T solve), one dispatch per color.</summary>
        public ComputeShader Primal => primal;

        /// <summary>Stage D: constraint warm start and dual update (multipliers, penalties, fracture).</summary>
        public ComputeShader Dual => dual;

        /// <summary>Stage U: velocity update from the displacement over the step, and the snapshot for the next warm start.</summary>
        public ComputeShader Velocity => velocity;

        /// <summary>Stage N: box-box narrow phase over every pair of bodies, with the contact warm start.</summary>
        public ComputeShader Narrowphase => narrowphase;

        /// <summary>Stage C: graph coloring of the solved bodies and bucketing by color.</summary>
        public ComputeShader Coloring => coloring;

        /// <summary>Stage B: broadphase (candidate pairs of overlapping bounding spheres).</summary>
        public ComputeShader Broadphase => broadphase;

        /// <summary>Stage G: per-body constraint lists (CSR), built from the constraint records and the manifold pool.</summary>
        public ComputeShader Lists => lists;

        /// <summary>Ray against the boxes of every body (<see cref="AvbdRaycaster"/>).</summary>
        public ComputeShader Raycast => raycast;

        /// <summary>Changes the group mask of the bodies near a point (<see cref="AvbdGroupLabeler"/>).</summary>
        public ComputeShader GroupLabel => groupLabel;

        /// <summary>
        /// Draw shader for boxes read straight from the body state buffer, one instance per box (<c>Shaders/Rendering/AvbdInstancedBoxes.shader</c>).
        /// Not a compute shader: it is for rendering only, and the solver does not use it.
        /// </summary>
        public Shader InstancedBoxes => instancedBoxes;

        /// <summary>
        /// Draw shader for the smooth surface pieces of blocks (<c>Shaders/Rendering/AvbdBlockChunks.shader</c>): the vertices of every piece are relative to its block,
        /// and the shader moves them with the block's pose read from the body state buffer. Rendering only, like <see cref="InstancedBoxes"/>.
        /// </summary>
        public Shader BlockChunks => blockChunks;

        /// <summary>Loads the shader set, or returns null if the asset is missing.</summary>
        public static AvbdShaderSet Load()
        {
            return Resources.Load<AvbdShaderSet>(ResourcePath);
        }
    }
}
