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

        /// <summary>Stage U: velocity update (BDF1) after the last iteration.</summary>
        public ComputeShader Velocity => velocity;

        /// <summary>Loads the shader set, or returns null if the asset is missing.</summary>
        public static AvbdShaderSet Load()
        {
            return Resources.Load<AvbdShaderSet>(ResourcePath);
        }
    }
}
