using UnityEngine;

namespace McKrill.Avbd
{
    /// <summary>
    /// Sets the solver constants of a compute shader. The HLSL side is Shaders/Include/AvbdSolverParams.hlsl;
    /// the packing here must match it.
    /// </summary>
    internal static class SolverUniforms
    {
        static readonly int Params0Id = Shader.PropertyToID("_SolverParams0");
        static readonly int Params1Id = Shader.PropertyToID("_SolverParams1");
        static readonly int GravityId = Shader.PropertyToID("_Gravity");

        public static void Bind(ComputeShader shader, in AvbdSolverSettings s)
        {
            shader.SetVector(Params0Id, new Vector4(s.dt, s.alpha, s.betaLin, s.betaAng));
            shader.SetVector(Params1Id, new Vector4(s.gamma, s.penaltyMin, s.penaltyMax, 0f));
            shader.SetVector(GravityId, new Vector4(s.gravity.x, s.gravity.y, s.gravity.z, 0f));
        }
    }
}
