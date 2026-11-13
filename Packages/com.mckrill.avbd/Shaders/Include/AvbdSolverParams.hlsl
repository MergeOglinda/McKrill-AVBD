#ifndef MCKRILL_AVBD_SOLVER_PARAMS_INCLUDED
#define MCKRILL_AVBD_SOLVER_PARAMS_INCLUDED

// Global solver constants. Set by Runtime/Solver/SolverUniforms.cs, which must pack them the same way.
// Plain uniforms: Unity gathers them into one constant buffer per kernel.

float4 _SolverParams0; // x = dt, y = alpha, z = betaLin, w = betaAng
float4 _SolverParams1; // x = gamma, y = penaltyMin, z = penaltyMax, w unused
float4 _Gravity;       // xyz = gravity acceleration

#define AVBD_DT          (_SolverParams0.x)
#define AVBD_ALPHA       (_SolverParams0.y)
#define AVBD_BETA_LIN    (_SolverParams0.z)
#define AVBD_BETA_ANG    (_SolverParams0.w)
#define AVBD_GAMMA       (_SolverParams1.x)
#define AVBD_PENALTY_MIN (_SolverParams1.y)
#define AVBD_PENALTY_MAX (_SolverParams1.z)
#define AVBD_GRAVITY     (_Gravity.xyz)

#endif // MCKRILL_AVBD_SOLVER_PARAMS_INCLUDED
