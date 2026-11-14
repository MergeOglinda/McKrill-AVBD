# Changelog

All notable changes to this package are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the package uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

### Added

- Package skeleton: manifest, assembly definitions, folder layout and documentation stubs.
- Smoke compute kernel and tests that prove assemblies compile, the shader is found and a dispatch works.
- CPU reference solver math layer (`McKrill.Avbd.Reference`): `Vec6`, `Mat6` with an LDL^T solver, and quaternion helpers (`AvbdQuat`), with Edit Mode tests.
- CPU reference solver (`ReferenceSolver`): a C# port of the upstream avbd-demo3d solver with boxes, contact manifolds (15-axis SAT, face clipping, edge contacts, feature-key warm start), ball-socket joints with optional angle lock and fracture, springs and IgnoreCollision. Includes the upstream demo scenes (`AvbdScenes`).
- Two primal-sweep orderings: `GaussSeidelMode.Sequential` (upstream's order) and `GaussSeidelMode.Colored` (greedy graph coloring, the order the GPU solver will run).
- Upstream-exact quaternion helpers (`AvbdQuat.Mul`, `Conjugate`, `Inverse`, `Rotate`, `Transform`, `DifferenceRaw`) and row/column-safe 3x3 helpers (`AvbdMat3`).
- Edit Mode tests: golden trajectories from the upstream C++ solver (`UpstreamParityTests`) and behavior, coloring, collision and fracture tests (`ReferenceSolverTests`).
- `THIRD_PARTY_NOTICES.md` with the upstream MIT attribution.
- Sample `Reference Solver Demo` (Package Manager > Samples): runs the CPU reference solver live with the upstream scenes, a sequential/colored toggle, graph-color tint, contact and joint overlays, and a drop-box button.
- Stage E of the GPU solver, for rigid boxes: per-body GPU arrays (`AvbdBodyBuffers`, 224 B per body), slot allocation with stale-handle detection, and an edit queue that merges spawn, remove and property edits into one command per slot. A compute kernel (`AvbdBodyEdits.compute`) clears the arrays and applies the commands. The HLSL and C# record definitions are checked against each other by a test.
- CPU oracle for stage E (`CpuBodyArrays`) and tests: layout, allocator, edit queue, and a GPU-versus-oracle comparison over random edit streams.
- Stage P of the GPU solver (predict): `AvbdSolverSettings`, `AvbdSolver` (`Step()` runs the stages that exist), the `AvbdPredict.compute` kernel, and the `CpuPredict` oracle. Tests check the oracle against the reference solver's predict phase and the kernel against the oracle.
- Stage V of the GPU solver (primal update): the `AvbdPrimal.compute` kernel solves each body's 6x6 system with an LDL^T factorization, one dispatch per graph color. Joints and springs are one 160 B `ConstraintRecord` (hard constraints are a flag, not an infinite stiffness); `AvbdConstraintBuffers` (`AddJoint`, `AddSpring`, `Rebuild`) owns them and builds the per-body lists on the CPU until stage G exists. `AvbdSolver.Step()` now also runs the primal sweeps.
- Stage D of the GPU solver (constraint variables): `AvbdDual.compute` with a `ConstraintInit` kernel (C0 at the start-of-step pose, warm start of multipliers and penalties) and a `Dual` kernel (multiplier and penalty update, fracture). `AvbdSolver.Step()` now runs predict, the warm start, and per iteration a primal sweep and the dual update.
- CPU oracle `CpuDual`, and `DualTests`: the oracles against the reference solver over a whole step, and the GPU solver against the oracles.
- Stage U of the GPU solver (velocity update): `AvbdVelocity.compute` derives each solved body's linear and angular velocity from its displacement over the step and snapshots the linear velocity for the next adaptive warm start. `AvbdSolver.Step()` now runs a whole step (edits, predict, warm start, iterations, velocity update). Oracles `CpuVelocity` and `CpuStep`; `VelocityTests` compares the oracle and the GPU solver with the reference solver over 60 steps on the rope and the spring scene, and checks a free fall against its closed form.
- `AvbdSolver` without constraint buffers now creates an empty set of its own, so free bodies get their primal solve; it is `IDisposable` and releases only what it created. The constraint lists are rebuilt when bodies are spawned, removed or change shape or kinematic state (`AvbdBodyBuffers.SolvedSetVersion`); before, a body spawned after the last `Rebuild()` was never solved.
- CPU oracles `CpuPrimal` and `CpuConstraintInit`, and tests: constraint layout, lists and coloring against the reference solver, the oracle against the reference solver after one colored iteration, and the kernel against the oracle.
- Design doc sections 3 and 5-10 (body arrays, capacity, slot allocation and edits, overflow policy, buffer owner, layout contract, tests).
- Stage C of the GPU solver (graph coloring): `AvbdColoring.compute` colors the solved bodies with Jones-Plassmann rounds and buckets them by color; `AvbdGraphColoring` runs it from `AvbdConstraintBuffers.Rebuild()`. The default `GraphColoringOrder.SlotOrder` gives exactly the reference solver's coloring; `GraphColoringOrder.Hashed` needs far fewer rounds on long chains. The graph color is stored in `Meta.state.x`. More than 64 colors throws. The CPU coloring in `ConstraintTopology` is now only the oracle (`CpuGraphColoring` adds the hashed order), and `GraphColoringTests` compares the GPU with it. Colors are recomputed on every rebuild; temporal coherence is not implemented.
- Sample `GPU Pyramid Demo`: a pyramid of 140 to 650 boxes (7 to 12 layers) on the GPU solver through the public API (`AvbdBodyBuffers`, `AvbdSolver.Step()`), a fly camera, left click to shoot a box in the direction the camera looks, and a panel for the shot's speed, size and mass, the solver iterations and the pyramid height. Includes the scene `PyramidShooter`. It contains no collision code and needs the contact stages inside `Step()`; until then the boxes fall through each other. It was written without access to a Unity editor and has not been run.

### Notes

- The stage U code and tests were written without access to a Unity editor or a GPU and have not been run. The speed limit listed for stage U in the design doc is not implemented (upstream has none).
- The stage E code and tests were written without access to a Unity editor or a GPU and have not been run. Growth of the buffers beyond their initial capacity is not implemented yet; `Spawn` returns an invalid handle when the buffers are full.
- Gravity is a vector (`AvbdSolverParams.gravity`), default `(0, 0, -10)` to match upstream's z-up demo scenes; use `(0, -10, 0)` for Unity's y-up.
- `AvbdQuat.Difference` flips the sign when `w < 0`; upstream does not. The solver uses upstream's `DifferenceRaw` unless `AvbdSolverParams.canonicalizeQuatDifference` is set.
- The reference solver compiles and its Edit Mode tests pass in Unity 6.6. The Reference Solver Demo sample has not been run in a Unity editor yet.
