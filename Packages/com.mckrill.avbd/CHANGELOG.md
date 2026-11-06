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

### Notes

- Gravity is a vector (`AvbdSolverParams.gravity`), default `(0, 0, -10)` to match upstream's z-up demo scenes; use `(0, -10, 0)` for Unity's y-up.
- `AvbdQuat.Difference` flips the sign when `w < 0`; upstream does not. The solver uses upstream's `DifferenceRaw` unless `AvbdSolverParams.canonicalizeQuatDifference` is set.
- The reference port has not been compiled or run inside Unity yet; the algorithms were checked against the upstream C++ by reading and by running the upstream code headless to generate the golden data.
