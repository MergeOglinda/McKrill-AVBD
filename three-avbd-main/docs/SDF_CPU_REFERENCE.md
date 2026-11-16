# SDF CPU reference — phase 1

The internal `src/avbd3d/ref/sdf/` modules port the default collision path from
`savant117/sdf-dcd-demo` at `1dcdf1e7fa07d16fda15826ad8c2e6697075a76e`.
They do not change the public `World` API or run any GPU kernels.

## Implemented

- Float32 vector/body transforms and the upstream PCG32 stream, isolated from the
  existing AVBD float64 math. Quaternion inputs are normalized on construction.
- Box, sphere, capsule, cylinder, cone, torus, spike plate and spiky wheel fields.
- Bend, twist and displacement fields with analytic gradients (Sec. 4.5).
- Float4 voxel-grid trilinear sampling, including boundary clamping and the
  upstream outside-grid distance/gradient extension (Sec. 3.6).
- Softmax/repulsion objective (Eqs. 2–3) and domain-clamped Nelder–Mead (Sec. 3.3).
- Manifold updates (Algorithm 1): velocity advection, curvature-seed reservoir
  sampling, cache top-up/convex skip, coherent normals, Newton separation,
  deduplication, replacement bias and stable ranking.
- An opt-in `SdfCollisionAdapter` for the AVBD CPU solver. Each contact has its own
  normal/friction basis; cached identities persist independently of sorted rank.
  Sticking anchors and warm-start forces are retained, with forces reprojected
  into the current contact basis. Separated/deleted pairs remove their forces.

The adapter uses the existing AVBD force solve, **not** the upstream sequential-
impulse solver. Its behavior tests therefore do not assert trajectory equality
with the phase-0 scene traces.

## Validation

`tests/sdf-reference.test.ts` compares all 4,200 phi/gradient samples, all 16
optimizer fixtures and all 21 manifold updates. Numeric tolerances are explicit:
field values/gradients 2e-6 relative with an absolute floor; optimizer/manifold
positions within upstream tau; optimizer objective 1e-5; manifold normal/depth/
separation/rank 2e-5. Counts, active/cached flags, ordered contacts, iteration/query
counts and final RNG states match exactly on the tested fixtures. These are test
envelopes, not proof of equality for every possible floating-point near-tie.

`tests/sdf-adapter.test.ts` covers falling/settling boxes and a torus, a sphere
passing through a torus hole, force cleanup, reattachment, disposal and COM/scale
transforms. The reference tests also settle three voxel gears and a voxel bunny
using supplied mass properties, with a surface-seed floor-penetration probe.

Final CPU gate on 2026-10-14:

- `pnpm typecheck`: passed.
- `node --experimental-transform-types --test --test-concurrency=1 tests/*.test.ts`:
  **101 passed, 0 failed, 0 skipped**, including the existing f64 oracle regressions.
- `pnpm build` and `pnpm build:lib`: passed. Existing Vite chunk-size warning remains.
- Independent upstream voxel-grid generation/verification: passed.
- `git diff --check`: passed.
- GPU tests and full `pnpm check`: **deferred by user request**, not passed.

## Voxel parity inputs

The four compressed, little-endian float4 grids in
`tests/fixtures/oracle-sdf-grids/` were generated directly from upstream C++.
They are test inputs, not a TypeScript bake or npm assets. Their manifest pins the
upstream source/assets/compiler and exporter, and hashes the decompressed bytes.
All original phase-0 goldens remain unchanged.

```sh
nice -n 15 python3 tools/cpp-oracle-sdf/grids.py verify --source reference/sdf-dcd-demo
# Intentional regeneration only, never to make a port pass:
nice -n 15 python3 tools/cpp-oracle-sdf/grids.py generate --source reference/sdf-dcd-demo
```

Verification uses a fresh temporary executable and does not overwrite inputs.
The compressed grids add roughly 20 MiB of test data, allowing actual voxel
contacts and interpolation to be checked without installing C++ for ordinary tests.

## Internal use and limits

Construct an `AnalyticField(spec)` or `VoxelField(dimensions, origin, cell, grid,
bounds, surfaceTriangles?)`, then use `SdfBody`/`SdfManifold` directly, or attach the field to an
existing reference `Rigid` through `new SdfCollisionAdapter(solver).attach(body,
field)`. Call `dispose()` to restore the default box collision backend.

- The default box oracle remains unchanged when no adapter is installed.
- `SdfBody.position` is the center of mass; `field.centerOfMass` locates that COM
  in the field's local frame. The AVBD solver retains its existing z-up convention.
- **Mass and diagonal inertia remain caller-owned Rigid properties.** Attaching
  geometry does not automatically bake mass properties or add general tensor
  inertia support. The gear/bunny tests explicitly supply oracle properties.
- CPU mesh baking, mass integration, seed generation and worker integration are
  now exposed through the unpublished `three-avbd/sdf` subpath; see
  `SDF_BAKING.md`. `finalizeSdf` assigns ordered seeds; without supplied seeds,
  the upstream uniform-domain fallback is used. GPU World binding remains deferred.
- All three upstream optimizers are ported: Nelder–Mead (0, default), the deep-cut
  ellipsoid method (1) with its 0.75/0.25 cached-ellipsoid warm start, and stochastic
  gradient descent (2) drawing from the manifold's shared PCG stream. Both 1 and 2 use the
  analytic objective gradient. They are pinned to upstream by `optimizers.json` (16 cases,
  exact iteration and query counts) and `manifolds-optimizers.json`, which chains 4-step warm
  ellipsoid and SGD manifolds and checks RNG consumption exactly. The GPU narrowphase ports
  all three; its ellipsoid is checked against this reference, while its SGD noise comes from
  the GPU per-pair stream, so SGD trajectories are statistically, not bitwise, comparable.
  Opt-in `sgdGradientStopOnly` (not in upstream) drops SGD's ‖Δx‖ < τ stop; the default
  keeps the released behaviour and the oracle fixtures.
- Collision distance parameters use world units, while epsilon has length-squared
  units. `scaleCollisionParams` provides explicit common scene/pair scaling;
  no automatic per-body policy is applied. Low-level `SdfBody` supports uniform
  scale; the AVBD bridge uses the geometry's supplied local dimensions.
- The bridge uses a quadratic broadphase suitable for reference scenes. This is
  correctness-oriented CPU code, not a performance replacement for the GPU solver.
- The six-second phase-0 scene baselines and the new settling tests do not recreate
  the paper's Fig. 6 penetration experiment or its GPU performance tables.

## Shared-machine execution

Compilation, fixture verification, tests and builds ran sequentially at verified
niceness 15. Test concurrency was one; the final Vite build also limited Rayon/Go
worker pools. No GPU test, rendering context or benchmark was launched.

The final CPU suite took 20.04 seconds with maximum child RSS 280.8 MiB; typecheck
peaked at 524.0 MiB. Grid verification took 2.96 seconds / 131.6 MiB. A follow-up
system spot check showed 81.5% CPU idle and zero swap-in/out counters. This is not
continuous monitoring and does not establish zero impact on the other task's
latency. Temporary validation logs were kept outside the repository.
