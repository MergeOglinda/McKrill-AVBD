# SDF GPU narrowphase — standalone draft, initial GPU tests passing

This is an internal standalone compute pass with initial small GPU tests passing.
It was first written and reviewed without GPU execution; subsequent local Dawn
validation is recorded below. Broader temporal parity and target-device performance
remain unverified.
It is not exported by the public package. An opt-in production bridge now exists;
see `SDF_PRODUCTION_BRIDGE.md`. The default collision backend is unchanged.

## Implemented code

- `src/avbd3d/gpu/sdf-layout.ts`: version-local CPU packers/decoder for field atlas,
  body transforms/bounds, pairs, collision parameters and contact records.
- `wgsl-sdf-fields.ts`: all CPU analytic kinds (box, sphere, capsule, cylinder, cone,
  torus, three warps, spike plate, spiky wheel), plus manual float4 voxel trilinear
  interpolation and outside-grid extension. Uniform scaling/COM transforms match
  the CPU body's convention. Custom fields must first be baked to voxels.
- `wgsl-sdf.ts`: one invocation per pair, 64 invocations/workgroup; bounded
  Nelder–Mead (`optimizer` 0, default), deep-cut ellipsoid (1, cached points warm
  started 0.75/0.25 from their stored ellipsoid) or stochastic gradient descent (2,
  `stepSize`, noise from the per-pair stream; opt-in `sgdGradientStopOnly` drops the ‖Δx‖ < τ stop);
  four cached plus one fresh candidate by default; cached-point
  velocity advection; curvature-seed reservoir sampling; first-wins duplicate
  removal; stable rank sorting; coherent per-point normals and projected anchors.
- `sdf-pass.ts`: explicit async creation on a **caller-supplied** GPU device,
  capacity/limit checks, uploads, double-buffered result/cache storage, per-step
  submission, explicit readback and destruction. Importing it does not request a
  device. No automatic browser startup, render loop or background compute.

The shader uses five storage bindings and one uniform binding, with no optional
WebGPU features. Float32 reference storage is intentional for first validation;
half packing/distance-only storage and profiling are later work. Default atlas
budget is 64 MiB. Atlas contents are snapshots: fields/COM/grids/seeds must remain
unchanged for the lifetime of the pass. Recreate the pass for shape edits.

## Data contract

All records use 16-byte alignment. Metadata lives as uint32 bits, never float32
numeric IDs, avoiding subnormal flush and float-ID precision problems.

- Atlas header: 8 vec4 records; kind/convex/seed offset/count; grid offset/dimensions;
  bounds min/cell; bounds max; COM; two analytic parameter records; voxel origin.
  The following records hold optional float4 grid nodes and vec4 seed positions.
- Body: 7 vec4 records; COM position/uniform scale, field quaternion, linear velocity/static
  flag, angular velocity/header uint32 bits, world bounds min/max, and the anchor rotation
  (field frame to production body frame; identity for the standalone pass).
- Pair: 4 uint32s; body A, body B, stable pair key, prior result slot or `0xffffffff`.
  Pairs must have unique keys. Keep body ordering consistent. Change keys when
  body slots are reused. Prior slots may explicitly remap reordered pairs.
  The shader checks A/B/key before using a cached record.
- Result: 148 words (592 bytes) at the default 5 points; identity/count, query/iteration/next-ID counters,
  then 28-word point records (the last 8 words hold the symmetric ellipsoid warm start for optimizer 1; they are present for every optimizer, a 40% per-point cost, see FINDINGS). Each point stores optimizer position/depth,
  world normal/separation, body-A anchor/softmax, body-B anchor/rank, and uint32
  active/cached/persistent-ID flags. Anchors are body-local offsets from COM in the
  production body frame (field frame rotated by the anchor rotation), **already in
  world-length units**, not unscaled field coordinates. Persistent ids stay below
  `SDF_ROLLING_BIT` (0x20000000) and skip `SPHERE_FEATURE` (0x03000000); opt-in vertex
  contacts use `0x40000000 | side << 27 | vertex`.

Only `active` points are contact constraints; inactive ranked samples still belong
in the next search cache. No friction/normal impulses are solved or transported by
this pass. IDs survive rank changes but are scoped to a pair key; they are not global.
The caller supplies candidate pairs and explicit common world-unit parameters.

Randomness intentionally uses a 32-bit hash stream seeded by ordered body IDs,
pair key and step index, independent of invocation scheduling. It does not reproduce
the CPU oracle's global 64-bit PCG stream. Deferred parity tests replay the GPU seed
policy on CPU to isolate field/optimizer differences from random sampling differences.

## Validation performed without a GPU

CPU tests cover atlas layout/deduplication, float and integer record encoding,
COM/scale bounds, invalid capacities/inputs, workload limits, stale pair slots,
contact decode, and host dispatch/cache buffer swapping using a plain JS mock.
The mock does not evaluate WGSL or simulate GPU execution. Typechecking includes
GPU test source, but the GPU test runner is never imported by CPU tests.

`tests-gpu/sdf.gpu.test.ts` now passes on the local Dawn adapter. It covers transformed field
samples vs CPU, shader/pass compilation, sphere contacts, reordered caches, stale
keys, empty/disjoint pairs, and optimizer parity under the same per-pair seed policy.

## CPU-only review (2026-10-14)

Independent static review found two P2 host issues, now fixed: mutable body scales
that underflow to zero or overflow in float32 are rejected before upload, and
failed pass construction destroys every buffer already acquired. CPU regressions
cover invalid scales and all six allocations, atlas upload and both bind groups.
Re-review found no further actionable regression by static inspection.

All 129 CPU tests, typecheck and both builds pass, sequentially at niceness 15.
No shader compilation or GPU execution was performed; these checks do not prove
WGSL validity or GPU numerical correctness.

## First small GPU validation (2026-10-14)

Real shader compilation caught reserved WGSL identifiers `meta` and `active`;
renamed them to `flags` and `activeCount` without changing buffer layout. All three
SDF tests then passed, including field samples, contact/cache behavior and optimizer
parity. The broader sequential run passed 73 selected GPU tests in 53.37 seconds.
Five stress/showcase tests were excluded by name to limit concurrent GPU load;
assertions and scene sizes were not weakened. No benchmarks were run.

System GPU utilization was 53% before and sampled at 62%, 88% during and 69%
afterward. These are system-wide snapshots, not per-process measurements or a
guarantee of no contention. CPU priority was niceness 15; GPU priority was not capped.

## Bounded temporal validation (2026-10-14)

Two more real-Dawn tests pass (five SDF tests total). The temporal case runs 12
frames for sphere and voxel shapes with A static, B static, or both dynamic: 72
pair updates. It compares CPU positions, separation, active/cached status and
checks persistent IDs and projected anchors under translation, angular velocity,
COM offset and scale. It includes separation/re-entry and warm-start disable/reset.
The comparison uses one cached candidate and alpha zero to isolate advection.

A separate default-parameter case runs 65 pairs across two workgroups with explicit
curvature seeds, then reverses pair order and remaps all cache slots. It checks
multiple candidates, pair identity, finite/unit normals, rank order, duplicate
rejection, cached IDs and static-static suppression. This is invariant coverage,
not full default-parameter CPU trajectory parity. No production code changed.

Integration inspection confirms a structural constraint: production GPU manifolds
store one shared normal per pair, whereas SDF contacts require independent normals.
The new bridge preserves one normal per SDF point and separates pair-level cache
lookup from solver manifold records. It bypasses the existing pair hash and reads
production body transforms directly on GPU; see `SDF_PRODUCTION_BRIDGE.md`.

## Next GPU work

1. Repeat validation on target devices when available; the initial local small tests pass.
2. Expand hardware parity to pinned manifold fixtures, temporal cache/anchor
   behavior, voxel assets and scene metrics. Measure workgroup/register pressure.
3. The World spatial broadphase is implemented (`SDF_WORLD.md`). Runtime
   body/field refresh and capacity growth are implemented; incremental voxel-atlas
   allocation and larger mixed-size validation remain. Existing hull-offset shape tags
   cannot simply be assigned an overlapping `SHAPE_SDF` value.
4. SDF pair events and bounded raycasts are implemented (`SDF_WORLD.md`); compression
   and target-device profiling remain pending.

Until those gates pass, this remains an internal pass and opt-in bridge, not a feature ready
for users or publication. The unrestricted `pnpm check` remains deferred because it
includes the excluded stress/showcase tests.
