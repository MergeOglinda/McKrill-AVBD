# Internal SDF → production AVBD bridge

`src/avbd3d/gpu/sdf-bridge.ts` installs an opt-in collision replacement in
`GpuSolver3D`. It feeds the existing adjacency, colouring, primal/dual solve and
velocity update. The class remains internal; `World.enableSdfCollisions()` now
provides the spatial automatic path. See `SDF_WORLD.md`.

```ts
const bridge = await SdfCollisionBridge.create(
  solver,
  [{ field: groundField }, { field: objectField, scale: 1 }],
  [{ a: 1, b: 0, key: 1 }],
);
solver.setCollisionBackend(bridge);
solver.step();
// Same physical pair, same key: retain its sampling and force history.
bridge.setPairs([{ a: 1, b: 0, key: 1 }]);
// Restore the normal backend and destroy the installed bridge:
solver.setCollisionBackend(null);
```

## Contact and cache contract

- Each active SDF point becomes a one-contact production manifold with its own
  normal. No shared-normal approximation and no production contact ABI change.
- Candidate keys identify ordered body pairs; persistent point IDs identify
  contacts within that pair. Cache lookup does not depend on rank, atomic output
  order or pair-list order. The SDF result stores the exported production contact
  slot plus one in `Point.flags.w`; zero means no exported contact.
- Warm-started multipliers come from the previous solved contact, are transformed
  from the old basis to world space and into the new basis, then scaled by the
  solver's alpha/gamma. Penalties decay/clamp as in the production solver, including
  its optional mass-based normal penalty floor.
- Sticking contacts retain their local anchors, except on rolling bodies: when either
  body is an analytic sphere, capsule, cylinder, cone or torus (atlas kinds 1-5), anchors
  are refreshed every step so a rotating material point is not pinned, and the contact
  key carries `SDF_ROLLING_BIT` so `evalContact` uses step-start lever arms (as for the
  built-in `SPHERE_FEATURE`). `Result.stats.w` carries this rolling-pair flag; sampling
  query/iteration/ID counters are unchanged. The flag applies to the whole pair, so a box
  resting on a sphere also refreshes its anchors (FINDINGS 2026-10-18).
- The normal constraint uses `separation - margin + allowedPenetration`, matching
  the CPU SDF bridge. This differs from upstream's sequential impulses inside the skin:
  upstream stops all approach once `separation < margin` and adds Baumgarte correction
  only beyond the allowed penetration, while this AVBD constraint applies force only
  below `margin - allowedPenetration`. Resting gaps agree; behaviour within the skin does not. Fresh tangent constraints start at zero; retained sticking
  anchors retain their tangential displacement.
- The standard pair hash is bypassed while this backend is installed. Multiple
  same-pair manifolds therefore cannot alias through its single-pair lookup.
  Replacing/detaching a backend clears collision counters/history; joints remain.

## Ownership and limits

The solver owns an installed backend and destroys it on replacement or solver
teardown. Destroy an uninstalled bridge yourself if it will not be installed.
Do not directly destroy a bridge while it is installed.

This first bridge replaces **all collision detection**, not just selected pairs.
Explicit mode accepts a candidate list; `createAutomatic` generates nearby
pairs through a GPU hashed grid for fixed slots. Both modes apply GPU SDF AABB, static/static, mask
and joint exclusions. Automatic mode reserves its required solver capacity.
Pair indices are GPU slots, with `a > b`, and each physical pair appears at most once.
Fields cover every current GPU body slot, respecting `solver.refToGpu` when spatial
sorting is enabled. Field data and uniform scales are snapshotted at creation.
Mass, diagonal inertia, friction, pose and velocities remain solver-owned.

Topology and field-slot assignments are fixed for each bridge instance. World refresh
builds a replacement and transfers unchanged pair history on the GPU. Recreate
it when adding/recycling body slots or changing fields. Reordering/removing candidates
is supported; increasing candidate capacity requires recreation. Change a pair key
to deliberately reset its history. Candidate capacity must fit five production
contacts **and manifolds** per pair; this check happens before installation/stepping.

All pose extraction, bounds construction, narrowphase and contact export run on
GPU in the solver command stream. There is no per-frame CPU pose/contact readback.
Only explicit `readResults()` diagnostics read back. The export stage uses eight
storage bindings and requires no optional WebGPU features. Host uploads carry
candidate indices, parameters and static field configuration.

## Validation and remaining work

Small GPU tests cover differing normals, normal/tangent force transport, retained
penalties, pair reorder, changed keys, empty candidates, sticking anchors,
parameter rejection, fallback to standard collisions and a falling sphere supported
for 120 production steps. The falling case caught and fixed spherical anchor pinning.

Validation on the local Dawn adapter: 83 selected GPU tests and 129 CPU tests
pass; nine bridge/World checks pass after the final teardown guard, including one
new regression. Typecheck and both builds pass. Five scale/showcase tests remain excluded under
the concurrent-work load constraint; this is not the unrestricted project gate.

Still pending: scalable spatial broadphase, runtime field/body allocation,
longer nonconvex/friction comparisons, and
target-device performance/large-scene validation. The World binding and spatial
automatic filter are documented in `SDF_WORLD.md`.


### Pair events and ray queries

The World binding requests a reporting-only GPU pass after solving. Each stable
SDF pair emits one synthetic manifold/contact into separate event buffers, using
exported contact indices to read solved forces/anchors. The ordinary contact
watch then compares physical pairs across steps. The solver retains every independent
normal and its warm start; these synthetic records never enter the solve.

Queries refresh SDF body transforms directly from production GPU poses and reuse
the field atlas plus the standard nearest-hit reduction. SDF body `lo.w` contains
a conservative field slope bound prepared from config.z; neither collision bounds
nor collision sampling depend on that component. Query limits and numerical
semantics are documented in `SDF_WORLD.md`.


World also replaces spatial bridges on capacity growth. `grown(required)` reuses
compiled pipelines and the immutable packed atlas; production buffers grow while
copying both solved-contact parities. `inheritHistory(previous, retained)` copies
only pairs whose two slots are retained, including their solved-contact indices.
Installation follows the copy synchronously, with no intervening solver step.
Explicit candidate bridges retain their fixed-capacity contract.

### Deterministic per-body accumulation

The 3D adjacency sort reads each manifold's first contact feature key (with the
sticking bit removed) as the secondary key after the other body ID. Standard
collisions have one manifold per pair; SDF exports multiple one-point manifolds.
Sorting only by the other body left all those SDF rows tied, allowing atomic
allocation order to change floating-point accumulation and eventually pile poses.
The additional read-only contact binding brings topology to eight storage buffers,
within the baseline WebGPU limit. It does not merge normals or change warm starts.
A bounded two-run bunny regression compares physical states bit-for-bit after
180 frames on the same device. This is not a cross-device determinism guarantee.
