# McKrill-AVBD (Unity, 3D) compared with three-avbd (WebGPU, 3D)

Compared: `Packages/com.mckrill.avbd` (HLSL compute, Unity) against `three-avbd-main/src/avbd3d` (WGSL compute, TypeScript, Three.js). Only the 3D solver is covered.

## How this was done, and what it does not show

- Read in full: `gpu/layout.ts`, `gpu/wgsl-solve.ts`, `gpu/wgsl-collision.ts` (broadphase, hash, warm start, reuse; the box SAT/clipping section only by outline), the step encoding and the `adapt` / `adaptColors` logic in `gpu/solver.ts`, `docs/API.md`, `docs/PLAN.md`, `AGENTS.md`, and `docs/FINDINGS.md` from the Stage 8 entries on. The SDF collision backend was skipped (experimental, separate).
- `src/avbd2d` is not in the copy on main. Their 2D-shared coloring and adjacency kernels could not be read; what is said about them comes from their docs and from `solver.ts`.
- Every timing below is **their** measurement (M4 Max through Dawn or a browser, mostly interleaved A/B on one machine). None was reproduced here. Nothing on our side was compiled or run either: there is no Unity, GPU or HLSL compiler in the sandbox. Ours-side statements come from reading our shaders and design doc.
- A verdict says which design is better *for what*, judged from the code and their measurements. Where a verdict depends on a number we do not have, it says so.

## Summary

| # | Difference | Better | Size of the effect |
|---|---|---|---|
| 1 | Graph coloring: carried over, async tuning (theirs) vs from scratch with a blocking readback each step (ours) | theirs | Large for CPU stalls, unmeasured for us |
| 2 | Primal lanes per body: 1/2/4/8 chosen per color (theirs) vs a fixed 8 (ours) | theirs | −5% to −53% in their sweep |
| 3 | Contact storage: 32 B pair + 64 B points, pair-level adjacency (theirs) vs 160 B generic records, one per point, 8 slots per manifold (ours) | theirs | Their probe: +16 B per point cost 5–8%; ours is +96 B |
| 4 | Broadphase pair test: sphere + AABB + face axes (theirs) vs bounding sphere only (ours) | theirs | −15% on a brick ring, −2% to −5% on others |
| 5 | Broadphase grid: counting-sort CSR, two levels (theirs) vs atomic linked list, one level (ours) | theirs, modestly | Their findPairs is about 1/3 of collision time |
| 6 | Contact reuse for still pairs | theirs | 6–12% at 4 iterations, 3–6% at 10 |
| 7 | Spatial order: one-off physical sort at build (theirs) vs runtime iteration-order sort (ours) | theirs | 2–13% on piles; ours measured slower |
| 8 | One dual dispatch for joints and contacts (theirs) vs `Dual` plus `DualContact` (ours) | theirs, small | Dispatch count only |
| 9 | Dual per manifold, vs per contact record over 8 × slots threads | theirs | Fewer idle threads |
| 10 | Capacity: derived from a CPU start count and grown from readbacks (theirs) vs fixed pool with overflow report (ours) | depends | Ours is simpler and predictable; theirs copes with growth |
| 11 | Numerics and verification: bit-identical C++ parity harness for the narrowphase and contact math (ours) vs an f64 oracle with envelope tests (theirs) | ours, for the contact math | Correctness evidence, not speed |
| 12 | Feature scope: spheres, hulls, filters, plastic joints, wind, SDF (theirs) vs boxes, joints, springs, fracture (ours) | theirs for breadth | Not a performance point |
| 13 | Host integration: stable handles, edit queue, slot allocator, capacity calculator (ours) | ours | Theirs is a standalone web engine |
| 14 | Same in both | n/a | Listed at the end |

## 1. Coloring

**Theirs.** One Jones-Plassmann pass per step with a **fixed** number of rounds (`colorRounds`, 16 at start, adapted). The previous step's colouring carries over, so most bodies are already colored. A color cap and the round count are tuned from counters (`colors`, `clashes`) read back **asynchronously**: a stale readback can cost time but never omits a body. Bodies that find no free color under the cap, or a color above 64, are committed to a clashing mask and solved serially in chunks until the cap grows. Each color is dispatched indirectly, `colorCap` of them per iteration, with no host wait. They measured 16 → 2 rounds as only ~3% faster because the carried-over colouring has little to do, and about 1 ms of colouring/adjacency in a 110k step of ~8 ms.

**Ours.** `ColorInit` marks every solved body `PENDING` each step, so the colouring is rebuilt from scratch. `AvbdGraphColoring.Run` runs a first batch of 16 rounds, then `GetData` on the control words after every batch: the host **blocks until the GPU has finished everything queued so far**, learns whether bodies are waiting and how many colors there are, and only then queues the rest of the step. The design doc already calls this "a stall".

**Verdict: theirs.** The cost of ours is not the GPU time of the rounds but the sync: the CPU cannot queue the next frame's work or overlap with the GPU, and Unity's `GetData` also waits for the previous frame's GPU work. Their scheme avoids it entirely. Their weakness is the serial fallback when the cap is stale, which they handle with a per-body clashing mask and grow-at-once rules; they found a real bug there (colour-cap overflow made a pyramid collapse in 13 of 40 runs), so it needs care and a test that starts with a cap that is too small.

**What it would take here.** Keep colors in `_BodyMeta.state.x` between steps (only re-color bodies whose neighbors' colors clash or that are new), run a fixed round count, dispatch `colorCap` indirect dispatches per iteration, and update the cap and round count from an `AsyncGPUReadback` of the counters one or two frames later. `ColorPlan` already writes per-color indirect arguments, so the sweep side exists.

## 2. Primal: lanes per body

Both designs use several threads per body, each stamping a share of the body's constraint list into a partial 6×6 system, summed in workgroup/groupshared memory, then one thread does the LDLᵀ solve. That part is the same idea (ours: `AvbdPrimal.compute`, `StorePartial`, sum in lane order; theirs: `primalColor` plus `finishBody`).

The difference: ours is `AvbdGraphColoring.PrimalLanes = 8`, a compile-time constant for every color. Theirs picks 1/2/4/8 per color **on the GPU** from the color's size (`primalLanes` rule: never 1, 2 lanes from 32k bodies, 4 from 4k, else 8), and the args kernel sizes each color's dispatch the same way.

Their sweep (against one lane): wall smash −53%, chain mail −15%, jointed drop 34k −35%, settled pile 32k −19%, ring 110k −14%, gables 506k −15%, but box columns 100k +4%. Against the previous commit on AC: ring −12.7%, jointed drop −16.5%, gables −5.2%, columns +8.2%.

**Verdict: theirs.** The big colors of a large scene need fewer lanes: eight lanes on a 100k-body color means eight times the threads, eight partial systems through groupshared memory, and only 8 bodies per 64-thread group. Small colors need all eight. A fixed 8 should be near their best for small scenes and a loss for big colors, but I cannot say by how much for our contact record size; that needs a profile. Both reduce through workgroup memory (ours 33 terms per thread); theirs first had two lanes because 64 bodies × 176 B of partials had to fit in the 16 KB guaranteed workgroup storage, and later moved to the 1/2/4/8 scheme. Ours has the same 16 KB question to check for `AVBD_GROUP_SIZE` × 33 × 4 B on the target GPUs.

**What it would take here.** Make the lane count a per-color value that `ColorPlan` writes next to the indirect args, and read it in `Primal` as a runtime value (the lane-sum loop is `[unroll]`ed on the constant today and would need a runtime bound, or one kernel variant per lane count). A one-line rule can start from theirs.

## 3. Contact storage and adjacency

**Theirs.** A **pair record** of 32 B (both bodies, first-point index, count + generation step, normal, friction) and **64 B points** stored consecutively (up to 15 per pair; normal and bodies were moved to the pair, the stick flag lives in bit 31 of the feature key). Adjacency, colouring, the hash table and the dual iterate **pairs**. The primal loads the partner body and the contact basis **once per pair** and loops over the points. They measured this at −31% to −33% on 100k box columns (4 points per pair), and +3% to +10% (slower) on a random pile where a pair has 1.9 points on average. Contact capacity comes from the host's count of starting pairs (floors of 4 pairs and 4 points per body) and grows at 60% full. They also measured that padding the point by 16 B cost 4.8% (4 iterations) and 8.2% (10 iterations) on a 110k pile.

**Ours.** A contact is a generic `ConstraintRecord` of **160 B** (ten 16 B words, kept a multiple of 32 B for gathers by id), shared with joints and springs; a manifold slot owns 8 of them, which is the design doc's **1.3 kB per slot**. Each contact occupies one list entry per body, so a body's list carries one entry per *point*, and the shader skips repeats for the same neighbor in the coloring. The primal stamps each contact through the generic record, re-reading both bodies' poses via `LoadPose`. The dual runs over `8 × highWater` threads, most of which return at once.

**Verdict: theirs.** The contact record is 2.5× larger than theirs (160 B against 64 B, with the normal and friction repeated per point) and carries fields a contact never uses (stiffness, rest length, the second penalty set, `reserved`); their one probe says 16 extra bytes per point cost 5–8% at this scale, so ours should cost more, although that is an extrapolation and not a measurement. Pair-level adjacency also cuts list entries and partner-body loads where pairs have several points. For a random pile with mostly single-point pairs, their pair record was slightly worse, so the win is scene dependent; our test scenes (pyramid, stacks) have several points per pair.

Our advantage: a **single record type** keeps the primal, lists and coloring generic and made adding springs and fracture cheap, and the per-slot pool with persistent slots means the warm start needs no hash lookup. Moving to a compact point plus a pair record is the largest change in this list; it touches `AvbdConstraintRecords`, the contact kernels, lists, coloring and primal.

## 4. Broadphase pair test

**Theirs** (`testPair`): collision-group filter, bounding sphere, then **world AABBs** (with 0.1 mm pad), then a **separating face-axis test** along both boxes' three axes (1 mm tolerance; the narrowphase would reject on the same axes anyway), then the no-collide list by binary search. The `findPairs` thread loads its own shape (position, radius, axes, AABB) once. Their numbers: on a brick ring 72% of the pairs that reached the narrowphase were apart; the AABB and face-axis rejection gave −14.7% on the ring 110k, −9.7% on ring 9k, −2% to −5% on jointed drop, pile, box columns and wall smash, and +0.9% on chain mail.

**Ours** (`BpPairsDynamic`, `BpPairsStatic`): bounding sphere overlap and `!Constrained(self, other)`. No AABB or axis test, no collision filters (`IgnoreCollision` is listed as a known gap in the design doc).

**Verdict: theirs.** A bounding sphere of a box overlaps far more often than the box does. In our design every candidate pair without a manifold costs a slot probe in `NpAllocate` and a **full SAT test, run a second time in `Narrowphase` for a pair that is accepted**; a pair that is apart costs the first test every step. Rejecting on AABB and face axes in the broadphase is a small kernel change and the cheapest item in this list, with the best evidence behind it. Keep the same conservative rule (a rejection must be one the narrowphase would also make), and the same CPU oracle comparison for the broadphase test.

## 5. Broadphase grid

**Theirs.** Counting-sort **CSR** grid: `gridCount` (atomic count per hashed cell), a prefix scan, `gridScatter`, with the cell of each body stored beside the list so a probe can reject a hash collision without touching the body. Two levels (`fineGrid` and `cellSize`) so a few large bodies do not force a huge cell for everyone, and oversized bodies (ground) are tested brute-force from a list. Fixed bodies do not probe; their pairs with large dynamic bodies are `findFixedLarge`. Pair order is stable once sorted.

**Ours.** One level, cell = 2 × max radius, `_GridHead` / `_GridNext` linked lists built with `InterlockedExchange`, one cell hash per probe cell. Oversized statics go to a brute-force list (same idea). Only solved bodies probe (same idea).

**Verdict: theirs, but this is not where their time goes.** `findPairs` is 1.4 of 4.0 ms of collision on the 110k ring and 2.2 of 8.4 ms on the 506k gables; the narrowphase is the rest. A CSR grid makes the bodies of one cell contiguous in an index array and stable between runs, but they did not pack position and radius in cell order either: each candidate still reads the body record. This bears on the earlier question about a cell-sorted layout: from their numbers the pair *search* is a minority of collision time, and the cheaper win is rejecting pairs earlier (section 4) and not re-testing still pairs (section 6). The two-level grid matters only for scenes with widely different body sizes. A linked list also visits the bodies of a cell in an order that depends on atomic timing; we already fix the summation order downstream by sorting the lists by pair, so that is not a determinism problem for us.

## 6. Contact reuse

Theirs (`reuseContacts`, on by default): a small `updateRefs` kernel keeps a reference pose per body; a body that moves more than 2 mm or turns more than 2 mrad takes a new reference and records the step. A pair whose points were computed no earlier than both bodies' last move **skips SAT and clipping** and keeps its points; `C0` is recomputed from the current poses so the constraint error stays exact, only the anchor locations can be up to the tolerance stale. Reuse rate: box columns 96% of pairs by step 120, a 32k random pile 22% at step 240 and 94% at 600. Gain 6–12% at 4 iterations and 3–6% at 10 (smaller than the SAT share suggested, because hash lookup and copying still cost). They also tried a per-pair relative-offset criterion: much higher reuse, only about 2% faster, reverted.

Ours: the narrowphase recomputes every slot every step.

**Verdict: theirs, but a modest gain, and a correctness trade.** Their parity tests run with reuse off, because anchors can be stale by up to the tolerance. In our design it would be natural (the slot already persists, so no hash lookup is needed), but it makes results differ from the upstream reference, so it needs to be a setting that defaults to the reference behaviour, matching how they gate it.

## 7. Spatial order

Theirs (`spatialSort`, default on): bodies are stored in **Z-curve order of their starting positions**, once, when the scene is built; the app uses GPU indices (`gpuIndex(i)` maps reference indices). Measured on a settled pile rebuilt three ways: scrambled 2.94 ms, builder order 2.83 ms, Morton 2.52 ms at 4 iterations, 32k bodies (7.84 / 7.75 / 6.79 ms at 108k). Combined with reuse: 13% (32k pile), 2% (100k columns), 26% (wall smash 2k) at 4 iterations. They explicitly say a runtime re-sort that remaps every index "was not worth building".

Ours (PR #27, `SortBodies`, off by default): a runtime Morton sort that produces a **permutation of iteration order** over slot-indexed arrays. It measured slower on the pyramid, which already had spatially coherent slots: the permutation turns streaming access into gathers and adds 18 dispatches.

**Verdict: theirs.** It confirms the diagnosis in the PR: the benefit is a physical arrangement of the data, applied once at build, not an iteration order. We have the pieces for that (`BodyHandle` indirection, `BodySlotAllocator`), so an initial Morton ordering of slots when a scene is spawned in one batch is the straightforward version. Their gain is biggest for scrambled or builder-ordered piles and nearly zero for regular columns.

## 8–9. Dual

Theirs launches **one** dual dispatch per iteration (`dual`, indirect, over `IA_CONSTRAINTS`) that covers joints and pair manifolds, plus the `warmStart*` kernels. Ours has `Dual` (joints and springs) and `DualContact` (indirect, one thread per contact *record* of every slot below the high-water mark, `8 × slots`). Free records return at once. A dual over manifolds with consecutive points, as in section 3, removes the idle threads and merges the two dispatches. They profile the dual at about a third of the solve (gables: 4.85 of 14.7 ms; streams ~700 MB per iteration at ~440 GB/s on a 546 GB/s device).

**Verdict: theirs**, but a small effect: it is a dispatch-count and thread-count saving that follows from section 3, not a change of its own.

## 10. Capacity and growth

Theirs counts the scene's starting pairs on a CPU grid at build, allocates from that (floors of 4 pairs and 4 points per body) and grows buffers when a readback shows 60–80% fill or an overflow bit. They found two real failure modes: an undersized first allocation dropped pairs for two steps and left a brick wall in a worse state for good, and a falling pile doubles its pair count between readbacks. Ours uses fixed capacity (`contactCapacity`, at most 500,000 manifolds = 640 MB) with an overflow report and a candidate that retries next step, plus a high-water mark that sizes indirect dispatches and never falls.

**Verdict: depends.** For a Unity package a fixed, predictable pool is easier to reason about and avoids reallocation hitches; theirs is friendlier to scenes whose size is unknown. The 1.3 kB per slot in ours makes the fixed pool expensive (see section 3). Both approaches need a documented overflow policy; their failure analysis is worth reading before we decide ours.

## 11. Numerics and verification

Ours has `Tools/HlslParity`: the contact HLSL compiled as C++ and compared against upstream's unmodified solver on about 800,000 narrowphase cases, bit-identical (apart from one quaternion product's rounding order). Theirs has a CPU f64 oracle in `src/avbd3d/ref` and validates the GPU with single-step diffs and metric envelopes, since f32 and GPU ordering make long trajectories diverge; they also run a determinism tool and fixed seeded parity tests with their optional features switched off.

**Verdict: ours, for the arithmetic of the contact and narrowphase math** (bit-level evidence); theirs for the breadth of scenes and metrics (energy, penetration, "wall falls" criteria) and for catching integration bugs such as the colour-cap overflow. Their LDLᵀ clamps each pivot to at least `1e-5 × diagonal` (documented as protecting directions held only by a light body's inertia in f32); ours counts a failed solve in `_Status[0]` and leaves the body where it was. Which is better for very light bodies is untested here; their clamp always produces a step, ours never produces a bad one.

## 12–13. Scope and integration

They add features we do not have: sphere and convex-hull shapes, collision groups and filters, joint plastic yield and bend-break, wind, per-body gravity scale and damping, lifecycle flags, and an experimental SDF backend. Ours has stable handles, an edit queue, slot allocation with dirty-slot handling, a capacity calculator, springs, fracture, and a Unity-native API. Neither list is performance related, and the feature gaps in ours (filters, spheres) are cheap to add on top of the contact path once sections 3 and 4 are done.

## Where the two are the same

- Per-color primal sweep with a Gauss-Seidel order, bodies of one color share no constraint, no atomics in the primal, `LDLᵀ` of the 6×6 system, indirect dispatch per color.
- Warm-started penalties and multipliers (`alpha`, `gamma`), contact `C0` at the step start, friction cone clamp, stick flag.
- Hash-based Jones-Plassmann priority; only dynamic bodies probe the grid; oversized static bodies tested brute-force.
- Reference-matching narrowphase: 15-axis SAT with face clipping, up to 8 points per pair.
- A graph of constraints rebuilt every step on the GPU (CSR lists).

## Suggested order of work

1. **Broadphase AABB and face-axis rejection** (section 4). Small change, strongest evidence, no layout change.
2. **Persistent colouring with asynchronous tuning** (section 1). Removes the per-step CPU stall; needs a test that starts with too small a cap.
3. **Per-color lane count** (section 2). Small shader and `ColorPlan` change; profile first on a large single-layer scene to see whether 8 lanes hurts.
4. **Compact contact point plus pair record** (section 3, which also gives 8–9). Largest change and largest expected effect on bandwidth.
5. **Initial Morton slot order at batch spawn** (section 7), replacing the runtime sort for the cases where it was meant to help.
6. **Contact reuse as an opt-in** (section 6).

Each step should be measured with a per-stage GPU profile (their method: per-dispatch timestamps in real steps, interleaved A/B in one process) before it is kept; their own history includes several reverted ideas that looked good locally (per-pair reuse offset, skipping still bodies in place, scalar SDF shortcut).
