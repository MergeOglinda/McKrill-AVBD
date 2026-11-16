# Mesh-quality bunny performance — STE-3016

The 27-body mesh-quality bunny scene now passes three bounded 30-second browser screens on
M4 Max with the candidate below. This is an unmerged follow-up to qualification PR #44,
not a claim that the dense paper piles or other devices reach 60 FPS.

## Changes and correctness contract

The optional exact mesh-vertex witness scan was serial inside each pair's narrowphase invocation.
It now uses one 64-lane workgroup per pair to select the lowest-field-distance vertex, with
feature-ID tie breaking preserving original A-then-B vertex order. The original scalar path
re-evaluates that one selected vertex and emits its contact. Reusing the workgroup's computed
point/field value changed normal/anchor rounding in early prototypes, so those were rejected.
The scalar winner path passes a strict 24-frame direct/parallel contact comparison, including
reversed pairs, previous-slot remapping, growth, disabling mesh contacts, and all optimizers.
A matched unjittered 180-frame rollout preserves every body-state and contact hash.

Independent point normals, vertex IDs, solver warm-force transport, history migration,
collision parameters, substeps, iterations, mesh resolution, shadows and rendering resolution
are unchanged. The added scratch is 32 bytes per pair, bounded to 64 MiB and device buffer
limits. Devices without six storage bindings, the existing large-scene seed-cache variant,
and the explicit diagnostic serial option retain the old path. Two-dimensional dispatch
handles pair capacities beyond one workgroup-count dimension.

Released-scene normal playback also bounds outstanding frames to two using queue completion.
When both slots are occupied, it defers the next whole simulation/render submission; deferred
wall time remains owed and is offered to the solver within its existing per-frame step budget.
Pause and scene reload clear old time debt. This applies to ordinary playback, not just
qualification URLs. It changes scheduling, not the physical solver or its quality settings.
Sustained overload can still grow time debt and fail the simulation-rate threshold; this is
not an unlimited-capacity guarantee. Diagnostic reference playback retains its existing path.

Browser testing also found a qualification-only capture-start race: a queued rAF timestamp
can precede the click's performance.now(). PR #44's follow-up commit ignores such pre-start
frames before the capture baseline; the full 30 seconds and later monotonicity checks remain.

## Browser evidence

Same production rendering settings and predeclared acceptance as
[SDF_BROWSER_QUALIFICATION.md](SDF_BROWSER_QUALIFICATION.md): Chromium 155, M4 Max / 128 GiB,
1280×720 CSS, 1920×1080 drawing buffer. Each capture runs alone, from reset, for 30 seconds,
then pauses, with at least two seconds idle between trials. No headless GPU work overlaps.
FPS is rAF submission cadence; timestamp values are queue envelopes, not active kernel time.

| Variant | FPS | Frame p95 ms | Physics p95 ms | Total p95 ms | Drain ms | Screen |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original mesh-quality, earlier trials | 68.0–87.8 | 16.9–17.6 | 5.636–7.406 | 5.833–7.668 | 48.0–62.5 | Fail |
| Cooperative witnesses only, trial 1 | 119.2 | 9.1 | 6.160 | 6.226 | 7.1 | Fail physics |
| Cooperative witnesses only, trial 2 | 116.8 | 9.2 | 4.915 | 5.177 | 46.9 | Fail drain |
| Witnesses + bounded queue, trial 1 | 118.5 | 9.3 | 3.801 | 3.867 | 6.9 | Pass |
| Witnesses + bounded queue, trial 2 | 119.6 | 9.3 | 3.801 | 4.194 | 7.8 | Pass |
| Original serial search + bounded queue, trial 1 | 57.2 | 25.3 | 5.439 | 5.636 | 28.7 | Fail pacing |
| Original serial search + bounded queue, trial 2 | 59.7 | 25.1 | 5.571 | 5.964 | 26.9 | Fail pacing |
| Restored cooperative candidate, trial 3 | 119.3 | 9.3 | 3.539 | 3.539 | 5.6 | Pass |
| Chairs, same final candidate | 120.0 | 9.2 | 3.670 | 3.801 | 6.5 | Pass |

All three final bunny captures and the chair capture retain 1.00× simulation rate, finite endpoint
poses, zero endpoint overflow, no GPU errors, and no invalid working timestamp samples. All
failed captures are retained, including the pre-click instrument failure. Independent raw-data
recomputation checks duration, FPS, percentiles, timestamp additivity and original thresholds.
Endpoint snapshots alone do not certify transient penetration or all-time overflow absence.

A final ablation used the same paced production demo with the original serial search; only
the three solver witness source files differ. Both 30-second trials failed pacing at 57.2–59.7
FPS and 0.945–0.978× simulation rate. Restoring the cooperative build immediately afterward
passed again at 119.3 FPS, 1.00× simulation and 5.6 ms drain. The queue alone is insufficient
in these matched browser screens, so the cooperative path is retained.

These sequential observations support the combined candidate on this device. They are not a
randomized causal study separating scheduler effects, thermal state and kernel cost. The
headless diagnostic independently isolates narrowphase work: original collision samples span
10.49–11.93 ms per four-substep frame; the scalar-winner candidate spans 6.75–10.94 ms. It drains
every substep and adds profiling boundaries, so its times must not be compared directly with
browser envelopes. A later 30-frame dispatch diagnostic measured 11.928 ms narrowphase plus
0.262 ms witnesses, so these variable native traces do not establish a robust isolated-kernel
speedup. Browser ablation provides the practical acceptance evidence. Earlier transform-cache, serial split-pass and lazy-gradient prototypes
failed parity and/or lacked clear speedup and are not in the production candidate.

## Evidence and reproduction

Raw captures, baseline/repeat/candidate native traces, direct rollout hash comparison and
source/build manifests are in [the evidence directory](measurements/sdf-bunny-performance-2026-10-24/).
Browser records distinguish the cooperative-only and final paced builds. Native diagnostics
predate the demo pacing change; their solver variant is the same scalar-winner implementation.
No timing assertion depends on the browser instrument's original invalid start capture.

- `pnpm check` runs the complete project gate. Use one GPU test worker when bounding load.
- `node --experimental-transform-types scripts/sdf-bunny-profile.ts OUT.json mesh-quality 180`
  produces a bounded, queue-drained native phase trace. Optional `dispatch` additionally splits
  collision dispatches and labels the new mesh-witness pass. It requires timestamp queries.
- Build/preview on port 5317; open `sdf-released.html?qualify=1&scene=9&profile=mesh-quality`.
  Click Measure 30s, save JSON, idle, Reset and repeat. Use scene 6 for chairs.
- `scripts/sdf-settling-ensemble.ts BASELINE_DIR OUT.json` runs the unchanged matched 11-seed,
  360-frame chair/bunny acceptance protocol, one bounded GPU process at a time.

Final `pnpm check`: **177 CPU + 186 real-adapter GPU tests pass**, zero failures/skips;
typecheck, Vite build and library build pass. Independent review found no remaining material
issue, and separately recomputed the browser records and source/build hashes.
The complete matched ensemble **passes the unchanged no-regression rule**: all 11 seeds ×
2 scenes × 360 frames retain exactly equal recorded metric series between baseline and
candidate. This includes search query/iteration counts, contact/pair counts, kinetic energy,
lowest centers, contact depth, and surface probes every five frames. No threshold is relaxed.

| Scene | Failing seeds, baseline → candidate | Probes ≥5 mm | Median peak penetration |
| --- | ---: | ---: | ---: |
| Chairs | 0 → 0 | 37 → 37 | 11.9653 mm → 11.9653 mm |
| Bunnies | 1 → 1 | 14 → 14 | 0.81324 mm → 0.81324 mm |

Bunny seed 3 still fails the existing 0.01 J final-energy limit at 0.0579925 J in both variants.
This is an explicitly retained existing limitation, not an all-seeds settling pass. The
optimization does not resolve transient penetration generally. Full ensemble/raw summary
are included in the evidence directory, with frozen baseline/candidate source manifests.
The implementation and evidence are a separate follow-up; PR #44 contains qualification only.
