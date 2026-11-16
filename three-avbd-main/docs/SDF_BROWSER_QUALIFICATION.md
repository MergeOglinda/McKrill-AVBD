# Sustained browser qualification — STE-3005

## Predeclared protocol (before candidate measurements)

Base: merged main `dcd330d3e762969d108c61dd1bb880aaa58a88fd` (PR #43).
Use a frozen production build plus opt-in qualification instrumentation. Record file/build
hashes, browser user agent, actual canvas/drawing resolution, device features, scene,
profile, body count, capacities, iteration/substep budget, raw frame intervals and simulation
progress for every run. Do not mix reference and mesh-quality configurations.

Each trial runs alone for 30 measured seconds and pauses automatically, with at least two
seconds idle between trials. No headless GPU suite overlaps browser playback. Stop after
five consecutive frames above 100 ms or a GPU error; retain the partial failing report.
Default playback is unchanged; `?qualify=1` enables the measurement. Optional `timestamps=0`
exercises the timing-disabled fallback; absent GPU timestamps never counts as timing success.

Pacing passes only with all of: duration >=30 s, p95 <=18.5 ms, <=1% of intervals >25 ms,
worst <=100 ms, simulation/wall ratio 0.98–1.02, and final queue drain <=33 ms. Native
high-refresh cadence is allowed; this is render-submission evidence, not verified compositor
presentation. Sparse timestamp queue envelopes surround all physics substeps and rendering
in the same frame; these include CPU submission gaps and are not active GPU kernel time.
Timestamp sampling varies at 113–191 ms spacing to avoid high-refresh/fixed-step parity aliasing; each sample records its actual step count and acceptance excludes zero-step frames. Timing budgets: physics p95 <=6 ms, total p95 <=14 ms, >=60 nonzero-step samples. Ordinary production
adaptation readbacks remain; no per-frame diagnostic pose readback is added.

Physical criteria remain the existing settling suite/ensemble limits; no new regression or
threshold relaxation. End-of-capture snapshots report finite poses, overflow and errors,
but cannot prove that no transient overflow occurred. Browser full-speed capture does not
replace deterministic 360-frame chair/bunny acceptance. Memory reports JS heap and reachable
GPU-buffer allocation lower bounds, not driver/process peak memory.

Matrix: repeat the existing 12-body rings/warps/baked gears, 36-body showcase and 27-body
chair/bunny workloads; include initial impact, interaction/camera motion and late settling.
Screen 128 then 512 bodies separately using the same showcase fields in an explicit 8×8
layered layout. Do not attempt 1024 unless smaller screens justify extending the bound.
Exclude optional contact overlay from the base envelope; measure its cost separately.
Retain slow/failed trials. M1/GTX1080/mobile performance is neither required nor implied.

## Results — 2026-10-24

Measured on **Apple M4 Max / 128 GiB**, in-app Chromium 155, native high-refresh
rAF, 1280×720 CSS with pixel ratio capped at 1.5 (**1920×1080 drawing buffer**).
All trials were sequential, foreground, 30 seconds, starting from reset/initial impact;
no headless GPU tests overlapped them. The optional contact overlay is off except where noted.
These are bounded workload screens on this device, not portable or long-duration thermal certification.

| Workload | Trials | FPS | Frame p95 ms | Physics envelope p95 ms | Final drain ms | Screen |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 12 rings | 2 | 120.0 | 9.3 | 1.049–2.621 | 0.8–5.8 | Pass |
| 12 warped boxes | 2 | 120.0 | 9.3 | 2.687–2.818 | 0.4–4.3 | Pass |
| 12 baked gears | 1 | 120.0 | 10.2 | 2.687 | 0.5 | Pass |
| 36 baked gears | 2 | 120.0 | 9.3–10.3 | 2.884–3.080 | 3.5–7.5 | Pass |
| 27 chairs, mesh-quality | 2 | 120.0 | 10.2–10.3 | 3.408–3.473 | 7.6–8.8 | Pass |
| 27 bunnies, mesh-quality | 2 | 68.0–87.8 | 16.9–17.6 | 5.636–7.406 | **48.0–62.5** | **Fail** |
| 27 bunnies, reference | 1 | 119.9 | 9.3 | 4.588 | 6.9 | Pass |
| 128 baked gears | 1 | 120.0 | 9.3 | 2.556 | 0.8 | Preliminary pass |
| 512 baked gears | 1 | 120.0 | 9.3 | 3.080 | 8.9 | Preliminary pass |
| 12 rings, contact overlay | 1 | 119.1 | 9.3 | 2.949 | 0.4 | Separate overlay pass |
| 12 rings, timestamps disabled + drag | 1 | 120.0 | 9.3 | Unavailable | 5.3 | Pacing only |

The second chair trial includes camera orbit during playback. The timestamp-disabled trial
includes a successful public raycast/body drag during playback; absence of timestamps correctly
prevents a full timing pass. The overlay's rendering envelope rose to 5.374 ms total p95;
its diagnostic inspection suppresses some rendered frames, so CPU update and rendered-frame
sample counts differ. The separate screen includes that cost rather than hiding it.

The 128/512 cases use resolution-32 showcase gear fields, the existing showcase box-inertia
approximation and an 8×8 layered layout, not the paper's dense bunny pile or released gear
mass/inertia. Capacity is explicitly enlarged only for qualification URLs. These single
screens justify further exploration, **not a generic 512-body guarantee**. No 1024-body run
was attempted; resolving the mesh-quality bunny miss is more useful than expanding the count.

Both mesh-quality bunny trials fail the predeclared 33 ms final-drain limit. The second also
fails the 6 ms physics budget. Its physics envelope (7.406 ms) accounts for most of the total
(7.668 ms); per-kernel attribution still needs measurement. Queue backlog and variation are
tracked in [STE-3016](https://linear.app/sbobyn/issue/STE-3016).
Reference settings are reported separately and are not a substitute for fixing the quality profile.

All accepted V2 records have finite postcapture poses, zero endpoint overflow, no reported GPU
errors, and every working timestamp triplet passes independent raw validation. Postcapture
kinetic energies, using original masses/principal inertia: chairs 0.00000649 J in both runs;
mesh-quality bunnies 0.004216 / 0.001532 J; reference bunnies 0.0000477 J. These endpoints are
below the existing 0.01 J settling limit, but do not establish transient penetration bounds or
replace the matched 360-frame ensembles. PR43 preserves those ensembles, including the known
bunny seed-3 failure; no solver/quality threshold was changed here. Memory snapshots are
included in raw captures, not presented as total VRAM or leak certification.

## Evidence and instrument validation

[Measurement index](measurements/sdf-browser-sustained-2026-10-24/index.json) lists every
capture and SHA256 of its decompressed JSON. Raw records are losslessly gzip-compressed,
including failures, all frame intervals, CPU submission arrays, GPU timestamps, completion
latency, endpoint poses, heap snapshots, configuration and resolution. Source/build manifests
are stored alongside them. V2 source SHA256:
`57d6cd1f015d193e0d78a202ef4148121cc3eb8c7d5ebbdaa5ccbb9200654b79`.

The first V1 ring capture yielded all-zero timestamps from empty boundary compute passes.
It is retained as **invalid timing evidence** (`rings12-1.json.gz`), despite its original
instrument's erroneous screen flag. V2 uses a real one-workgroup atomic marker at each boundary
and rejects zero p95. V3 additionally rejects even one invalid working sample; independent
CPU audit applied that stricter rule to every V2 sample and found none invalid. Zero render-only
segments are allowed because observed timestamp quantization is 65.536 microseconds; working
physics and total spans must remain positive and additive. Do not treat a zero-query result
as zero-cost execution.

The final candidate changes only opt-in demo measurement and capture-window correctness, not
solver search, contact generation, independent normals or warm starts. Production playback
keeps its normal capacity and rendering settings. Instrumentation itself adds sparse boundary
submissions; these numbers should not be described as instrumentation-free kernel timings.

## Reproduce

Build and serve with `pnpm build` and `pnpm exec vite preview --host 127.0.0.1 --port 5317 --strictPort`.
Open one foreground tab, wait for ready, and click **Measure 30s**. It pauses automatically;
use **Save JSON**. Idle at least two seconds, Reset, then repeat. Do not run another GPU workload.

- `sdf.html?qualify=1&count=12` — select Rings, Twisted solids, or Baked gears before measuring.
- `sdf.html?qualify=1&count=36` (also 128 or 512) — choose Baked gears; counts/layout are explicit.
- `sdf-released.html?qualify=1&scene=6&profile=mesh-quality` — chairs.
- `sdf-released.html?qualify=1&scene=9&profile=mesh-quality` — bunnies; keep the failing workload visible in reports.
- Replace profile with `reference` for separately labelled released settings.
- Add `&timestamps=0` for fallback. Enable contact overlay only for its separate trial.

Validation: full sequential `pnpm check`, independent static review and raw-data recomputation.
Final gate: **173 CPU tests + 185 real-adapter GPU tests passed, zero failures/skips**;
typecheck, Vite and library builds passed. Tests ran at low process priority and sequentially.
Independent review found no remaining material issue; all 14 V2 records were recomputed from
raw samples. Two final V3 browser repeats (warps12 and gears36) passed at 120 FPS with zero
invalid working samples. V3 source SHA256:
`66d316bde8214a1597c771497fafe55d9183894a3fe80bde2dceec82eb6123b9`.

There are 17 retained captures: 14 V2 trials, two V3 repeats, and one invalid V1 instrument trial.
STE-3005 remains In Progress pending the bunny follow-up and a repeatable supported envelope;
this report does not claim all SDF workloads are qualified for 60 FPS.
