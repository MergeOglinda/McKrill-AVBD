# Small SDF demos: release preparation

2026-10-24. Experimental source-build verification, recorded before Git publication.
The reviewable demo-only work is isolated on `codex/sdf-demo-release`, based on main 93a6fd8.
It contains no collision runtime changes or large-pile search scripts. The original research
checkout is preserved.
Large-pile C4 performance research remains separate and is not a release prerequisite for these
bounded demonstrations. No rejected search backend is enabled by either page.

## Useful scope

- `/sdf.html`: analytic rings, twisted boxes and resolution-32 baked gears. Twelve initial dynamic
  bodies, spawning capped at 36; public SDF ray picking, a ball-joint grab, reset/frame controls
  and optional contact points/normals. Mass/inertia remain box proxies, disclosed in the UI.
- `/sdf-released.html`: all eleven original-asset scenes, with captured mass/full inertia and
  resolution-100 mesh fields. Initial framing excludes the ground; the Frame button reads the
  current poses once to fit a spread-out pile. Default playback
  steps the existing GPU AVBD solver without diagnostic per-substep readbacks.
- Reference comparison is explicit, resets the scene and retains the previous bounded 360-frame
  diagnostic path. It is capped at 30 reference frames/s and cannot start an FPS capture.
- Both pages start paused, stop stepping while hidden, require Play after returning, and stop
  rendering a paused view once orbit damping settles. No background benchmark is started.
- Existing collision budgets, solver iteration counts, fixed timesteps, independent normals,
  warm-start transport and collision defaults are unchanged by this demo pass. The interactive
  loop retains at most two fixed steps/callback; released playback retains four. Time beyond the
  catch-up cap can be dropped; simulation speed is reported rather than hidden.

The experimental `three-avbd/sdf` preparation entry point and `World.enableSdfCollisions()` are
already present in public main. See [baking](SDF_BAKING.md) and [World integration](SDF_WORLD.md).
This source-build readiness record does not certify a published npm version or every open PR.

## Frame capture contract

**Measure 4s** explicitly resumes playback, excludes one second of warmup, records four seconds
of render-submission intervals at requestAnimationFrame timestamps and pauses automatically.
A 12-second watchdog cancels a capture without a complete visible sample. Final GPU draining
has a separate five-second deadline; a stalled/lost device produces a failed capture and restores
the Measure control. Scene/settings changes,
resize, hiding and manual playback changes cancel/reset the appropriate sample. Save JSON exports
raw intervals, viewport/backing resolution, browser version, scene configuration and timing scope.
No pose/contact readback is added to normal playback by this observer.

Reports distinguish:

- Browser render cadence: interval count / actual measured wall duration; no averaging of per-frame FPS.
- Median/p95 intervals and count of intervals over 25ms; long intervals are retained, not clipped.
- Simulation speed: fixed-step simulated seconds / the same interval duration. Minor differences
  around 1.00 are endpoint quantization, not proof of exact timing. A fast rendering loop cannot
  conceal a slow simulation in this metric.
- Final GPU queue drain and cadence including that drain. This checks outstanding submitted work;
  it is not per-frame GPU duration or a verified compositor presentation measurement.

No fixed sleep/idle is inserted between frames inside a capture. Different captures and all
headless GPU tests run sequentially; CPU/build gates run after browser captures, not alongside them.

## Production browser results

The table records captures before the review correction that bounded final GPU draining. Its
measurement-time source fingerprints are kept separately from the final reviewed assembly hashes.
The correction leaves normal playback unchanged; a final-build capture smoke is recorded below.

The clean main-based release branch was exercised through the Codex in-app browser after the
Chrome extension connection failed. Both pages ran from Vite's production preview, not the dev
server. All eleven released scenes and all three playground presets loaded and completed a
bounded capture. Final-build samples below cover the three presets, the spawn cap/overlay and
four released stack/pile scenes. The released samples were repeated after the final camera fix;
the playground's built page and dependencies remained byte-identical across that rebuild.

Apple M4 Max host, in-app Chromium 155, Three.js 0.186.1; desktop viewport and backing canvas
both 1728×913, DPR/render ratio 1, antialiasing and 1024² shadows enabled. Each report contains
one second of warmup and four seconds of render intervals. These are short observations on one
machine, not confidence bounds, GPU timestamps or cross-device guarantees. Other applications'
GPU activity was not controlled. No physical phone was measured.

| Scene | Dynamic bodies | Browser FPS | p95 interval (ms) | Simulation/wall | Final GPU drain (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rings | 12 | 120.00 | 10.2 | 1.000 | 6.9 |
| Twisted solids | 12 | 120.01 | 10.2 | 1.000 | 5.6 |
| Baked gears | 12 | 119.99 | 10.2 | 1.000 | 5.2 |
| Mixed spawned scene | 36 | 119.99 | 10.1 | 1.000 | 3.7 |
| Mixed, contact overlay | 36 | 115.75 | 10.3 | 0.965 | 3.7 |
| Released boxes | 28 | 120.01 | 10.3 | 1.000 | 5.9 |
| Released gears | 10 | 119.99 | 10.2 | 1.000 | 6.0 |
| Released chairs | 27 | 120.00 | 9.7 | 1.000 | 7.6 |
| Released bunnies, trial 1 | 27 | 120.00 | 10.0 | 1.000 | 8.3 |
| Released bunnies, reset trial 2 | 27 | 87.43 | 16.8 | 0.989 | 45.3 |

The 582×939 native browser pane also measured 120.00 FPS for twelve rings, p95 10.3 ms,
simulation/wall 1.000 and 5.8 ms final GPU drain, at DPR 2 capped to 1.5 (873×1408 backing).
Raw reports retain their exact viewport, resolution, timestamp and intervals. The slower final
bunny trial includes four intervals above 25 ms (maximum 25.5 ms); the other final samples have
none. The 45.3 ms bunny queue drain is retained separately; frame cadence
alone does not rule out outstanding GPU work.

The mixed scene adds eight rings, eight twisted solids and eight gears to the twelve-gear scene
after its capture. The overlay measurement follows it at a later state, so it is not a controlled
A/B cost measurement. Its simulation runs below real time; keep contact diagnostics optional.
Bunny cadence varied materially between resets, including an earlier 87.43 FPS / p95 17.0 ms
sample. Both final trials and earlier runs are retained instead of selecting the fastest result.
These measurements support useful small desktop demos, not the 4,314-body performance target.

Historical research-tree/development measurements remain in the machine record with their original
provenance. Initial production captures used differing backing resolutions after viewport/DPR
changes; those are retained separately and are not mixed into the consistent-resolution table.

## Production findings and fixes

- Both resize handlers now refresh the renderer's capped pixel ratio. Previously, a change of
  display DPR could retain the startup scale. The final browser check observed a 1728×913 backing
  at desktop DPR 1 and 873×1408 after returning to the 582×939 native pane at capped DPR 2.
- The released viewer's Frame button now reads the current body poses once. Previously it fitted
  initial poses, cropping a spread-out bunny pile at phone width. The corrected pile fits completely
  at 390×844. The read is explicit, cancels an active capture and blocks stepping during that read;
  it adds no per-frame pose readback to normal playback.

## Review corrections

Independent review found an unbounded final GPU-drain wait and an ambiguously labeled historical
source block. Draining now has a five-second deadline, tested with stalled and rejected CPU mock
queues; both return through capture cleanup without producing a successful report. Historical
research, measurement-time source hashes and current release assembly hashes are labeled separately.
A fresh production Spiky Wheel capture after the observer correction measured 119.87 FPS, p95
8.8 ms, simulation/wall 1.000 and 9.1 ms final drain. It paused automatically, restored Measure
and had no captured warning/error entries. No native GPU stall was injected. The earlier matrix
remains its original observation, not a claim that every row was remeasured after the correction.

## Verification and release boundary

All components of `pnpm check` pass on the final source: typecheck, **161 CPU tests**, **178 GPU
tests**, Vite and library builds, with zero skips or failures. Components ran sequentially at nice
15, with GPU file concurrency one; the final GPU suite took about 131 seconds. Browser playback
was paused during the gate. Collision runtime, defaults, independent normals and warm starts are
unchanged from main. The broader pre-isolation research gate is historical evidence, not this
release branch's test count.

Production browser checks covered preset/scene selection, reset, Play/Pause, the 36-body spawn cap
and disabled Drop at capacity, contact visualization, a body-drag gesture, released-viewer orbit and
Frame, responsive layouts at 390×844 and the native narrow pane, and automatic capture pause.
The original-settings gear reference diagnostic completed six simulated seconds, populated its
metric table, paused at 360 frames, restarted to zero and disabled FPS measurement while enabled.
The Save JSON download completed and its parsed contents matched the on-page report. Inspected
warning/error logs were empty. The settled bunny pile's phone framing was checked again after
its fix. Temporary viewport overrides were reset; the final demo is left paused.

The in-app surface continued reporting `document.visibilityState = visible` when another pane tab
was opened or the browser pane was hidden. That route cannot verify the native hidden-tab event;
no such runtime verification is claimed. The existing visibility handler remains in place. A
physical mobile device, other GPU/browser combinations and sustained thermal performance were
not measured. These are explicit limits of the bounded acceptance, not performance guarantees.

The previously checked local npm tarball installs into a fresh project with root/SDF imports,
a bounded CPU mesh bake and TypeScript public-API checks passing. Library sources did not change
in this browser pass. Both final production pages and all 93 distinct referenced assets, including
lazy scene data, return HTTP 200 and match the built files byte for byte (95 files total).

Known collision-quality limits remain: local optimization can miss contact regions, especially at
low sample budgets; transient penetration and mesh settling are not universally eliminated. Do
not describe this as penetration-free, paper-GPU-equivalent or device-independent 60 FPS.
API and capacity limits remain in [SDF_WORLD.md](SDF_WORLD.md).

The bounded production-browser/FPS acceptance is complete and the isolated demo change is ready
for review. Git publication/merge status is tracked by its PR; no deployment or npm publication
was performed during this verification. Keep large-pile
benchmark data, rejected search backends and C4 redesign outside this demo-polish change.

[Raw reports, build/source provenance and gate results](measurements/sdf-demo-release-2026-10-24.json).
