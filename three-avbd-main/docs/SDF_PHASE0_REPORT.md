# SDF phase 0 — CPU-only completion report

2026-10-14. Work is on `codex/sdf-oracle-phase0`, uncommitted and unpushed.

## Delivered

- Field/gradient fixtures: 24 shapes × 175 points = 4,200 samples.
- Optimizer fixtures: 16 cases.
- Manifold/cache fixtures: 21 updates, with initial cache, complete collision inputs,
  exact PCG draws, ranked outputs and counters.
- Curvature fixtures: all 24 finalized, ordered seed sets with scale-dependent
  thresholds and mass properties.
- Scene baselines: 20 runs × 360 frames = 7,200 frames, with contact/query/iteration,
  kinetic-energy and penetration metrics. Includes independent vertex probes.
- 71 provenance-checked JSON files, approximately 3.8 MiB.

Everything comes from the pinned upstream C++ implementation. Only asset-path/error
handling is adapted to run the released scenes without SDL or graphics. The engine
itself has not gained SDF support yet; phase 1 is the CPU reference port.

## Validation

| Check | Result |
|---|---|
| Independent C++ rebuild and exact fixture reproduction | Passed |
| TypeScript typecheck | Passed |
| CPU suite, test concurrency 1 | 89 passed, 0 failed, 0 skipped |
| Vite build | Passed; existing large-chunk warning |
| Library build | Passed |
| Whitespace diff check | Passed |
| GPU tests / full `pnpm check` | Deferred at user request; not a pass |

Paper scene counts and GPU timings are not reproduced: these are released-demo
CPU sequential-impulse baselines. The README explains the different scene/settings
and counter conventions. Fig. 6/7 experiment reconstruction and target-hardware
benchmarks remain future validation, separate from oracle capture.

## Compute contention observations

One harness/compiler/test command at a time. CPU tests were explicitly serialized.
No GPU test, rendering context, or GPU benchmark was launched. No other task's
process, checkout, or server was modified.

19 sampled one-second CPU intervals between 15:06:58 and 15:12:20 PDT:

- CPU idle: minimum 49.66%, median 62.31%, maximum 77.67%.
- Swap-in/out counters remained zero throughout.
- System unused memory ranged about 4.0–6.9 GiB; this is aggregate system memory,
  not an allocation attributable to either task.
- Initial `nice` calls were rejected by the sandbox (commands still ran serially).
  After detecting that, niceness 15 was verified and used for the final oracle
  verification, CPU suite, typecheck and both builds.

| Final operation | Wall time | Maximum child RSS |
|---|---:|---:|
| Oracle rebuild + all fixtures verified | 23.47 s | 143.1 MiB |
| Typecheck | 2.35 s | 574.3 MiB |
| CPU tests | 21.28 s | 290.5 MiB |
| Vite build | 0.57 s | 252.8 MiB |
| Library build | 1.06 s | 360.6 MiB |

RSS is the maximum child process resident set reported by macOS, not a sum of
concurrent allocations. CPU sampling is intermittent and may miss brief spikes.
There was no observed CPU saturation or memory swapping. The other task's own
throughput/latency was not measured, so this does **not** prove zero interference.

Temporary raw logs are local operational evidence and intentionally not
checked into the repository.

## Next

1. Implement CPU SDF fields/optimizer against these goldens, then manifold/cache
   logic with the captured seed sets and RNG draws as fixed inputs.
2. Define explicit f32/near-tie tolerances before comparing a port; never regenerate
   the upstream goldens to make a port pass.
3. Run the deferred GPU/full gate when GPU use is authorized. GPU implementation,
   baking/API, broadphase scaling and device benchmarks follow the CPU reference.
