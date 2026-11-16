# SDF research reconciliation — STE-3004

R3 follows the small-demo/package preparation release. The deployed gear demo is PR #42,
merged as `005c16bb336b18e8f21c0269dd0018f0169d6391`. Research reconciliation starts from
that main revision; rejected C4 experiments and unrelated dirty work remain separate.

## Exact source heads and disposition

| PR | Retained head | Scope | Current disposition |
|---|---|---|---|
| [#36](https://github.com/sbobyn/three-avbd/pull/36) | `e6753450f050c4d262ff3fe9e9237fb6f7da8442` | Mesh-witness ambiguity filter; old settling ensemble | Blocked and excluded from the runtime candidate. The original rule erases valid opposing gap contacts; the narrowed rule regresses chair seed 5. Historical records remain for provenance, not current acceptance. |
| [#37](https://github.com/sbobyn/three-avbd/pull/37) | `e584004213b4c60cab086162ee312589040e9374` | 4,314-bunny measurement harness and corrected records | Reconciled locally against current main. Historical witness-enabled measurements depend on the older #36 rule; the harness itself does not. Fresh matched acceptance and final sequential gate pass; local candidate is ready for authorized publication. |
| [#38](https://github.com/sbobyn/three-avbd/pull/38) | `c748661fa4a680384d41a38c8a787b7121c4c808` | Ablations, dense diagnostic backends, timing records, history/seed-cache optimization | Reconciled locally after #37. Main's public API/gear guidance and historical research records are preserved. Cache/history changes do not depend on #36; no witness-direction filter is imported. Fresh matched acceptance, final sequential gate and independent review pass. |

No research PR has been merged, pushed or force-pushed by this checkpoint. The final runtime
candidate contains #37/#38 only and preserves main's supplemental-contact semantics. The
new two-flank regression protects valid opposing contacts within the same non-convex pair.

## Combined-candidate checks

The first combined source tree `bb69d628003a058ae6f7156245b21fa0caf1404e` passed sequential
`pnpm check`: 166 CPU and 186 real-Dawn GPU tests, zero failures/skips, typecheck and both
builds. Independent GPT-6 Sol static review found no additional issue in #37, #38 or their
interaction with the narrowed #36 rule. That gate did not establish ensemble acceptance:
the subsequent matched run exposed a chair regression. The first candidate is rejected.
These are local candidate reviews; the remote PR heads have not changed.

The matched runner `scripts/sdf-settling-ensemble.ts` freezes tracked source/fixture hashes,
alternates baseline/candidate order across seeds, uses identical measurement instrumentation,
and rejects adapter changes. Each worker runs two 360-frame scenes, checks overflow every
substep, drains the queue and idles between frames; workers run sequentially with a timeout.
The instrumentation includes geometric probes and CPU readbacks, so these runs are settling
evidence, not playback-performance measurements. The adapter guard was added after independent
review; re-review of staged tree `602dfc1a338788557f4b116db06edd22a6ae4cc7` was clean.

## New correctness finding

The original filter treated all outside-both opposing optimizer points as ambiguous. They
can instead be legitimate gap contacts at a second flank. A two-body regression with two
parallel walls and an offset inner box fails against the old rule: the right witness drops
a valid left contact within the margin. The first candidate removed that arm of the filter,
retaining only the inside-both/aligned-gradient test. Its fresh ensemble then failed chair
seed 5: 0.02848 J final energy versus baseline 0.00170 J, above the unchanged 0.01 J limit.
The sampled final overlap was 4.80 mm, and probes at or above 5 mm increased from 12 to 31.
The end-window geometric witness involves two dynamic chairs, so a floor-only culling
amendment is not established as a fix. A gradient-sign heuristic also does not prove that
an independent contact at another flank is invalid. Neither speculative amendment is shipped.

The safe disposition is to quarantine #36 and retain main's behavior in the #37/#38
candidate. A robust replacement needs pair/contact-level traces and a geometric validity
argument, then the two-flank regression and complete unchanged settling ensemble. The
rejected candidate and raw results are retained; its test requiring contact rejection is
not imported into the candidate that intentionally preserves those contacts.

The complete 11-seed matched run rejects the narrowed rule under the predeclared decision:

| Scene | Failed seeds, main → narrowed | Probes ≥5 mm | Median sampled peak |
|---|---:|---:|---:|
| Chairs | 0 → 1 | 37 → 63 | 11.965 → 4.826 mm |
| Bunnies | 1 → 0 | 14 → 1 | 0.813 → 0.813 mm |

Improving bunnies and chair median depth does not compensate for the new chair failure or
increased count of deep probes. All seeds and the original 360-frame window are retained.
[Raw rejected ensemble](measurements/sdf-r3-rejected-narrowed-2026-10-24.json) contains every
frame sample and source-file hash. The [reconstruction patch](measurements/sdf-r3-rejected-filter-2026-10-24.patch)
recreates that rejected runtime source from the isolated #37/#38 candidate; it is evidence,
not a proposed runtime patch. Baseline manifest SHA-256:
`15dfa4112bedcd96651218528847b2d34c418f2edefbf90754b562364362dfbd`;
rejected candidate: `1c8df8caf68956f3b97e3b2f4f5fb665433e05f206d01c9463444797dd825f74`.

## Isolated #37/#38 matched acceptance

The fresh #37/#38-only run passes the unchanged non-regression decision. All recorded scene
samples match main exactly for all 11 seeds and both scenes: 44 scene runs, 15,840 frame
samples and 3,168 geometric probes. No overflow, non-finite state or GPU validation error
occurred. This is equality of the recorded metrics, not proof that every unrecorded internal
state is identical or that all physical limits pass on every seed.

| Scene | Failed seeds, main → candidate | Probes ≥5 mm | Median sampled peak |
|---|---:|---:|---:|
| Chairs | 0 → 0 | 37 → 37 | 11.965 → 11.965 mm |
| Bunnies | 1 → 1 | 14 → 14 | 0.813 → 0.813 mm |

Bunny seed 3 remains above the energy limit on both sources: 0.05799 J at frame 360 versus
0.01 J. No extended window is used to turn that failure into a pass. The accepted result
preserves baseline behavior; it does not solve this known settling limitation.

[Raw accepted ensemble](measurements/sdf-r3-paper-matched-2026-10-24.json) retains all seeds,
settings, adapter information and source manifests. Candidate source SHA-256:
`245610c008c7824b98e0d1f807c2453395c13368e7a346d06a1d1cafb2e942cc`.
The baseline is main `005c16b`, with the manifest hash recorded above. Use a separate frozen
checkout of that baseline and run from the candidate checkout:

```sh
nice -n 15 node --experimental-transform-types scripts/sdf-settling-ensemble.ts BASELINE_CHECKOUT OUT.json
```

The runner's JSON `complete` and `accepted` fields are the acceptance result; process
completion alone does not establish physical acceptance. Both source roots must remain
unchanged, and the new runner must be tracked before the candidate manifest is collected.
Each worker is capped at 180 seconds, runs alone, and idles 5 ms per frame plus 2 seconds
between workers. GPU runs use real Dawn on M4 Max; no FPS or untested-device claim follows.

## Evidence audit and measurement policy

Historical raw records stay immutable. Their interpretation must identify the retained source,
configuration and scope; measurements on an older solver are not current-main performance.

| Evidence | Retained source / record in #38 | Scope and remaining limit |
|---|---|---|
| Tables 3/4 | `scripts/sdf-ablations.ts`; `docs/SDF_REPRODUCTION_STATUS.md` and its linked records | 18 GPU-AVBD ablations, individual runs, not the paper CPU solver or timing baseline. No-cache also disables SDF multiplier transport. |
| Figure 6 | `scripts/sdf-fig6.ts`, `scripts/sdf-dense-plane.ts`; linked budget/dense records | Six-budget sweep and independent fixed-surface dense analogue. Low-budget transient penetration and inconsistent curvature benefit remain. Dense samples are not the exact paper oracle. |
| Table 2 | `scripts/sdf-table2.ts`; `sdf-table2-full-2026-10-19.json` | 12 historical full-size budgets, reconstructed 4,314-bunny layout, M4 Max, explicitly scoped timestamp region. Record lacks a complete solver-source hash manifest; retain the introducing commit `d80c3ac53e322de174335c9191bd4f3c0b0bdda3` and do not call it a new measurement. |
| Cache/history endpoints | `sdf-performance-endpoints-2026-10-19.json`; retained change `58a6a128c64fc66f5879aa82d85be9d9c6f7d6b0` | Single 4+1 and 16+4 observations, not interleaved statistics. High budget remains about 110 ms collision, not practical 60 FPS. |
| Figure 7 | `scripts/sdf-fig7.ts`, dense backends; `sdf-fig7-piles-2026-10-19.json` | 30 historical runs / six aggregates. All four recorded helper hashes match retained #38 sources. They do not hash the entire solver. Fixed-surface all-contact AVBD analogue, not the unpublished Iron Maiden layout or exact 1,000-optimized-seed baseline. |

See the [retained research ledger](https://github.com/sbobyn/three-avbd/blob/c748661fa4a680384d41a38c8a787b7121c4c808/docs/SDF_REPRODUCTION_STATUS.md)
for raw links and numerical tables. Missing Table 1 layouts, SAT/GJK/EPA/hierarchical
baselines and exact paper CPU timings remain research gaps. M1/GTX1080 measurements are
waived by the user; M4 Max results imply no performance guarantee for those devices.

## Final verification and review

The isolated candidate passes the final sequential `pnpm check`: **166 CPU / 185 real-Dawn
GPU tests**, zero failures and skips, typecheck, Vite build and library build. Existing Vite
chunk-size warnings remain; no test/limit/ignore was weakened. This includes the valid opposing
gap regression, independent normals, identity-based warm starts, cached/direct seed parity,
capacity/lifecycle and gear transmission checks. Runtime sources remained frozen throughout
the accepted ensemble and final gate.

Independent GPT-6 Sol review of staged tree `82dfa125867ac02af1896c97f678211cdade0def`
found no actionable issue in the retained code, evidence or documentation. The data audit
independently recomputed acceptance and verified both source manifests. The rejected-filter
audit independently reproduced its failure summary and reconstructed its exact shader from
the retained patch. The final gate completed after that review. This local branch has no
open PR, so its branch-review reports were not posted against an unrelated remote head.

## Remaining work

1. Keep #36 blocked. A replacement needs pair/contact-level traces, a geometric validity
   argument, the valid two-flank regression and complete unchanged ensemble acceptance.
   Do not merge either the original broad rule or the rejected narrowed rule.
2. Commit/push the reviewed #37/#38 candidate only with authorization; remote research
   heads remain unchanged. Obtain authorization before research merges.
3. R4 measures sustained rendered practical workloads. Historical full-pile timing remains
   historical until remeasured on the reconciled source; search-only timing is not FPS.
   Exact paper layouts/oracles, missing baselines and C4's performance target remain open.
