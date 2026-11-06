# Upstream golden data and colored-GS prototype

Headless C++ harness around the vendored upstream solver (`savant117-avbd-demo3d/avbd-demo3d-main`).
It exists to give the C# reference solver an oracle that does not depend on anyone's memory of the algorithm.

| File | Purpose |
|---|---|
| `build.sh` | Builds `driver_up` (unmodified upstream) and `driver_colored` (upstream with the colored step). Needs `g++` and `python3`. |
| `driver.cpp` | `golden` / `jitter` modes dump body states per step; `compare` mode (colored build) runs sequential and colored side by side. |
| `colored_step.cpp` | `Solver::step()` with a greedy first-fit graph-colored primal sweep. The C# `GaussSeidelMode.Colored` mirrors it. |
| `generate_golden.py` | Writes `Packages/com.mckrill.avbd/Tests/Reference/UpstreamGolden.cs`. |

```sh
OUT=$(Tools/UpstreamGolden/build.sh)
python3 Tools/UpstreamGolden/generate_golden.py "$OUT/driver_up" Packages/com.mckrill.avbd/Tests/Reference/UpstreamGolden.cs
"$OUT/driver_colored" compare stack 240      # sequential vs colored on one scene
```

Scenes for `driver`: `ground`, `stack`, `stackratio`, `rope`, `spring`, `pyramid`.
Rerun `generate_golden.py` whenever the upstream sources or solver parameters change.
