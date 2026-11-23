# PerfCompare

`compare_perf.py` compares two runs of the performance tests (`Tests/Performance`, Performance Testing API) and writes a CSV with one row per test and sample group (for example `Scenario5_BrickRing110k` / `Step, window`).

```
python3 Tools/PerfCompare/compare_perf.py baseline.xml candidate.xml -o comparison.csv [--threshold 5]
```

Both inputs are the NUnit XML result files that Unity's Test Runner exports (*Export results*). The script reads the `##performancetestresult2:` JSON lines in the output of each test case and matches tests by full name. Python 3, standard library only.

Per row: the median, average, sum, min, max, standard deviation and number of samples of both runs, the change of the median (absolute and in percent of the baseline) and of the average, and a verdict:

- `better` / `worse`: the median moved by more than `--threshold` percent (default 5). The direction follows the group's `IncreaseIsBetter` flag, so for the time groups a lower median is `better`.
- `same`: within the threshold.
- Counters without a unit (bodies, joints, manifold slots, overflow) are `same` or `changed`.
- `only in baseline` / `only in candidate`: the test or group is missing from one run.

The console lists the time groups (except `Build scene` and `Step, first`, which are only in the CSV) whose median moved by more than the threshold, and, on separate `sum` lines, those with more than one sample (in both runs) whose sum of the samples moved by more than the threshold (with the difference in the group's unit). The sum depends on the number of samples, so the console shows the sample counts when they differ.

Units that differ between the runs are converted to the baseline's unit. A group with a single sample (`Build scene`, `Step, first`) is noisy; look at the sampled groups first.

## Several runs per side: `multi_compare_perf.py`

```
python3 Tools/PerfCompare/multi_compare_perf.py baseline_folder candidate_folder -o comparison.csv [--threshold 5]
```

The same comparison, but each side is a **folder** with the result files (`*.xml`, not searched recursively) of several runs of the suite. For every test and sample group, each field (median, average, sum, min, max, standard deviation) is the **median over the runs** of that field, and the sample count is the median count (the lower one for an even number of runs, so it stays whole). The CSV columns, the verdicts, the threshold and the console output are those of `compare_perf.py`, applied to those medians, so one noisy run does not move a result. A folder with one file gives the same CSV as `compare_perf.py`; a single file path is accepted in place of a folder.

- The console starts with the number of runs on each side.
- A test or group that some runs of a folder lack is aggregated over the runs that have it, with a warning on stderr.
- `baseline_status` / `candidate_status` hold the test's status in the runs; if the runs disagree they list all of them (`Failed/Passed`).
- A file that is not a performance result (no test cases or no `##performancetestresult2:` lines) stops the script and is named in the message.
- It imports `compare_perf.py` from the same folder, so keep the two together.

Tests of both: `python3 -I Tools/PerfCompare/test_multi_compare_perf.py` (synthetic result files).

## Automated A/B runs: `run_perf_ab.ps1`

```
.\Tools\PerfCompare\run_perf_ab.ps1 -Baseline origin/main -Candidate origin/my-branch [-Runs 8] [-Warmup] [-TestFilter "Scenario5_BrickRing110k"]
```

Windows PowerShell 5.1 or PowerShell 7, with git, Python 3 and the Unity editor version of the project installed (Unity Hub folder, or `-UnityPath`). The script:

1. resolves both refs and checks each out in its own git worktree under `-WorkRoot` (default: `mckrill-avbd-perf` next to the repository), so each side is a separate Unity project with its own `Library`; keep the folder, because the second invocation then skips the long first import. The worktrees are scratch folders: local changes in them are discarded;
2. imports each project once in batch mode (no tests), which stops the script early when a side does not compile;
3. runs the suite `-Runs` times per side (default 8), one Unity process per run, alternating who goes first (A B, B A, ...). `-Warmup` adds a discarded run per side first;
4. runs `multi_compare_perf.py` on the two result folders.

The output goes to `<OutRoot>\<timestamp>\`: `baseline\runNN.xml` and `.log`, `candidate\...`, `meta.json` (refs, commits, GPU, editor) and `comparison.csv`. A run without a result file, a result with a failed test or with another number of test cases than `-ExpectedTests` (default 5) is listed at the end and makes the exit code 1. The tests need a GPU: the script never passes `-nographics`, and the machine should be idle while it runs. Use `origin/<branch>` to take the remote state of a branch (the script fetches first unless `-NoFetch`); a plain name is the local branch.
