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

Units that differ between the runs are converted to the baseline's unit. A group with a single sample (`Build scene`, `Step, first`) is noisy; look at the sampled groups first.
