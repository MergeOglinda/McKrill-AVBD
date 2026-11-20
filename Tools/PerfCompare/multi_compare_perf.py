#!/usr/bin/env python3
"""Compares two sets of Unity Performance Testing runs (folders of NUnit XML result files) and writes one CSV row per test and sample group.

    python3 Tools/PerfCompare/multi_compare_perf.py baseline_folder candidate_folder -o comparison.csv [--threshold 5]

Like compare_perf.py, but each side is a folder with the result files (*.xml) of several runs of the same suite. For every test and sample group, each field
(median, average, sum, min, max, standard deviation) is the median over the runs of that field, and the sample count is the median (the lower one when
the number of runs is even, so it stays a whole number) of the counts. The comparison, the CSV columns, the verdicts and the console output are those of
compare_perf.py, applied to these medians, so one run of noise does not move a result. A folder with a single file gives the same result as compare_perf.py.

A test or group that some runs of a folder lack is aggregated over the runs that have it (a warning on stderr says so). The status columns hold the status
of the test in the runs; when the runs disagree they list them all, for example 'Failed/Passed'. The folder is not searched recursively.
Python 3 standard library only.
"""
import argparse
import os
import statistics
import sys

import compare_perf as cp


def result_files(folder):
    """The XML files directly in the folder, sorted by name. A single file is accepted as a folder of one run."""
    if os.path.isfile(folder):
        return [folder]
    if not os.path.isdir(folder):
        sys.exit(folder + ": not a folder")
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".xml") and os.path.isfile(os.path.join(folder, f)))
    if not files:
        sys.exit(folder + ": no .xml result files in the folder")
    return files


def median_group(groups):
    """One SampleGroup dict whose fields are the medians of the fields of the given ones (all of the same test and group, one per run).

    The values are converted to the unit of the first group, in case the runs were written in different units of the same kind."""
    first = groups[0]
    unit, kind, scale = cp.unit_info(first["Unit"])
    values = {key: [] for key in ("Median", "Average", "Sum", "Min", "Max", "StandardDeviation")}
    for g in groups:
        u = cp.unit_info(g["Unit"])
        if u[1] != kind:
            sys.exit("%s: the units of the runs differ (%s, %s)" % (first["Name"], unit, u[0]))
        for key in values:
            values[key].append(g[key] if u[2] == scale else g[key] * u[2] / scale)  # unchanged values keep their type, so one run prints as in compare_perf.py
    merged = {"Name": first["Name"], "Unit": first["Unit"], "IncreaseIsBetter": first.get("IncreaseIsBetter", False)}
    for key, v in values.items():
        merged[key] = statistics.median(v)
    merged["SampleCount"] = statistics.median_low([cp.sample_count(g) for g in groups])
    return merged


def merge_runs(runs, label):
    """Combines the runs (each as returned by compare_perf.read_run) into one in the same form, with the median of every field."""
    names = []
    for run in runs:
        names += [n for n in run if n not in names]
    merged = {}
    for test in names:
        have = [run[test] for run in runs if test in run]
        if len(have) < len(runs):
            print("warning: %s: %s is in %d of %d runs" % (label, test, len(have), len(runs)), file=sys.stderr)
        order = []
        for r in have:
            order += [g for g in r["order"] if g not in order]
        groups = {}
        for group in order:
            present = [r["groups"][group] for r in have if group in r["groups"]]
            if len(present) < len(have):
                print("warning: %s: %s / %s is in %d of %d runs" % (label, test, group, len(present), len(have)), file=sys.stderr)
            groups[group] = median_group(present)
        statuses = []
        for r in have:
            if r["status"] not in statuses:
                statuses.append(r["status"])
        merged[test] = {"status": "/".join(sorted(statuses)), "groups": groups, "order": order}
    return merged


def read_folder(folder):
    files = result_files(folder)
    return merge_runs([cp.read_run(f) for f in files], folder), len(files)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("baseline", help="folder with the NUnit XML result files of the baseline runs")
    p.add_argument("candidate", help="folder with the NUnit XML result files of the runs to compare")
    p.add_argument("-o", "--output", default="perf_comparison.csv", help="CSV file to write (default: perf_comparison.csv)")
    p.add_argument("--threshold", type=float, default=5.0, help="percent change of the median below which a group is 'same' (default 5)")
    args = p.parse_args()

    baseline, base_runs = read_folder(args.baseline)
    candidate, cand_runs = read_folder(args.candidate)
    header, rows = cp.compare(baseline, candidate, args.threshold)
    cp.write_csv(args.output, header, rows)
    print("median of %d baseline run%s and %d candidate run%s" % (base_runs, "" if base_runs == 1 else "s", cand_runs, "" if cand_runs == 1 else "s"))
    cp.print_summary(rows, args.output, args.threshold)


if __name__ == "__main__":
    main()
