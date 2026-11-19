#!/usr/bin/env python3
"""Compares two Unity Performance Testing runs (NUnit XML result files) and writes one CSV row per test and sample group.

    python3 Tools/PerfCompare/compare_perf.py baseline.xml candidate.xml -o comparison.csv [--threshold 5]

The numbers come from the "##performancetestresult2:{...}" JSON lines that the Performance Testing API writes into the output of every test case
(the same lines also appear in the suite's own output, so the script reads the test cases only). Tests are matched by their full name.
Python 3 standard library only.

Columns: test, group, unit, kind, baseline/candidate status, then for the median, average, sum, min, max, standard deviation and sample count the value in
each run, and delta_median / delta_median_pct / delta_avg_pct (candidate minus baseline, in the baseline's unit), and a verdict.
On the console, every time group whose median moved by more than the threshold is listed, and so is every time group whose sum (baseline_sum to
candidate_sum) moved by more than the threshold, with the difference in the group's unit; the sum depends on the number of samples, so a different
count is shown next to it.

Verdict: 'better' or 'worse' when the median moved by more than --threshold percent (the direction follows the group's IncreaseIsBetter flag), 'same'
otherwise. A group that is a plain count (unit Undefined: bodies, joints, manifold slots, ...) is 'same' or 'changed'. A test or group that is in only one
run gets 'only in baseline' or 'only in candidate'.
"""
import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET

MARKER = "##performancetestresult2:"

# SampleUnit of the Performance Testing API: (name, family, size in the family's base unit).
UNITS = {
    0: ("ns", "time", 1e-9), 1: ("us", "time", 1e-6), 2: ("ms", "time", 1e-3), 3: ("s", "time", 1.0),
    4: ("B", "size", 1.0), 5: ("KB", "size", 1e3), 6: ("MB", "size", 1e6), 7: ("GB", "size", 1e9),
    8: ("", "count", 1.0),
}


def unit_info(code):
    return UNITS.get(code, ("unit" + str(code), "other", 1.0))


def read_run(path):
    """Returns {test name: {"status": str, "groups": {group name: SampleGroup dict}, "order": [group names]}} in file order."""
    root = ET.parse(path).getroot()
    tests = {}
    for case in root.iter("test-case"):
        name = case.get("fullname") or case.get("name")
        result = {"status": case.get("result", ""), "groups": {}, "order": []}
        output = case.find("output")
        text = output.text if output is not None and output.text else ""
        for line in text.splitlines():
            if not line.startswith(MARKER):
                continue
            data = json.loads(line[len(MARKER):])
            name = data.get("Name") or name
            for g in data.get("SampleGroups", []):  # a later line of the same test replaces an earlier one
                if g["Name"] not in result["groups"]:
                    result["order"].append(g["Name"])
                result["groups"][g["Name"]] = g
        tests[name] = result
    if not tests:
        sys.exit(path + ": no test cases found (is this an NUnit XML result file?)")
    if not any(t["groups"] for t in tests.values()):
        sys.exit(path + ": no '" + MARKER + "' lines in the test outputs (was it run with the Performance Testing package?)")
    return tests


def num(v):
    return "" if v is None else round(v, 4)


def compare(base_tests, cand_tests, threshold):
    header = ["test", "group", "unit", "kind", "baseline_status", "candidate_status",
              "baseline_median", "candidate_median", "delta_median", "delta_median_pct",
              "baseline_avg", "candidate_avg", "delta_avg_pct", "baseline_sum", "candidate_sum",
              "baseline_min", "candidate_min", "baseline_max", "candidate_max",
              "baseline_stddev", "candidate_stddev", "baseline_samples", "candidate_samples", "verdict"]
    rows = []
    names = list(base_tests) + [n for n in cand_tests if n not in base_tests]
    for test in names:
        b, c = base_tests.get(test), cand_tests.get(test)
        groups = list(b["order"]) if b else []
        groups += [g for g in (c["order"] if c else []) if g not in groups]
        if not groups:  # the test exists but has no samples (failed before measuring)
            groups = [""]
        for group in groups:
            gb = b["groups"].get(group) if b else None
            gc = c["groups"].get(group) if c else None
            row = {"test": test, "group": group,
                   "baseline_status": b["status"] if b else "", "candidate_status": c["status"] if c else ""}
            ref = gb or gc
            unit, kind, scale = unit_info(ref["Unit"]) if ref else ("", "", 1.0)
            row["unit"], row["kind"] = unit, kind
            if gb and gc:
                ub, uc = unit_info(gb["Unit"]), unit_info(gc["Unit"])
                if ub[1] != uc[1]:
                    sys.exit("%s / %s: the units of the two runs differ (%s, %s)" % (test, group, ub[0], uc[0]))
                factor = uc[2] / ub[2]  # candidate values in the baseline's unit
            else:
                factor = 1.0
            if gb:
                row.update(baseline_median=gb["Median"], baseline_avg=gb["Average"], baseline_sum=gb["Sum"], baseline_min=gb["Min"], baseline_max=gb["Max"],
                           baseline_stddev=gb["StandardDeviation"], baseline_samples=len(gb["Samples"]))
            if gc:
                row.update(candidate_median=gc["Median"] * factor, candidate_avg=gc["Average"] * factor, candidate_sum=gc["Sum"] * factor, candidate_min=gc["Min"] * factor,
                           candidate_max=gc["Max"] * factor, candidate_stddev=gc["StandardDeviation"] * factor,
                           candidate_samples=len(gc["Samples"]))
            if gb and gc:
                delta = row["candidate_median"] - row["baseline_median"]
                row["delta_median"] = delta
                if row["baseline_median"]:
                    row["delta_median_pct"] = 100.0 * delta / abs(row["baseline_median"])
                if row["baseline_sum"]:
                    row["delta_sum"] = row["candidate_sum"] - row["baseline_sum"]
                    row["delta_sum_pct"] = 100.0 * row["delta_sum"] / abs(row["baseline_sum"])
                    increase_is_better = bool(gb.get("IncreaseIsBetter"))
                    row["sum_verdict"] = "better" if (row["delta_sum"] > 0) == increase_is_better else "worse"
                if row["baseline_avg"]:
                    row["delta_avg_pct"] = 100.0 * (row["candidate_avg"] - row["baseline_avg"]) / abs(row["baseline_avg"])
                if kind == "count":
                    row["verdict"] = "same" if delta == 0 else "changed"
                else:
                    pct = row.get("delta_median_pct", 0.0 if delta == 0 else float("inf") * delta)
                    if abs(pct) <= threshold:
                        row["verdict"] = "same"
                    else:
                        increase_is_better = bool(gb.get("IncreaseIsBetter"))
                        row["verdict"] = "better" if (pct > 0) == increase_is_better else "worse"
            elif gb:
                row["verdict"] = "only in baseline"
            elif gc:
                row["verdict"] = "only in candidate"
            else:
                row["verdict"] = ""
            rows.append(row)
    return header, rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("baseline", help="NUnit XML result file of the baseline run")
    p.add_argument("candidate", help="NUnit XML result file of the run to compare")
    p.add_argument("-o", "--output", default="perf_comparison.csv", help="CSV file to write (default: perf_comparison.csv)")
    p.add_argument("--threshold", type=float, default=5.0, help="percent change of the median below which a group is 'same' (default 5)")
    args = p.parse_args()

    header, rows = compare(read_run(args.baseline), read_run(args.candidate), args.threshold)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([(num(r[h]) if isinstance(r.get(h), float) else r.get(h, "")) for h in header])

    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("%d rows written to %s (%s)" % (len(rows), args.output, ", ".join("%s: %d" % kv for kv in sorted(counts.items()))))
    for r in rows:
        if r["kind"] != "time":
            continue
        name = "%s / %s" % (r["test"].split(".")[-1], r["group"])
        if r["verdict"] in ("better", "worse"):
            print("  %-8s %+7.1f%%  %s  (median %.2f -> %.2f %s)" % (r["verdict"], r["delta_median_pct"], name,
                                                                  r["baseline_median"], r["candidate_median"], r["unit"]))
        # The sum of the samples, when it moved by more than the threshold too. It depends on the number of samples, which is shown when it differs.
        if abs(r.get("delta_sum_pct", 0.0)) > args.threshold:
            samples = "" if r["baseline_samples"] == r["candidate_samples"] else ", samples %d vs %d" % (r["baseline_samples"], r["candidate_samples"])
            print("  sum %-4s %+7.1f%%  %s  (sum %.2f -> %.2f %s, difference %+.2f %s%s)" % (
                r["sum_verdict"], r["delta_sum_pct"], name, r["baseline_sum"], r["candidate_sum"], r["unit"], r["delta_sum"], r["unit"], samples))


if __name__ == "__main__":
    main()
