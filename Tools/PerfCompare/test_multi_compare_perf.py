#!/usr/bin/env python3
"""Tests of multi_compare_perf.py with synthetic result files:  python3 -I Tools/PerfCompare/test_multi_compare_perf.py"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_perf as cp  # noqa: E402
import multi_compare_perf as mcp  # noqa: E402

MS, COUNT = 2, 8


def group(name, samples, unit=MS, increase_is_better=False):
    s = sorted(samples)
    n = len(s)
    mid = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    mean = sum(s) / n
    return {"Name": name, "Unit": unit, "IncreaseIsBetter": increase_is_better, "Samples": list(samples), "Min": s[0], "Max": s[-1], "Median": mid,
            "Average": mean, "StandardDeviation": (sum((x - mean) ** 2 for x in s) / n) ** 0.5, "Sum": sum(s)}


def write_run(folder, filename, tests):
    """tests: {full test name: (status, [SampleGroup dicts])}. Writes a minimal NUnit result file."""
    cases = []
    for name, (status, groups) in tests.items():
        line = "##performancetestresult2:" + json.dumps({"Name": name, "SampleGroups": groups})
        cases.append('<test-case fullname="%s" name="%s" result="%s"><output>%s</output></test-case>' % (name, name.split(".")[-1], status, escape(line)))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?><test-run><test-suite>' + "".join(cases) + "</test-suite></test-run>")


def run_cli(*args):
    out = subprocess.run([sys.executable, os.path.join(HERE, "multi_compare_perf.py")] + list(args), capture_output=True, text=True)
    return out.returncode, out.stdout, out.stderr


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class MultiCompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = os.path.join(self.tmp.name, "base")
        self.cand = os.path.join(self.tmp.name, "cand")
        self.csv = os.path.join(self.tmp.name, "out.csv")

    def test_every_field_is_the_median_over_the_runs(self):
        T = "Ns.Tests.Scenario1"
        for i, samples in enumerate(([10, 10, 10], [12, 12, 12], [100, 100, 100])):   # the median run is the 12 ms one, the 100 ms run is an outlier
            write_run(self.base, "r%d.xml" % i, {T: ("Passed", [group("Step, window", samples)])})
        for i, samples in enumerate(([20, 20, 20], [22, 22, 22], [24, 24, 24])):
            write_run(self.cand, "r%d.xml" % i, {T: ("Passed", [group("Step, window", samples)])})
        code, out, err = run_cli(self.base, self.cand, "-o", self.csv)
        self.assertEqual(code, 0, err)
        (row,) = read_csv(self.csv)
        self.assertEqual(float(row["baseline_median"]), 12)
        self.assertEqual(float(row["baseline_avg"]), 12)
        self.assertEqual(float(row["baseline_sum"]), 36)
        self.assertEqual(float(row["baseline_min"]), 12)
        self.assertEqual(float(row["baseline_max"]), 12)
        self.assertEqual(float(row["candidate_median"]), 22)
        self.assertAlmostEqual(float(row["delta_median_pct"]), 100.0 * 10 / 12, places=3)
        self.assertEqual(row["verdict"], "worse")
        self.assertEqual(row["baseline_samples"], "3")
        self.assertIn("median of 3 baseline runs and 3 candidate runs", out)
        self.assertIn("worse", out)

    def test_the_columns_are_those_of_compare_perf(self):
        T = "Ns.Tests.Scenario1"
        write_run(self.base, "a.xml", {T: ("Passed", [group("Step, window", [1, 2, 3])])})
        write_run(self.cand, "a.xml", {T: ("Passed", [group("Step, window", [1, 2, 3])])})
        header, _ = cp.compare(*(mcp.read_folder(f)[0] for f in (self.base, self.cand)), 5.0)
        header_single, _ = cp.compare(cp.read_run(os.path.join(self.base, "a.xml")), cp.read_run(os.path.join(self.cand, "a.xml")), 5.0)
        self.assertEqual(header, header_single)

    def test_a_folder_of_one_file_gives_the_csv_of_compare_perf(self):
        T1, T2 = "Ns.Tests.Scenario1", "Ns.Tests.Scenario2"
        write_run(self.base, "a.xml", {T1: ("Passed", [group("Build scene", [900]), group("Step, window", [5, 6, 7, 8]), group("Bodies", [100], COUNT)]),
                                        T2: ("Passed", [group("Step, window", [1, 1, 2])])})
        write_run(self.cand, "a.xml", {T1: ("Passed", [group("Build scene", [950]), group("Step, window", [4, 5, 6, 6]), group("Bodies", [100], COUNT)]),
                                        T2: ("Failed", [group("Step, window", [1, 1, 2])])})
        single = os.path.join(self.tmp.name, "single.csv")
        done = subprocess.run([sys.executable, os.path.join(HERE, "compare_perf.py"), os.path.join(self.base, "a.xml"),
                               os.path.join(self.cand, "a.xml"), "-o", single], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        code, out, err = run_cli(self.base, self.cand, "-o", self.csv)
        self.assertEqual(code, 0, err)
        with open(single, encoding="utf-8") as a, open(self.csv, encoding="utf-8") as b:
            self.assertEqual(a.read(), b.read())

    def test_the_sample_count_is_the_median_count_and_stays_whole(self):
        T = "Ns.Tests.Scenario1"
        for i, n in enumerate((10, 11)):       # two runs: the lower count
            write_run(self.base, "r%d.xml" % i, {T: ("Passed", [group("Step, window", [5.0] * n)])})
        write_run(self.cand, "r0.xml", {T: ("Passed", [group("Step, window", [5.0] * 10)])})
        code, out, err = run_cli(self.base, self.cand, "-o", self.csv)
        self.assertEqual(code, 0, err)
        (row,) = read_csv(self.csv)
        self.assertEqual(row["baseline_samples"], "10")
        self.assertEqual(float(row["baseline_sum"]), 52.5)   # the median of 50 and 55

    def test_runs_in_different_units_are_converted(self):
        T = "Ns.Tests.Scenario1"
        write_run(self.base, "a.xml", {T: ("Passed", [group("Step, window", [10.0, 10.0], MS)])})
        write_run(self.base, "b.xml", {T: ("Passed", [group("Step, window", [12000.0, 12000.0], 1)])})   # 12 ms in microseconds
        write_run(self.base, "c.xml", {T: ("Passed", [group("Step, window", [11.0, 11.0], MS)])})
        write_run(self.cand, "a.xml", {T: ("Passed", [group("Step, window", [11.0, 11.0], MS)])})
        code, out, err = run_cli(self.base, self.cand, "-o", self.csv)
        self.assertEqual(code, 0, err)
        (row,) = read_csv(self.csv)
        self.assertEqual(row["unit"], "ms")
        self.assertAlmostEqual(float(row["baseline_median"]), 11.0)
        self.assertEqual(row["verdict"], "same")

    def test_tests_and_groups_missing_from_some_runs_use_the_runs_that_have_them(self):
        T1, T2 = "Ns.Tests.Scenario1", "Ns.Tests.Scenario2"
        write_run(self.base, "a.xml", {T1: ("Passed", [group("Step, window", [4.0, 4.0])]), T2: ("Passed", [group("Step, window", [8.0, 8.0])])})
        write_run(self.base, "b.xml", {T1: ("Passed", [group("Step, window", [6.0, 6.0]), group("Extra", [1.0, 1.0])])})
        write_run(self.cand, "a.xml", {T1: ("Passed", [group("Step, window", [5.0, 5.0])])})
        code, out, err = run_cli(self.base, self.cand, "-o", self.csv)
        self.assertEqual(code, 0, err)
        self.assertIn("Scenario2 is in 1 of 2 runs", err)
        self.assertIn("Extra is in 1 of 2 runs", err)
        rows = {(r["test"], r["group"]): r for r in read_csv(self.csv)}
        self.assertEqual(float(rows[(T1, "Step, window")]["baseline_median"]), 5.0)
        self.assertEqual(rows[(T2, "Step, window")]["verdict"], "only in baseline")
        self.assertEqual(rows[(T1, "Extra")]["verdict"], "only in baseline")

    def test_status_lists_the_statuses_when_the_runs_disagree(self):
        T = "Ns.Tests.Scenario1"
        write_run(self.base, "a.xml", {T: ("Passed", [group("Step, window", [4.0])])})
        write_run(self.base, "b.xml", {T: ("Failed", [group("Step, window", [4.0])])})
        write_run(self.cand, "a.xml", {T: ("Passed", [group("Step, window", [4.0])])})
        code, out, err = run_cli(self.base, self.cand, "-o", self.csv)
        self.assertEqual(code, 0, err)
        (row,) = read_csv(self.csv)
        self.assertEqual(row["baseline_status"], "Failed/Passed")
        self.assertEqual(row["candidate_status"], "Passed")

    def test_other_files_in_the_folder_are_ignored_and_the_folder_is_not_searched_recursively(self):
        T = "Ns.Tests.Scenario1"
        write_run(self.base, "a.xml", {T: ("Passed", [group("Step, window", [4.0, 4.0])])})
        write_run(os.path.join(self.base, "old"), "b.xml", {T: ("Passed", [group("Step, window", [400.0, 400.0])])})
        with open(os.path.join(self.base, "notes.txt"), "w") as f:
            f.write("not a result")
        write_run(self.cand, "a.xml", {T: ("Passed", [group("Step, window", [4.0, 4.0])])})
        code, out, err = run_cli(self.base, self.cand, "-o", self.csv)
        self.assertEqual(code, 0, err)
        self.assertIn("median of 1 baseline run and 1 candidate run", out)
        (row,) = read_csv(self.csv)
        self.assertEqual(float(row["baseline_median"]), 4.0)

    def test_errors_for_a_missing_folder_an_empty_folder_and_a_file_that_is_not_a_result(self):
        T = "Ns.Tests.Scenario1"
        write_run(self.cand, "a.xml", {T: ("Passed", [group("Step, window", [4.0])])})
        code, out, err = run_cli(os.path.join(self.tmp.name, "missing"), self.cand, "-o", self.csv)
        self.assertNotEqual(code, 0)
        self.assertIn("not a folder", err)

        empty = os.path.join(self.tmp.name, "empty")
        os.makedirs(empty)
        code, out, err = run_cli(empty, self.cand, "-o", self.csv)
        self.assertNotEqual(code, 0)
        self.assertIn("no .xml result files", err)

        os.makedirs(self.base)
        with open(os.path.join(self.base, "bad.xml"), "w") as f:
            f.write('<?xml version="1.0"?><test-run><test-suite><test-case fullname="x" result="Passed"><output>nothing</output></test-case></test-suite></test-run>')
        code, out, err = run_cli(self.base, self.cand, "-o", self.csv)
        self.assertNotEqual(code, 0)
        self.assertIn("bad.xml", err)

    def test_the_console_lists_sum_changes_of_the_medians_like_compare_perf(self):
        T = "Ns.Tests.Scenario1"
        for i, v in enumerate((10.0, 10.0, 10.0)):
            write_run(self.base, "r%d.xml" % i, {T: ("Passed", [group("Step, window", [v] * 4)])})
        for i, v in enumerate((20.0, 20.0, 1.0)):
            write_run(self.cand, "r%d.xml" % i, {T: ("Passed", [group("Step, window", [v] * 4)])})
        code, out, err = run_cli(self.base, self.cand, "-o", self.csv)
        self.assertEqual(code, 0, err)
        self.assertIn("sum worse", out)
        self.assertIn("sum 40.00 -> 80.00 ms", out)


if __name__ == "__main__":
    unittest.main()
