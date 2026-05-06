#!/usr/bin/env python3
"""
Rebuild EX3 correctness data from the real `results/correctness/<bench>/EX3/correctness.jsonl`
records (capture + check stages), at per-(benchmark, model, version, dataset) granularity.

A (bench, model, version, dataset) is considered correct iff:
  - its capture record has status='ok' (binary ran without crashing), AND
  - its corresponding check record has status='pass' (output matches reference).

Outputs:
  - analysis_summaries/correctness/EX3_correctness_real.csv          per-dataset granularity
  - analysis_summaries/correctness/EX3_correctness_real_aggregated.csv   collapsed to (bench, model, version) for back-compat with build_unified_dataframe_ex3.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORR_DIR = PROJECT_ROOT / "results" / "correctness"
OUT_DIR  = PROJECT_ROOT / "analysis_summaries" / "correctness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# file_name pattern: output_<model>_v<version>_<dataset>_output.<ext>
FILENAME_RE = re.compile(r"output_(?P<model>[^_]+)_v(?P<version>\d+)_(?P<dataset>[a-zA-Z\-]+)_output")


def parse_check_file(file_name: str):
    m = FILENAME_RE.search(file_name or "")
    if not m:
        return None
    return m.group("model"), int(m.group("version")), m.group("dataset")


def main():
    # Map (bench, model, version, dataset) -> dict of evidence
    cells: dict = {}

    benches = sorted(p.parent.parent.name for p in CORR_DIR.glob("*/EX3/correctness.jsonl"))

    for bench in benches:
        f = CORR_DIR / bench / "EX3" / "correctness.jsonl"
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            stage = rec.get("stage")
            if stage == "capture":
                model = rec.get("model")
                version = rec.get("version")
                dataset = rec.get("dataset")
                if model is None or version is None or model == "baseline":
                    continue
                key = (bench, model, int(version), dataset)
                slot = cells.setdefault(key, {})
                slot["capture_status"] = rec.get("status")
                slot["capture_returncode"] = rec.get("returncode")
                slot["capture_stderr"] = rec.get("stderr") or ""
            elif stage == "check":
                parsed = parse_check_file(rec.get("file_name") or "")
                if not parsed:
                    continue
                model, version, dataset = parsed
                if model == "baseline":
                    continue
                key = (bench, model, version, dataset)
                slot = cells.setdefault(key, {})
                slot["check_status"] = rec.get("status")
                slot["check_message"] = rec.get("message", "")

    # Resolve correctness verdict per cell
    rows_per_dataset = []
    for (bench, model, version, dataset), ev in sorted(cells.items()):
        cap = ev.get("capture_status")
        chk = ev.get("check_status")
        # correct iff capture ok AND check pass
        correct = (cap == "ok") and (chk == "pass")
        if cap == "ok" and chk is None:
            verdict_note = "no_check_record"
        elif cap == "ok" and chk == "pass":
            verdict_note = "passed"
        elif cap == "ok" and chk == "fail":
            verdict_note = "wrong_output"
        elif cap == "ok" and chk == "skip":
            verdict_note = "check_skipped"
        elif cap == "error":
            verdict_note = "runtime_error"
        else:
            verdict_note = f"cap={cap}|chk={chk}"
        rows_per_dataset.append({
            "benchmark": bench,
            "model": model,
            "version": version,
            "dataset": dataset,
            "correctness": "true" if correct else "false",
            "capture_status": cap or "",
            "check_status": chk or "",
            "note": verdict_note,
        })

    out_per = OUT_DIR / "EX3_correctness_real.csv"
    with out_per.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["benchmark","model","version","dataset","correctness","capture_status","check_status","note"])
        w.writeheader()
        for r in rows_per_dataset:
            w.writerow(r)
    print(f"  wrote {len(rows_per_dataset)} per-dataset rows -> {out_per}")

    # Aggregated to (bench, model, version): correct iff ANY dataset is correct (same convention as old summary)
    agg = {}
    for r in rows_per_dataset:
        k = (r["benchmark"], r["model"], r["version"])
        slot = agg.setdefault(k, {"any_correct": False, "any_seen": True})
        if r["correctness"] == "true":
            slot["any_correct"] = True

    out_agg = OUT_DIR / "EX3_correctness_real_aggregated.csv"
    with out_agg.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["benchmark","model","version","correctness","note"])
        for (bench, model, version), s in sorted(agg.items()):
            w.writerow([bench, model, version, "true" if s["any_correct"] else "false", "real_data"])
    print(f"  wrote {len(agg)} aggregated rows -> {out_agg}")

    # Quick stats
    n_total = len(rows_per_dataset)
    n_correct = sum(1 for r in rows_per_dataset if r["correctness"] == "true")
    print(f"\nPer-dataset cells: {n_total}  ({n_correct} correct, {100*n_correct/max(n_total,1):.1f}%)")
    n_agg_total = len(agg)
    n_agg_correct = sum(1 for s in agg.values() if s["any_correct"])
    print(f"Aggregated: {n_agg_total}  ({n_agg_correct} correct, {100*n_agg_correct/max(n_agg_total,1):.1f}%)")


if __name__ == "__main__":
    main()
