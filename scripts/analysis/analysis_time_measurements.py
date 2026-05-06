#!/usr/bin/env python3
"""
分析 time_measurement 产出的 JSONL/缺失记录，按 benchmark 汇总性能与缺失组合。

示例：
    python scripts/analysis/analysis_time_measurements.py --benchmarks 3mm --ex-version EX1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DATASET_ORDER = {
    "mini": 0,
    "small": 1,
    "medium": 2,
    "large": 3,
    "extra-large": 4,
    "default": 5,
}


def dataset_rank(name: Optional[str]) -> int:
    if not name:
        return DATASET_ORDER["default"]
    return DATASET_ORDER.get(name, len(DATASET_ORDER))


def parse_list(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    tokens = []
    for chunk in raw.replace(";", ",").split(","):
        item = chunk.strip()
        if item:
            tokens.append(item)
    return tokens or None


def discover_benchmarks(results_root: Path, requested: Optional[Sequence[str]]) -> List[str]:
    if requested is None or any(token.lower() == "all" for token in requested):
        if not results_root.exists():
            return []
        return sorted(p.name for p in results_root.iterdir() if p.is_dir())
    available = set(p.name for p in results_root.iterdir() if p.is_dir()) if results_root.exists() else set()
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"Benchmark(s) not found under {results_root}: {', '.join(missing)}")
    return list(requested)


def load_summary_records(file_path: Path, trails_filter: Optional[Sequence[int]]) -> List[Dict[str, Any]]:
    if not file_path.exists():
        return []
    allowed_trails = set(trails_filter) if trails_filter else None
    exec_trail: Dict[str, int] = {}
    exec_seq: Dict[str, int] = {}
    summaries: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_type = record.get("record_type")
            exe_path = record.get("executable")
            if record_type == "run":
                exec_trail[exe_path] = int(record.get("trail", 0))
            elif record_type == "summary":
                trail = exec_trail.get(exe_path, 0)
                record["trail"] = trail
                record["sequence_id"] = exec_seq.get(exe_path, 0)
                exec_seq[exe_path] = record["sequence_id"] + 1
                if allowed_trails and trail not in allowed_trails:
                    continue
                summaries.append(record)
    return summaries


def load_missing_entries(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        return []
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    missing = payload.get("missing")
    if isinstance(missing, list):
        return missing
    return []


def best_baseline_lookup(records: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, int, Optional[int]], Dict[str, Any]]:
    lookup: Dict[Tuple[str, int, Optional[int]], Dict[str, Any]] = {}
    for entry in records:
        if not entry.get("is_baseline"):
            continue
        dataset = entry.get("dataset") or "default"
        trail = int(entry.get("trail", 0))
        threads = entry.get("threads")
        key = (dataset, trail, threads)
        kernel_stats = entry.get("kernel_stats") or {}
        current_best = lookup.get(key)
        candidate_mean = kernel_stats.get("mean")
        if candidate_mean is None:
            continue
        if current_best is None:
            lookup[key] = entry
        else:
            best_mean = (current_best.get("kernel_stats") or {}).get("mean")
            if best_mean is None or candidate_mean < best_mean:
                lookup[key] = entry
    return lookup


def resolve_baseline_candidate(lookup: Dict[Tuple[str, int, Optional[int]], Dict[str, Any]],
                               dataset: str,
                               trail: int,
                               threads: Optional[int]) -> Optional[Dict[str, Any]]:
    candidates: List[Tuple[str, int, Optional[int]]] = []
    candidates.append((dataset, trail, threads))
    if threads is not None:
        candidates.append((dataset, trail, None))
        if threads != 1:
            candidates.append((dataset, trail, 1))
    else:
        candidates.append((dataset, trail, 1))
    seen = set()
    ordered: List[Tuple[str, int, Optional[int]]] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    for key in ordered:
        entry = lookup.get(key)
        if entry:
            return entry
    return None


def compute_speedup(baseline_mean: Optional[float], candidate_mean: Optional[float]) -> Optional[float]:
    if baseline_mean is None or candidate_mean is None or candidate_mean == 0:
        return None
    return baseline_mean / candidate_mean


def success_rate(entry: Dict[str, Any]) -> Optional[float]:
    measure = entry.get("measure_runs")
    success = entry.get("successful_runs")
    if not measure:
        return None
    return success / measure


def summarize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline_map = best_baseline_lookup(records)
    rows: List[Dict[str, Any]] = []
    for entry in records:
        dataset = entry.get("dataset") or "default"
        trail = int(entry.get("trail", 0))
        threads = entry.get("threads")
        baseline = resolve_baseline_candidate(baseline_map, dataset, trail, threads)
        kernel_mean = (entry.get("kernel_stats") or {}).get("mean")
        total_mean = (entry.get("total_stats") or {}).get("mean")
        baseline_kernel_mean = (baseline.get("kernel_stats") or {}).get("mean") if baseline else None
        baseline_total_mean = (baseline.get("total_stats") or {}).get("mean") if baseline else None
        rows.append(
            {
                "dataset": dataset,
                "trail": trail,
                "model": entry.get("model"),
                "version": entry.get("version"),
                "compiler": entry.get("compiler"),
                "sequence_id": entry.get("sequence_id", 0),
                "is_baseline": bool(entry.get("is_baseline")),
                "threads": threads,
                "kernel_mean_s": kernel_mean,
                "total_mean_s": total_mean,
                "kernel_speedup": compute_speedup(baseline_kernel_mean, kernel_mean),
                "total_speedup": compute_speedup(baseline_total_mean, total_mean),
                "success_rate": success_rate(entry),
                "warmup_runs": entry.get("warmup_runs"),
                "measure_runs": entry.get("measure_runs"),
                "hint_key": entry.get("hint_key"),
                "executable": entry.get("executable"),
            }
        )
    rows.sort(
        key=lambda row: (
            dataset_rank(row["dataset"]),
            row["trail"],
            -1 if row.get("threads") is None else row.get("threads"),
            (0 if row["is_baseline"] else 1),
            row["model"],
            row.get("version") or 0,
            row["sequence_id"],
        )
    )
    return rows


def output_table(benchmark: str, ex_version: str, rows: List[Dict[str, Any]], missing_entries: List[Dict[str, Any]], source_path: Path) -> None:
    print(f"\n=== Benchmark: {benchmark} ({ex_version}) ===")
    print(f"[INFO] Loaded {len(rows)} summary record(s) from {source_path}")
    if not rows:
        return
    header = (
        f"{'dataset':<12} {'trail':<5} {'model':<12} {'thr':>5} {'ver':<5} {'kernel(us)':>12} "
        f"{'total(ms)':>12} {'k_speedup':>11} {'t_speedup':>11} {'succ%':>8} {'hint':>8}"
    )
    print(header)
    for row in rows:
        kernel_us = row["kernel_mean_s"] * 1e6 if row["kernel_mean_s"] is not None else None
        total_ms = row["total_mean_s"] * 1e3 if row["total_mean_s"] is not None else None
        speed_k = row["kernel_speedup"]
        speed_t = row["total_speedup"]
        succ = row["success_rate"]
        thread_label = row.get("threads")
        print(
            f"{row['dataset']:<12} "
            f"{row['trail']:<5d} "
            f"{(row['model'] or '-'):<12} "
            f"{(str(thread_label) if thread_label is not None else '-'):>5} "
            f"{str(row['version'] or '-'):>5} "
            f"{(f'{kernel_us:.3f}' if kernel_us is not None else '-'):>12} "
            f"{(f'{total_ms:.3f}' if total_ms is not None else '-'):>12} "
            f"{(f'{speed_k:.2f}x' if speed_k is not None else '-'):>11} "
            f"{(f'{speed_t:.2f}x' if speed_t is not None else '-'):>11} "
            f"{(f'{succ*100:.1f}' if succ is not None else '-'):>8} "
            f"{(row['hint_key'] or '-'):>8}"
        )
    if missing_entries:
        print(f"[WARN] Missing combinations reported: {len(missing_entries)}")
        for item in missing_entries[:10]:
            dataset = item.get("dataset", "-")
            model = item.get("model", "-")
            version = item.get("version", "-")
            print(f"  - dataset={dataset}, model={model}, version={version}")
        if len(missing_entries) > 10:
            print(f"  ... ({len(missing_entries) - 10} more)")


def maybe_write_csv(output_dir: Optional[Path], benchmark: str, ex_version: str, rows: List[Dict[str, Any]]) -> None:
    if not output_dir or not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{benchmark}_{ex_version}_summary.csv"
    fieldnames = [
        "dataset",
        "trail",
        "model",
        "threads",
        "version",
        "compiler",
        "sequence_id",
        "is_baseline",
        "kernel_mean_s",
        "total_mean_s",
        "kernel_speedup",
        "total_speedup",
        "success_rate",
        "warmup_runs",
        "measure_runs",
        "hint_key",
        "executable",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[INFO] CSV summary written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze time_measurement outputs for one or more benchmarks.")
    parser.add_argument("--benchmarks", default="all", help="Comma-separated benchmark names or 'all'.")
    parser.add_argument(
        "--ex-version",
        default="EX1",
        help="Experiment version to inspect (EX1, EX2, or EX3).",
    )
    parser.add_argument("--results-root", default="results/time_measurements", help="Root directory storing measurement JSONL files.")
    parser.add_argument("--missing-root", default="results/missing_bins", help="Root directory storing missing bin reports.")
    parser.add_argument("--trails", help="Comma-separated trail IDs to include (default: all).")
    parser.add_argument("--output-dir", help="Optional directory to write CSV summaries.")
    args = parser.parse_args()

    benchmark_list = discover_benchmarks(Path(args.results_root), parse_list(args.benchmarks))
    if not benchmark_list:
        print(f"[WARN] No benchmarks found under {args.results_root}")
        return
    trail_filter = parse_list(args.trails)
    trail_ids = [int(token) for token in trail_filter] if trail_filter else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    for benchmark in benchmark_list:
        measurement_path = Path(args.results_root) / benchmark / f"{args.ex_version}_time_measurements.jsonl"
        missing_path = Path(args.missing_root) / benchmark / f"{args.ex_version}_missing_bins.json"
        summaries = load_summary_records(measurement_path, trail_ids)
        rows = summarize_records(summaries)
        missing_entries = load_missing_entries(missing_path)
        output_table(benchmark, args.ex_version, rows, missing_entries, measurement_path)
        maybe_write_csv(output_dir, benchmark, args.ex_version, rows)


if __name__ == "__main__":
    main()

