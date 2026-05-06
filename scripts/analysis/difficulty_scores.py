from __future__ import annotations

"""
Compute static \"optimization difficulty\" for ~92–100 benchmark kernels and emit
CSV buckets.
Metrics: LOC, function count, branch count, max nesting depth (lizard).
Buckets: total 0–2.5/2.5–5/5–7.5/7.5–10 => Very Low/Low/Medium/High.
Normalization: each metric is mapped to 0–10 using 5%/95% percentiles.
"""

import csv
import math
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import lizard
from lizard import analyze_files, get_extensions
from lizard_ext.lizardnd import LizardExtension

# allow importing benchmark_catalog
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(PROJECT_ROOT / "scripts"))

from benchmark_catalog import (  # type: ignore
    BENCHMARK_CODEE,
    BENCHMARK_MACHSUITE,
    BENCHMARK_MIBENCH,
    BENCHMARK_PARBOIL,
    BENCHMARK_PARSEC,
    BENCHMARK_POLYBENCH,
    BENCHMARK_RODINIA,
    BenchmarkMeta,
)

CATALOGS: Dict[str, BenchmarkMeta] = {
    "polybench": BENCHMARK_POLYBENCH,
    "rodinia": BENCHMARK_RODINIA,
    "codee": BENCHMARK_CODEE,
    "parboil": BENCHMARK_PARBOIL,
    "mibench": BENCHMARK_MIBENCH,
    "parsec": BENCHMARK_PARSEC,
    "machsuite": BENCHMARK_MACHSUITE,
}

EX1_DIR = PROJECT_ROOT / "EX1"
OUTPUT_CSV = SCRIPT_DIR / "difficulty_scores.csv"
EXTENSIONS = get_extensions([LizardExtension()])

WEIGHTS = {
    # Token count removed (professor request): LOC is already a good proxy for length.
    # Assign the removed 0.10 token weight to LOC (professor request).
    "branch_count": 0.45,
    "loop_depth": 0.20,
    "loc": 0.25,
    "functions": 0.10,
}

BUCKETS = [
    (0.0, 2.5, "d1"),
    (2.5, 5.0, "d2"),
    (5.0, 7.5, "d3"),
    (7.5, 10.1, "d4"),  # 10.1 to include 10
]


def strip_comments_keep_lines(text: str) -> List[str]:
    """Remove block/line comments while keeping line count."""

    def repl(match: re.Match[str]) -> str:
        return "\n" * match.group(0).count("\n")

    no_block = re.sub(r"/\*.*?\*/", repl, text, flags=re.S)
    lines: List[str] = []
    for line in no_block.splitlines():
        line = re.sub(r"//.*", "", line)
        lines.append(line)
    return lines


def count_loc_tokens(lines: Iterable[str]) -> int:
    loc = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        loc += 1
    return loc


def count_branches(lines: Iterable[str]) -> int:
    """
    Count branches in a simple, explainable way (no CC / CFG).
    We count:
      - 'if' keywords
      - 'case' labels (switch cases)
      - ternary operator '?' (best-effort: count '?' not part of '??' etc.)
    """
    if_count = 0
    case_count = 0
    ternary_count = 0
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if_count += len(re.findall(r"\bif\b", s))
        case_count += len(re.findall(r"\bcase\b", s))
        # best-effort ternary: count '?' characters
        ternary_count += s.count("?")
    return if_count + case_count + ternary_count


@lru_cache(maxsize=None)
def analyze_full_file(path: str):
    return next(analyze_files([path], exts=EXTENSIONS))


def analyze_segment(file_path: Path, start: int, end: int) -> Dict[str, Any]:
    if not file_path.exists():
        return {
            "loc": 0,
            "functions": 0,
            "branch_count": 0,
            "loop_depth": 0,
            "missing": True,
        }

    text = file_path.read_text(encoding="utf-8", errors="ignore")
    raw_lines = text.splitlines()
    segment_lines = raw_lines[start - 1 : end]
    cleaned_lines = strip_comments_keep_lines("\n".join(segment_lines))
    loc = count_loc_tokens(cleaned_lines)
    branch_count = count_branches(cleaned_lines)

    analysis = analyze_full_file(str(file_path))
    funcs = [
        f for f in analysis.function_list if f.start_line >= start and f.end_line <= end
    ]
    if not funcs:
        funcs = [
            f
            for f in analysis.function_list
            if not (f.end_line < start or f.start_line > end)
        ]

    loop_depth = (
        max((getattr(f, "max_nesting_depth", 0) or 0 for f in funcs), default=0)
        if funcs
        else 0
    )

    return {
        "loc": loc,
        "functions": len(funcs),
        "branch_count": branch_count,
        "loop_depth": loop_depth,
        "missing": False,
    }


def aggregate_segments(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not segments:
        return {
            "loc": 0,
            "functions": 0,
            "branch_count": 0,
            "loop_depth": 0,
        }
    loc = sum(s["loc"] for s in segments)
    functions = sum(s["functions"] for s in segments)
    branch_count = sum(s["branch_count"] for s in segments)
    loop_depth = max((s["loop_depth"] for s in segments), default=0)
    return {
        "loc": loc,
        "functions": functions,
        "branch_count": branch_count,
        "loop_depth": loop_depth,
    }


def to_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else [value]


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_vals[int(pos)]
    frac = pos - low
    return sorted_vals[low] * (1 - frac) + sorted_vals[high] * frac


def score_value(value: float, low: float, high: float) -> float:
    if high == low:
        return 5.0
    if value <= low:
        return 0.0
    if value >= high:
        return 10.0
    return (value - low) / (high - low) * 10.0


def bucket(total_score: float) -> str:
    for lower, upper, name in BUCKETS:
        if lower <= total_score < upper:
            return name
    return BUCKETS[-1][2]


def main() -> None:
    results: List[Dict[str, Any]] = []

    for category, catalog in CATALOGS.items():
        for bench, meta in catalog.items():
            target_files = to_list(meta["target_file"])
            starts = to_list(meta["start_line"])
            ends = to_list(meta["end_line"])
            if not (len(target_files) == len(starts) == len(ends)):
                raise ValueError(
                    f"{bench} meta length mismatch: files={len(target_files)}, "
                    f"starts={len(starts)}, ends={len(ends)}"
                )

            segment_metrics: List[Dict[str, Any]] = []
            file_paths: List[str] = []
            for filename, start, end in zip(target_files, starts, ends):
                file_path = EX1_DIR / bench / "EX1_optimized_codes" / filename
                file_paths.append(str(file_path))
                segment_metrics.append(analyze_segment(file_path, int(start), int(end)))

            agg = aggregate_segments(segment_metrics)
            results.append(
                {
                    "benchmark": bench,
                    "category": category,
                    "files": "|".join(file_paths),
                    **agg,
                }
            )

    metrics = ["loc", "functions", "branch_count", "loop_depth"]
    quantiles: Dict[str, Tuple[float, float]] = {}
    for m in metrics:
        values = [float(r[m]) for r in results]
        low = quantile(values, 0.05)
        high = quantile(values, 0.95)
        quantiles[m] = (low, high)

    for r in results:
        scores = {}
        for m in metrics:
            low, high = quantiles[m]
            scores[f"{m}_score"] = score_value(float(r[m]), low, high)
        total = sum(
            scores[f"{m}_score"] * WEIGHTS[m]
            for m in ["branch_count", "loop_depth", "loc", "functions"]
        )
        r.update(scores)
        r["total_score"] = total
        r["bucket"] = bucket(total)

    fieldnames = [
        "benchmark",
        "category",
        "files",
        "loc",
        "functions",
        "branch_count",
        "loop_depth",
        "loc_score",
        "functions_score",
        "branch_count_score",
        "loop_depth_score",
        "total_score",
        "bucket",
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["benchmark"]):
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Generated: {OUTPUT_CSV} (total {len(results)} records)")
    print("Percentiles (5%/95%):")
    for m in metrics:
        low, high = quantiles[m]
        print(f"  {m}: {low:.3f} / {high:.3f}")


if __name__ == "__main__":
    main()

