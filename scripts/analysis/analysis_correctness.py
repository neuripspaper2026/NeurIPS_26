#!/usr/bin/env python3
"""
汇总 correctness checker 的 JSONL 结果，输出每个 benchmark/ex-version 的状态统计。

示例：
    python scripts/analysis/analysis_correctness.py \
        --benchmarks 2mm,3mm \
        --ex-versions EX1 \
        --results-root results/correctness \
        --output-dir analysis_summaries
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_list(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    tokens = []
    for chunk in raw.replace(";", ",").split(","):
        item = chunk.strip()
        if item:
            tokens.append(item)
    return tokens or None


def extract_model_and_version_from_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """
    从文件名中提取 model 和 version 信息。
    
    示例：
        "output_claude_v10_mini_output.txt" -> ("claude", 10)
        "output_gpt5.1_v3_small_output.txt" -> ("gpt5.1", 3)
        "output_qwen_v7_large_output.txt" -> ("qwen", 7)
    
    Args:
        filename: 文件名字符串
    
    Returns:
        (model, version) 元组，如果无法提取则返回 (None, None)
    """
    # 尝试匹配模式：output_{model}_v{version}_{dataset}_output.txt
    # 支持的 model: claude, gpt4, gpt5.1, qwen 等
    pattern = r'output_([a-zA-Z0-9.]+)_v(\d+)_'
    match = re.search(pattern, filename)
    
    if match:
        model = match.group(1)
        version = int(match.group(2))
        return (model, version)
    
    return (None, None)


def normalize_version(version_value: any) -> Optional[int]:
    """
    将 version 值标准化为整数。
    
    示例：
        "v10" -> 10
        "v1" -> 1
        10 -> 10
        "10" -> 10
        None -> None
    
    Args:
        version_value: 版本值（可能是字符串或整数）
    
    Returns:
        标准化后的整数版本号，如果无法转换则返回 None
    """
    if version_value is None:
        return None
    
    # 如果是整数，直接返回
    if isinstance(version_value, int):
        return version_value
    
    # 如果是字符串
    if isinstance(version_value, str):
        # 去掉开头的 'v' 或 'V'
        cleaned = version_value.strip().lstrip('vV')
        # 尝试转换为整数
        try:
            return int(cleaned)
        except ValueError:
            return None
    
    # 其他类型，尝试直接转换
    try:
        return int(version_value)
    except (ValueError, TypeError):
        return None


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


def load_entries(summary_path: Path) -> List[Dict[str, Any]]:
    if not summary_path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with summary_path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def summarize_entries(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {"total": len(entries)}
    for entry in entries:
        status = entry.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def ensure_output_dir(path: Optional[Path]) -> Optional[Path]:
    if path:
        path.mkdir(parents=True, exist_ok=True)
    return path


def write_summary_csv(
    output_dir: Optional[Path],
    benchmark: str,
    ex_version: str,
    entries: List[Dict[str, Any]],
) -> None:
    if not output_dir or not entries:
        return
    # Create detail subdirectory
    detail_dir = output_dir / "detail" / ex_version
    detail_dir.mkdir(parents=True, exist_ok=True)
    detail_path = detail_dir / f"{benchmark}_{ex_version}_correctness_detail.csv"
    fieldnames = [
        "timestamp",
        "stage",
        "dataset",
        "model",
        "version",
        "file_name",
        "status",
        "message",
        "checker",
    ]
    with detail_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            # 如果 model 或 version 为 null，尝试从 file_name 中提取
            model = entry.get("model")
            version = entry.get("version")
            file_name = entry.get("file_name", "")
            
            if (model is None or version is None) and file_name:
                extracted_model, extracted_version = extract_model_and_version_from_filename(file_name)
                if model is None:
                    model = extracted_model
                if version is None:
                    version = extracted_version
            
            # 标准化 version 为纯数字（去掉 "v" 前缀）
            version = normalize_version(version)
            
            writer.writerow(
                {
                    "timestamp": entry.get("timestamp"),
                    "stage": entry.get("stage"),
                    "dataset": entry.get("dataset"),
                    "model": model,
                    "version": version,
                    "file_name": file_name,
                    "status": entry.get("status"),
                    "message": entry.get("message"),
                    "checker": entry.get("checker"),
                }
            )
    print(f"[INFO] Detail CSV saved: {detail_path}")


def print_table(benchmark: str, ex_version: str, counts: Dict[str, int]) -> None:
    total = counts.get("total", 0)
    pass_count = counts.get("pass", 0)
    fail_count = counts.get("fail", 0)
    skip_count = counts.get("skip", 0)
    error_count = counts.get("error", 0)
    unknown_count = counts.get("unknown", 0)
    print(
        f"{benchmark:<20} {ex_version:<8} {total:>5} "
        f"{pass_count:>5} {fail_count:>5} {skip_count:>5} {error_count:>5} {unknown_count:>7}"
    )


def summarize_by_model(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for entry in entries:
        model = entry.get("model") or "unknown"
        bucket = summary.setdefault(model, {"total": 0})
        bucket["total"] += 1
        status = entry.get("status", "unknown")
        bucket[status] = bucket.get(status, 0) + 1
    return summary


def write_model_summary_csv(
    output_dir: Optional[Path],
    benchmark: str,
    ex_version: str,
    model_summary: Dict[str, Dict[str, int]],
) -> None:
    if not output_dir or not model_summary:
        return
    # Create summary subdirectory
    summary_dir = output_dir / "summary" / ex_version
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"{benchmark}_{ex_version}_model_summary.csv"
    fieldnames = ["model", "total", "pass", "fail", "skip", "error", "unknown"]
    with path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for model, stats in sorted(model_summary.items()):
            row = {"model": model}
            row.update({name: stats.get(name, 0) for name in fieldnames if name != "model"})
            writer.writerow(row)
    print(f"[INFO] Model summary CSV saved: {path}")


def print_model_summary(benchmark: str, ex_version: str, model_summary: Dict[str, Dict[str, int]]) -> None:
    if not model_summary:
        return
    print(f"Model summary for {benchmark} {ex_version}:")
    for model, stats in sorted(model_summary.items()):
        total = stats.get("total", 0)
        pass_count = stats.get("pass", 0)
        fail_count = stats.get("fail", 0)
        skip_count = stats.get("skip", 0)
        error_count = stats.get("error", 0)
        unknown_count = stats.get("unknown", 0)
        print(
            f"  {model:<12} total={total:<4} pass={pass_count:<4} fail={fail_count:<4} "
            f"skip={skip_count:<4} error={error_count:<4} unknown={unknown_count:<4}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze correctness JSONL outputs.")
    parser.add_argument("--benchmarks", default="all", help="Comma-separated benchmark names or 'all'.")
    parser.add_argument(
        "--ex-versions",
        required=True,
        help="Comma-separated EX versions (e.g., EX1,EX2,EX3).",
    )
    parser.add_argument(
        "--results-root",
        default="results/correctness",
        help="Root directory storing correctness JSONL files.",
    )
    parser.add_argument("--output-dir", help="Optional directory to store CSV detail files.")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    benchmarks = discover_benchmarks(results_root, parse_list(args.benchmarks))
    if not benchmarks:
        print(f"[WARN] No benchmarks found under {results_root}")
        return
    ex_versions = parse_list(args.ex_versions)
    if not ex_versions:
        print("[WARN] --ex-versions must specify at least one EX directory (e.g., EX1).")
        return
    base_output_dir = ensure_output_dir(Path(args.output_dir).resolve()) if args.output_dir else None
    output_dir = None
    if base_output_dir:
        if base_output_dir.name.lower() == "correctness":
            output_dir = base_output_dir
        else:
            output_dir = ensure_output_dir(base_output_dir / "correctness")

    print(f"{'benchmark':<20} {'ex':<8} {'total':>5} {'pass':>5} {'fail':>5} {'skip':>5} {'error':>5} {'unknown':>7}")
    for benchmark in benchmarks:
        for ex_version in ex_versions:
            summary_path = results_root / benchmark / ex_version / "correctness.jsonl"
            entries = [entry for entry in load_entries(summary_path) if entry.get("stage") == "check"]
            if not entries:
                counts = {"total": 0}
                print_table(benchmark, ex_version, counts)
                continue
            write_summary_csv(output_dir, benchmark, ex_version, entries)
            counts = summarize_entries(entries)
            print_table(benchmark, ex_version, counts)
            model_summary = summarize_by_model(entries)
            write_model_summary_csv(output_dir, benchmark, ex_version, model_summary)
            print_model_summary(benchmark, ex_version, model_summary)


if __name__ == "__main__":
    main()

