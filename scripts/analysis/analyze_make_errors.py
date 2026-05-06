#!/usr/bin/env python3
"""
遍历 `results/make_results` 下的 build.log，抽取编译/链接错误并做基础分类。

典型用法：
    python scripts/analysis/analyze_make_errors.py \
        --results-root results/make_results \
        --benchmarks all \
        --ex-versions EX2 \
        --output-csv analysis_make_errors.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_list(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    tokens: List[str] = []
    for chunk in raw.replace(";", ",").split(","):
        item = chunk.strip()
        if item:
            tokens.append(item)
    return tokens or None


def discover_children(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


@dataclass
class ErrorRecord:
    benchmark: str
    ex_version: str
    log_path: Path
    source: Optional[str]
    line: Optional[int]
    column: Optional[int]
    message: str
    category: str
    model: Optional[str]
    version: Optional[int]


# 分类规则，顺序很重要（先匹配更具体的类别）
CATEGORY_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("openmp_clause", re.compile(r"predetermined.+shared", re.IGNORECASE)),
    ("implicit_function", re.compile(r"implicit declaration of function", re.IGNORECASE)),
    ("implicit_int", re.compile(r"implicit declaration", re.IGNORECASE)),
    ("undeclared_identifier", re.compile(r"undeclared \(first use", re.IGNORECASE)),
    ("unknown_type", re.compile(r"unknown type name", re.IGNORECASE)),
    ("unknown_member", re.compile(r"has no member named", re.IGNORECASE)),
    ("conflicting_types", re.compile(r"conflicting types for", re.IGNORECASE)),
    ("redefinition", re.compile(r"redefinition of", re.IGNORECASE)),
    ("syntax_expected", re.compile(r"expected .* before", re.IGNORECASE)),
    ("syntax_token", re.compile(r"expected expression", re.IGNORECASE)),
    ("too_many_arguments", re.compile(r"too many arguments", re.IGNORECASE)),
    ("too_few_arguments", re.compile(r"too few arguments", re.IGNORECASE)),
    ("incompatible_pointer", re.compile(r"incompatible (pointer|types)", re.IGNORECASE)),
    ("pointer_from_integer", re.compile(r"makes pointer from integer", re.IGNORECASE)),
    ("array_size", re.compile(r"storage size of", re.IGNORECASE)),
    ("missing_header", re.compile(r"No such file or directory", re.IGNORECASE)),
    ("undefined_reference", re.compile(r"undefined reference to", re.IGNORECASE)),
    ("ld_cannot_find", re.compile(r"cannot find -", re.IGNORECASE)),
    ("linker_other", re.compile(r"ld returned 1 exit status|collect2: error", re.IGNORECASE)),
]

ERROR_LINE_HINTS = (
    "error:",
    "fatal error:",
    "undefined reference",
    "cannot find -",
    "ld returned 1 exit status",
    "collect2: error",
)

FILE_ERROR_RE = re.compile(
    r"""
    ^(?P<path>[^:\s][^:]*)                      # 文件路径
    (?::(?P<line>\d+))?                         # 行号
    (?::(?P<column>\d+))?                       # 列号
    :\s*(?P<kind>fatal\s+error|error):\s*(?P<message>.+)$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def should_keep_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    lowered = text.lower()
    if "failed to build" in lowered:
        return False
    if text.startswith("make[") and "error" in lowered:
        return False
    return any(hint in lowered for hint in ERROR_LINE_HINTS)


def normalize_message(message: str) -> str:
    msg = message.strip()
    # 某些信息会附带多余空格或连续空白
    msg = re.sub(r"\s+", " ", msg)
    return msg


def infer_variant(source: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    if not source:
        return None, None
    base = Path(source).name
    stem = Path(base).stem
    parts = stem.split("_")
    version: Optional[int] = None
    model: Optional[str] = None
    if len(parts) >= 2 and re.fullmatch(r"v\d+", parts[-1]):
        try:
            version = int(parts[-1][1:])
        except ValueError:
            version = None
        model = parts[-2]
    elif len(parts) >= 2:
        model = parts[-1]
    return model, version


def categorize(message: str) -> str:
    for name, pattern in CATEGORY_PATTERNS:
        if pattern.search(message):
            return name
    return "other"


def parse_error_line(
    line: str,
    *,
    benchmark: str,
    ex_version: str,
    log_path: Path,
) -> ErrorRecord:
    source = None
    line_no = None
    column = None
    message = line.strip()
    match = FILE_ERROR_RE.match(message)
    if match:
        source = match.group("path")
        message = match.group("message").strip()
        if match.group("line"):
            line_no = int(match.group("line"))
        if match.group("column"):
            column = int(match.group("column"))
    message = normalize_message(message)
    model, version = infer_variant(source)
    category = categorize(message)
    return ErrorRecord(
        benchmark=benchmark,
        ex_version=ex_version,
        log_path=log_path,
        source=source,
        line=line_no,
        column=column,
        message=message,
        category=category,
        model=model,
        version=version,
    )


def parse_build_log(log_path: Path, benchmark: str, ex_version: str) -> List[ErrorRecord]:
    if not log_path.exists():
        return []
    records: List[ErrorRecord] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as fp:
        for raw_line in fp:
            if not should_keep_line(raw_line):
                continue
            try:
                record = parse_error_line(
                    raw_line.rstrip("\n"),
                    benchmark=benchmark,
                    ex_version=ex_version,
                    log_path=log_path,
                )
            except Exception:
                continue
            records.append(record)
    return records


def collect_records(
    results_root: Path,
    benchmarks: Sequence[str],
    ex_versions: Optional[Sequence[str]],
) -> Tuple[List[ErrorRecord], int]:
    all_records: List[ErrorRecord] = []
    logs_seen = 0
    for benchmark in benchmarks:
        bench_dir = results_root / benchmark
        if not bench_dir.exists():
            continue
        available_ex = discover_children(bench_dir)
        selected_ex: Iterable[str]
        if not ex_versions or any(token.lower() == "all" for token in ex_versions):
            selected_ex = available_ex
        else:
            selected_ex = [name for name in ex_versions if name in available_ex]
        for ex_version in selected_ex:
            log_path = bench_dir / ex_version / "build.log"
            if not log_path.exists():
                continue
            logs_seen += 1
            all_records.extend(parse_build_log(log_path, benchmark, ex_version))
    return all_records, logs_seen


def write_csv(records: Sequence[ErrorRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "benchmark",
                "ex_version",
                "model",
                "version",
                "category",
                "message",
                "source",
                "line",
                "column",
                "log_path",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.benchmark,
                    record.ex_version,
                    record.model or "",
                    record.version or "",
                    record.category,
                    record.message,
                    record.source or "",
                    record.line or "",
                    record.column or "",
                    str(record.log_path),
                ]
            )


def write_json(records: Sequence[ErrorRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(
            [
                {
                    "benchmark": record.benchmark,
                    "ex_version": record.ex_version,
                    "model": record.model,
                    "version": record.version,
                    "category": record.category,
                    "message": record.message,
                    "source": record.source,
                    "line": record.line,
                    "column": record.column,
                    "log_path": str(record.log_path),
                }
                for record in records
            ],
            fp,
            indent=2,
            ensure_ascii=False,
        )


def print_summary(records: Sequence[ErrorRecord], logs_seen: int, top_n: int) -> None:
    print(f"已扫描 {logs_seen} 个 build.log，收集到 {len(records)} 条错误信息。")
    if not records:
        return

    category_counter = Counter(rec.category for rec in records)
    print("\n按错误类别统计：")
    for category, count in category_counter.most_common():
        print(f"  {category:>20}: {count}")

    bench_counter = Counter((rec.benchmark, rec.ex_version) for rec in records)
    print("\n错误最多的基准 (含 EX 版本)：")
    for (benchmark, ex_version), count in bench_counter.most_common(top_n):
        print(f"  {benchmark}/{ex_version}: {count}")

    model_counter = Counter(rec.model or "baseline" for rec in records)
    print("\n错误最多的模型来源：")
    for model, count in model_counter.most_common(top_n):
        print(f"  {model}: {count}")

    message_counter = Counter(rec.message for rec in records)
    print("\n最常见的报错内容：")
    for message, count in message_counter.most_common(top_n):
        print(f"  ({count}x) {message}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="分类分析 results/make_results 下的 build.log 编译错误。"
    )
    parser.add_argument(
        "--results-root",
        default="results/make_results",
        type=Path,
        help="build.log 的根目录（默认：results/make_results）",
    )
    parser.add_argument(
        "--benchmarks",
        default="all",
        help="逗号分隔的基准列表，或 all",
    )
    parser.add_argument(
        "--ex-versions",
        default="all",
        help="逗号分隔的 EX 版本列表（如 EX1,EX2），或 all",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="可选，把详细记录写入 CSV",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="可选，把详细记录写入 JSON",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="摘要中展示的 Top-N 条目数量",
    )
    args = parser.parse_args()

    benchmark_filter = parse_list(args.benchmarks)
    if benchmark_filter is None or any(token.lower() == "all" for token in benchmark_filter):
        benchmarks = discover_children(args.results_root)
    else:
        benchmarks = benchmark_filter

    ex_filter = parse_list(args.ex_versions)

    records, logs_seen = collect_records(args.results_root, benchmarks, ex_filter)
    print_summary(records, logs_seen, args.top_n)

    if args.output_csv:
        write_csv(records, args.output_csv)
        print(f"\nCSV 结果已输出到 {args.output_csv}")
    if args.output_json:
        write_json(records, args.output_json)
        print(f"JSON 结果已输出到 {args.output_json}")


if __name__ == "__main__":
    main()

