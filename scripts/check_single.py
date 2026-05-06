#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.checkers.manual_runner import run_single_check


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single correctness comparison between two files."
    )
    parser.add_argument("--benchmark", required=True, help="Benchmark name (e.g., 2mm).")
    parser.add_argument("--baseline", required=True, help="Path to baseline output file.")
    parser.add_argument("--target", required=True, help="Path to target output file.")
    parser.add_argument("--threshold", type=float, help="Override tolerance threshold.")
    parser.add_argument(
        "--root",
        help="Repository root path (defaults to repo root inferred from this script).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_single_check(
        benchmark=args.benchmark,
        baseline=Path(args.baseline),
        target=Path(args.target),
        threshold=args.threshold,
        root=Path(args.root).resolve() if args.root else None,
    )
    status = result.status.upper()
    message = result.message
    print(f"[{status}] {message}")
    if status != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()

