from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from .base import CandidateCollectionResult, CheckerContext


def collect_candidate_files(
    context: CheckerContext,
    *,
    pattern: str = "*.txt",
    datasets: Optional[Sequence[str]] = None,
) -> CandidateCollectionResult:
    """
    Shared routine for locating correctness artifacts to be consumed by suite matchers.
    """

    correctness_dir = context.benchmark_dir / "EX5_correctness"
    if not correctness_dir.exists():
        return CandidateCollectionResult(
            status="skip",
            message=(
                f"{correctness_dir} not found. Run "
                "`python scripts/arg.py benchmark correctness-run` first."
            ),
            files=[],
        )

    dataset_filter = (
        {token.lower() for token in datasets} if datasets else None
        )

    candidates: List[Path] = sorted(correctness_dir.glob(pattern))
    if dataset_filter:
        filtered: List[Path] = []
        for path in candidates:
            dataset = _extract_dataset(path)
            if dataset and dataset.lower() in dataset_filter:
                filtered.append(path)
        candidates = filtered
    if not candidates:
        return CandidateCollectionResult(
            status="skip",
            message=(
                f"No {pattern} outputs in {correctness_dir}"
                if not dataset_filter
                else f"No outputs matched dataset filters: {', '.join(datasets)}"
            ),
            files=[],
        )

    return CandidateCollectionResult(status="ok", files=candidates)


def _extract_dataset(path: Path) -> Optional[str]:
    stem = path.stem
    if not (stem.startswith("output_") and stem.endswith("_output")):
        return None
    core = stem[len("output_") : -len("_output")]
    tokens = [token for token in core.split("_") if token]
    if not tokens:
        return None
    return tokens[-1]

