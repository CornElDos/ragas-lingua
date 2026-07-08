"""Judge confidence via self-consistency.

An LLM judge is noisy: run it twice at a non-zero temperature and the score
moves. Most eval tools — RAGAS included — report a single number as if it were
exact. This runs a metric several times and reports the mean plus how much it
wobbled, so you know how far to trust the number: ``0.61 +/- 0.09``.

Use a judge sampled above temperature 0 (e.g. ``ClaudeJudge(temperature=0.7)``);
at temperature 0 every run is identical and the confidence is meaningless.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import Any, Iterable, Sequence

from ._concurrent import run_concurrent
from .dataset import EvalDataset, EvalSample
from .judge import Judge
from .language import LanguageProfile, get_profile
from .metrics.base import Metric


@dataclass(frozen=True)
class ConfidenceBands:
    """Std-deviation cutoffs that turn wobble into a high/medium/low label."""

    high_max_std: float = 0.05
    medium_max_std: float = 0.15


DEFAULT_BANDS = ConfidenceBands()


@dataclass
class MetricConfidence:
    name: str
    mean: float
    std: float
    runs: int
    scores: list[float] = field(default_factory=list)
    confidence: str = "unknown"  # high | medium | low | unknown

    def summary(self) -> str:
        if math.isnan(self.mean):
            return f"{self.name}: n/a (no scorable runs)"
        return (
            f"{self.name}: {self.mean:.2f} +/- {self.std:.2f} "
            f"({self.confidence} confidence, n={self.runs})"
        )


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _label(std: float, bands: ConfidenceBands) -> str:
    if math.isnan(std):
        return "unknown"
    if std <= bands.high_max_std:
        return "high"
    if std <= bands.medium_max_std:
        return "medium"
    return "low"


def _aggregate(
    name: str, scores: Sequence[float], attempted: int, bands: ConfidenceBands
) -> MetricConfidence:
    usable = [s for s in scores if not _is_nan(s)]
    if not usable:
        return MetricConfidence(
            name=name, mean=math.nan, std=math.nan, runs=0, scores=[], confidence="unknown"
        )
    mean = fmean(usable)
    std = pstdev(usable) if len(usable) > 1 else 0.0
    return MetricConfidence(
        name=name,
        mean=mean,
        std=std,
        runs=len(usable),
        scores=list(usable),
        confidence=_label(std, bands),
    )


def score_with_confidence(
    metric: Metric,
    sample: EvalSample,
    *,
    judge: Judge,
    profile: LanguageProfile,
    runs: int = 5,
    bands: ConfidenceBands = DEFAULT_BANDS,
    max_concurrency: int = 1,
) -> MetricConfidence:
    """Score one sample ``runs`` times and report the mean and its wobble.

    Returns a :class:`MetricConfidence`. Runs that come back as NaN (e.g. a
    metric with nothing to score) are dropped before aggregating; if every run
    is NaN the result is NaN with ``unknown`` confidence. ``max_concurrency``
    overlaps the runs in a thread pool.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if runs > 1 and getattr(judge, "temperature", None) == 0:
        warnings.warn(
            "score_with_confidence at temperature 0 gives identical runs and a "
            "meaningless 0.0 spread; pass a judge with temperature > 0 "
            "(e.g. ClaudeJudge(temperature=0.7)).",
            stacklevel=2,
        )
    thunks = [
        lambda: metric.score(sample, judge=judge, profile=profile).score
        for _ in range(runs)
    ]
    scores = run_concurrent(thunks, max_concurrency)
    return _aggregate(metric.name, scores, runs, bands)


def evaluate_with_confidence(
    dataset: EvalDataset | Iterable[dict[str, Any]],
    metrics: Sequence[Metric],
    *,
    judge: Judge,
    language: str = "sv",
    runs: int = 5,
    bands: ConfidenceBands = DEFAULT_BANDS,
    max_concurrency: int = 1,
) -> list[dict[str, MetricConfidence]]:
    """Run :func:`score_with_confidence` for every sample and every metric.

    Returns one dict per sample (metric name -> MetricConfidence), mirroring
    ``EvaluationResult.per_sample`` so the two line up row for row.
    ``max_concurrency`` overlaps the (sample, metric) units; each unit's own
    runs stay sequential so the pool never nests.
    """
    if not isinstance(dataset, EvalDataset):
        dataset = EvalDataset.from_dicts(dataset)

    samples = list(dataset)
    default_profile = get_profile(language)
    profiles = [
        get_profile(s.language) if s.language else default_profile for s in samples
    ]

    thunks = []
    index: list[tuple[int, str]] = []
    for i, sample in enumerate(samples):
        for metric in metrics:
            thunks.append(
                lambda m=metric, s=sample, p=profiles[i]: score_with_confidence(
                    m, s, judge=judge, profile=p, runs=runs, bands=bands, max_concurrency=1
                )
            )
            index.append((i, metric.name))

    results = run_concurrent(thunks, max_concurrency)
    rows: list[dict[str, MetricConfidence]] = [{} for _ in samples]
    for (i, name), result in zip(index, results):
        rows[i][name] = result
    return rows
