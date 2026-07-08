"""The single entry point: ``evaluate(dataset, metrics, judge=..., language=...)``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ._concurrent import run_concurrent
from .dataset import EvalDataset
from .judge import Judge
from .language import get_profile
from .metrics.base import Metric, MetricResult


@dataclass
class EvaluationResult:
    scores: dict[str, float]  # metric name -> mean score across the dataset
    per_sample: list[dict[str, MetricResult]] = field(default_factory=list)

    def __getitem__(self, key: str) -> float:
        return self.scores[key]

    def to_dict(self) -> dict[str, float]:
        return dict(self.scores)


def evaluate(
    dataset: EvalDataset | Iterable[dict[str, Any]],
    metrics: Sequence[Metric],
    *,
    judge: Judge,
    language: str = "sv",
    max_concurrency: int = 1,
) -> EvaluationResult:
    """Score every sample with every metric and return per-metric means.

    ``language`` sets the default LanguageProfile; a sample may override it via
    its own ``language`` field. ``max_concurrency`` runs the (sample, metric)
    judge calls in a thread pool — leave it at 1 for a plain sequential loop,
    raise it to overlap the I/O-bound judge round-trips on large datasets.
    """
    if not isinstance(dataset, EvalDataset):
        dataset = EvalDataset.from_dicts(dataset)

    samples = list(dataset)
    default_profile = get_profile(language)
    profiles = [
        get_profile(s.language) if s.language else default_profile for s in samples
    ]

    # One thunk per (sample, metric); results come back in submission order.
    thunks = []
    index: list[tuple[int, str]] = []
    for i, sample in enumerate(samples):
        for metric in metrics:
            thunks.append(
                lambda m=metric, s=sample, p=profiles[i]: m.score(s, judge=judge, profile=p)
            )
            index.append((i, metric.name))

    results = run_concurrent(thunks, max_concurrency)

    per_sample: list[dict[str, MetricResult]] = [{} for _ in samples]
    totals: dict[str, float] = {m.name: 0.0 for m in metrics}
    for (i, name), result in zip(index, results):
        per_sample[i][name] = result
        totals[name] += result.score

    n = max(len(samples), 1)
    scores = {name: total / n for name, total in totals.items()}
    return EvaluationResult(scores=scores, per_sample=per_sample)
