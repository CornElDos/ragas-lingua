"""The single entry point: ``evaluate(dataset, metrics, judge=..., language=...)``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

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
) -> EvaluationResult:
    """Score every sample with every metric and return per-metric means.

    ``language`` sets the default LanguageProfile; a sample may override it via
    its own ``language`` field.
    """
    if not isinstance(dataset, EvalDataset):
        dataset = EvalDataset.from_dicts(dataset)

    default_profile = get_profile(language)
    per_sample: list[dict[str, MetricResult]] = []
    totals: dict[str, float] = {m.name: 0.0 for m in metrics}

    for sample in dataset:
        profile = get_profile(sample.language) if sample.language else default_profile
        row: dict[str, MetricResult] = {}
        for metric in metrics:
            result = metric.score(sample, judge=judge, profile=profile)
            row[metric.name] = result
            totals[metric.name] += result.score
        per_sample.append(row)

    n = max(len(dataset), 1)
    scores = {name: total / n for name, total in totals.items()}
    return EvaluationResult(scores=scores, per_sample=per_sample)
