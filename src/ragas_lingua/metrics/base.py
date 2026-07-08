"""Metric protocol and result type.

Concrete metrics (faithfulness, answer_correctness, context_precision,
answer_relevancy) land in M1. Each will follow the same shape: build a
language-native prompt from the profile, ask the judge for structured output,
then apply RAGAS-equivalent scoring math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..dataset import EvalSample
from ..judge import Judge
from ..language import LanguageProfile


@dataclass
class MetricResult:
    name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Metric(Protocol):
    name: str

    def score(
        self, sample: EvalSample, *, judge: Judge, profile: LanguageProfile
    ) -> MetricResult: ...
