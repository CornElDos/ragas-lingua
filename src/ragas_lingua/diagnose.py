"""Diagnosis — a labelled failure taxonomy over the metric scores.

This is analysis, not advice. It classifies *what* a low score means — a
retrieval gap vs a hallucination vs noisy retrieval — using the metric outputs
the tool already produced, and surfaces the ungrounded statements as evidence.
It deliberately does not prescribe how to change your pipeline; describing the
observation is the tool's job, deciding the fix is yours.

A single low number can't say "tool or RAG"; the combination can:

- low faithfulness + high answer_correctness -> retrieval gap (correct but ungrounded)
- low faithfulness + low answer_correctness  -> hallucination (ungrounded and wrong)
- low context_precision                      -> noisy retrieval (a secondary flag)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from .evaluate import EvaluationResult
from .metrics.base import MetricResult

FAITHFULNESS = "faithfulness"
ANSWER_CORRECTNESS = "answer_correctness"
CONTEXT_PRECISION = "context_precision"
ANSWER_RELEVANCY = "answer_relevancy"


@dataclass(frozen=True)
class Thresholds:
    high: float = 0.8
    low: float = 0.5


_EXPLANATIONS = {
    "well_grounded": "The answer's claims are supported by the retrieved context.",
    "retrieval_gap": (
        "The answer is largely correct, but its claims are not supported by the "
        "retrieved context — the supporting material was not retrieved."
    ),
    "hallucination": (
        "The answer contains claims that are neither supported by the context nor "
        "correct against the reference."
    ),
    "partial": "Some of the answer's claims are ungrounded and its correctness is mixed.",
    "ungrounded_unverified": (
        "The answer's claims are not supported by the retrieved context; add a "
        "ground_truth (answer_correctness) to tell a retrieval gap apart from a "
        "hallucination."
    ),
    "inconclusive": "Faithfulness could not be computed for this sample.",
}

_FLAG_EXPLANATIONS = {
    "noisy_retrieval": "Some retrieved contexts are not relevant to the question.",
    "off_topic": "The answer does not fully address the question.",
}


@dataclass
class Diagnosis:
    label: str
    flags: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    ungrounded_statements: list[str] = field(default_factory=list)
    explanation: str = ""

    def summary(self) -> str:
        head = self.label
        if self.flags:
            head += " [" + ", ".join(self.flags) + "]"
        scores = " ".join(f"{k}={v:.2f}" for k, v in self.metrics.items())
        return f"{head}  ({scores})  {self.explanation}"


def _score(results: Mapping[str, MetricResult], name: str) -> float | None:
    result = results.get(name)
    if result is None:
        return None
    score = result.score
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return None
    return float(score)


def diagnose(
    results: Mapping[str, MetricResult], *, thresholds: Thresholds = Thresholds()
) -> Diagnosis:
    """Classify one sample from its metric results (name -> MetricResult)."""
    faith = _score(results, FAITHFULNESS)
    correct = _score(results, ANSWER_CORRECTNESS)
    precision = _score(results, CONTEXT_PRECISION)
    relevancy = _score(results, ANSWER_RELEVANCY)

    flags: list[str] = []
    if precision is not None and precision < thresholds.low:
        flags.append("noisy_retrieval")
    if relevancy is not None and relevancy < thresholds.low:
        flags.append("off_topic")

    ungrounded: list[str] = []
    faith_result = results.get(FAITHFULNESS)
    if faith_result is not None:
        for verdict in faith_result.details.get("verdicts", []):
            if isinstance(verdict, Mapping) and not verdict.get("supported"):
                ungrounded.append(str(verdict.get("statement", "")))

    if faith is None:
        label = "inconclusive"
    elif faith >= thresholds.high:
        label = "well_grounded"
    elif correct is None:
        label = "ungrounded_unverified"
    elif correct >= thresholds.high:
        label = "retrieval_gap"
    elif correct < thresholds.low:
        label = "hallucination"
    else:
        label = "partial"

    metrics = {
        name: value
        for name, value in (
            (FAITHFULNESS, faith),
            (ANSWER_CORRECTNESS, correct),
            (CONTEXT_PRECISION, precision),
            (ANSWER_RELEVANCY, relevancy),
        )
        if value is not None
    }
    return Diagnosis(
        label=label,
        flags=flags,
        metrics=metrics,
        ungrounded_statements=ungrounded,
        explanation=_EXPLANATIONS[label],
    )


def diagnose_all(
    result: EvaluationResult, *, thresholds: Thresholds = Thresholds()
) -> list[Diagnosis]:
    """Diagnose every sample in an EvaluationResult (in order)."""
    return [diagnose(row, thresholds=thresholds) for row in result.per_sample]
