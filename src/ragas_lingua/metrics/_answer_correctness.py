"""Answer correctness — factual F1 (TP/FP/FN vs the reference) + semantic similarity.

Same structure as RAGAS answer_correctness, weighted 0.75/0.25. Prompts are
loaded from the language pack (prompts/<code>.toml).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..dataset import EvalSample
from ..judge import Judge
from ..language import LanguageProfile
from .base import MetricResult, get_metric_prompts


@dataclass(frozen=True)
class _Prompts:
    classify_instruction: str
    similarity_instruction: str
    answer_label: str
    reference_label: str


_CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "TP": {"type": "array", "items": {"type": "string"}},
        "FP": {"type": "array", "items": {"type": "string"}},
        "FN": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["TP", "FP", "FN"],
}

_SIMILARITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"similarity": {"type": "number"}},
    "required": ["similarity"],
}


@dataclass
class AnswerCorrectness:
    name: str = field(default="answer_correctness")
    weights: tuple[float, float] = (0.75, 0.25)  # (factual F1, semantic similarity)

    def _prompts(self, profile: LanguageProfile) -> _Prompts:
        return _Prompts(**get_metric_prompts(profile, self.name))

    def score(
        self, sample: EvalSample, *, judge: Judge, profile: LanguageProfile
    ) -> MetricResult:
        if not sample.ground_truth:
            return MetricResult(
                name=self.name,
                score=float("nan"),
                details={"note": "answer_correctness needs a ground_truth reference"},
            )
        p = self._prompts(profile)
        pair = f"{p.answer_label}:\n{sample.answer}\n\n{p.reference_label}:\n{sample.ground_truth}"

        classification = judge.structured(
            system=f"{p.classify_instruction}\n\n{profile.keep_language}",
            user=pair,
            schema=_CLASSIFY_SCHEMA,
            tool_name="classification",
        )
        tp = len(classification.get("TP", []))
        fp = len(classification.get("FP", []))
        fn = len(classification.get("FN", []))
        denom = tp + 0.5 * (fp + fn)
        f1 = tp / denom if denom else float("nan")

        sim_out = judge.structured(
            system=f"{p.similarity_instruction}\n\n{profile.keep_language}",
            user=pair,
            schema=_SIMILARITY_SCHEMA,
            tool_name="similarity",
        )
        similarity = float(sim_out.get("similarity", float("nan")))

        w_f1, w_sim = self.weights
        if math.isnan(f1):
            score = similarity
        elif math.isnan(similarity):
            score = f1
        else:
            score = w_f1 * f1 + w_sim * similarity

        return MetricResult(
            name=self.name,
            score=score,
            details={
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "f1": f1,
                "similarity": similarity,
                "classification": classification,
            },
        )
