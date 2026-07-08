"""Answer relevancy — does the answer actually address the question?

Generates the questions the answer would best answer (natively, so they stay in
the source language) and judges their closeness to the real question; a
noncommittal answer scores 0. No embeddings. Prompts are loaded from the
language pack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..dataset import EvalSample
from ..judge import Judge
from ..language import LanguageProfile
from .base import MetricResult, get_metric_prompts


@dataclass(frozen=True)
class _Prompts:
    generate_instruction: str
    similarity_instruction: str
    answer_label: str
    context_label: str
    question_label: str
    generated_label: str


_GENERATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": {"type": "string"}},
        "noncommittal": {"type": "boolean"},
    },
    "required": ["questions", "noncommittal"],
}

_SIMILARITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"similarities": {"type": "array", "items": {"type": "number"}}},
    "required": ["similarities"],
}


@dataclass
class AnswerRelevancy:
    name: str = field(default="answer_relevancy")

    def _prompts(self, profile: LanguageProfile) -> _Prompts:
        return _Prompts(**get_metric_prompts(profile, self.name))

    def score(
        self, sample: EvalSample, *, judge: Judge, profile: LanguageProfile
    ) -> MetricResult:
        p = self._prompts(profile)

        context_block = "\n\n".join(f"- {c}" for c in sample.contexts) or "(–)"
        generated = judge.structured(
            system=f"{p.generate_instruction}\n\n{profile.keep_language}",
            user=f"{p.answer_label}:\n{sample.answer}\n\n{p.context_label}:\n{context_block}",
            schema=_GENERATE_SCHEMA,
            tool_name="generated",
        )
        questions = [q for q in generated.get("questions", []) if q and q.strip()]
        noncommittal = bool(generated.get("noncommittal", False))

        if noncommittal:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"noncommittal": True, "questions": questions, "similarities": []},
            )
        if not questions:
            return MetricResult(
                name=self.name,
                score=float("nan"),
                details={"noncommittal": False, "questions": [], "similarities": []},
            )

        numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        sim_out = judge.structured(
            system=f"{p.similarity_instruction}\n\n{profile.keep_language}",
            user=f"{p.question_label}:\n{sample.question}\n\n{p.generated_label}:\n{numbered}",
            schema=_SIMILARITY_SCHEMA,
            tool_name="similarities",
        )
        similarities = [float(s) for s in sim_out.get("similarities", [])]
        score = sum(similarities) / len(similarities) if similarities else float("nan")
        return MetricResult(
            name=self.name,
            score=score,
            details={"noncommittal": False, "questions": questions, "similarities": similarities},
        )
