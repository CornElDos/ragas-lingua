"""Faithfulness — the fraction of the answer's claims that the context supports.

Same shape as RAGAS faithfulness (extract atomic statements from the answer,
then judge each against the retrieved context, score = supported / total). The
prompts are loaded from the language pack (prompts/<code>.toml), so they are
authored natively per language and carry the keep-language directive.
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
    extract_instruction: str
    verdict_instruction: str
    question_label: str
    answer_label: str
    context_label: str
    statements_label: str


_EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"statements": {"type": "array", "items": {"type": "string"}}},
    "required": ["statements"],
}

_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["statement", "supported"],
            },
        }
    },
    "required": ["verdicts"],
}


@dataclass
class Faithfulness:
    name: str = field(default="faithfulness")

    def _prompts(self, profile: LanguageProfile) -> _Prompts:
        return _Prompts(**get_metric_prompts(profile, self.name))

    def score(
        self, sample: EvalSample, *, judge: Judge, profile: LanguageProfile
    ) -> MetricResult:
        p = self._prompts(profile)

        system_extract = f"{p.extract_instruction}\n\n{profile.keep_language}"
        user_extract = f"{p.question_label}:\n{sample.question}\n\n{p.answer_label}:\n{sample.answer}"
        extracted = judge.structured(
            system=system_extract, user=user_extract, schema=_EXTRACT_SCHEMA, tool_name="statements"
        )
        statements = [s for s in extracted.get("statements", []) if s and s.strip()]
        if not statements:
            return MetricResult(
                name=self.name,
                score=float("nan"),
                details={"statements": [], "verdicts": [], "note": "no statements extracted"},
            )

        context_block = "\n\n".join(f"- {c}" for c in sample.contexts) or "(–)"
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
        system_verdict = f"{p.verdict_instruction}\n\n{profile.keep_language}"
        user_verdict = f"{p.context_label}:\n{context_block}\n\n{p.statements_label}:\n{numbered}"
        judged = judge.structured(
            system=system_verdict, user=user_verdict, schema=_VERDICT_SCHEMA, tool_name="verdicts"
        )
        verdicts = judged.get("verdicts", [])

        supported = sum(1 for v in verdicts if v.get("supported"))
        total = len(verdicts) or len(statements)
        score = supported / total if total else float("nan")
        return MetricResult(
            name=self.name,
            score=score,
            details={
                "statements": statements,
                "verdicts": verdicts,
                "supported": supported,
                "total": total,
            },
        )
