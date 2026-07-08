"""Context precision — are the useful contexts ranked near the top?

Same idea as RAGAS context precision: the judge marks each retrieved context as
useful (or not) for answering the question given the reference, and the score is
the average precision of that ranked list. The relevance prompt is authored
natively per language so the verdicts don't degrade on non-English text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..dataset import EvalSample
from ..judge import Judge
from ..language import LanguageProfile
from .base import MetricResult


@dataclass(frozen=True)
class _Prompts:
    verdict_instruction: str
    question_label: str
    reference_label: str
    contexts_label: str


# Swedish (sv) — native-authored and native-verifiable.
_SV = _Prompts(
    verdict_instruction=(
        "Du avgör vilka KONTEXTER som är användbara för att besvara FRÅGAN, "
        "givet FACIT. Ge ett omdöme per kontext, i samma ordning som de "
        'anges: sätt "relevant" till sant om kontexten bidrar till att nå '
        "facit, annars falskt. Bedöm utifrån innehållet, inte ordningen."
    ),
    question_label="FRÅGA",
    reference_label="FACIT",
    contexts_label="KONTEXTER",
)

_PROMPTS: dict[str, _Prompts] = {"sv": _SV}

_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "relevant": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["relevant"],
            },
        }
    },
    "required": ["verdicts"],
}


def _average_precision(relevance: list[int]) -> float:
    total_relevant = sum(relevance)
    if total_relevant == 0:
        return 0.0
    cumulative = 0
    running = 0.0
    for k, rel in enumerate(relevance, start=1):
        if rel:
            cumulative += 1
            running += (cumulative / k) * rel
    return running / total_relevant


@dataclass
class ContextPrecision:
    name: str = field(default="context_precision")

    def _prompts(self, profile: LanguageProfile) -> _Prompts:
        try:
            return _PROMPTS[profile.code]
        except KeyError:
            raise NotImplementedError(
                f"ContextPrecision prompts are not yet authored for language "
                f"{profile.code!r}. Swedish (sv) is implemented; other languages "
                f"need native-authored, natively-reviewed prompts."
            ) from None

    def score(
        self, sample: EvalSample, *, judge: Judge, profile: LanguageProfile
    ) -> MetricResult:
        if not sample.contexts:
            return MetricResult(
                name=self.name,
                score=float("nan"),
                details={"note": "context_precision needs retrieved contexts"},
            )
        p = self._prompts(profile)
        reference = sample.ground_truth or sample.answer
        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(sample.contexts))
        judged = judge.structured(
            system=f"{p.verdict_instruction}\n\n{profile.keep_language}",
            user=(
                f"{p.question_label}:\n{sample.question}\n\n"
                f"{p.reference_label}:\n{reference}\n\n"
                f"{p.contexts_label}:\n{numbered}"
            ),
            schema=_VERDICT_SCHEMA,
            tool_name="verdicts",
        )
        verdicts = judged.get("verdicts", [])
        relevance = [1 if v.get("relevant") else 0 for v in verdicts]
        score = _average_precision(relevance)
        return MetricResult(
            name=self.name,
            score=score,
            details={"relevance": relevance, "verdicts": verdicts},
        )
