"""Answer relevancy — does the answer actually address the question?

Like RAGAS answer relevancy, this generates the questions the answer would best
answer and measures how close they are to the real question; a noncommittal
answer scores 0. RAGAS compares with embeddings, which is where its generated
questions come out in the wrong language for non-English input. Here the
questions are generated with a native prompt and the closeness is judged by the
LLM itself, so there is no embedding model and no language drift.
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
    generate_instruction: str
    similarity_instruction: str
    answer_label: str
    context_label: str
    question_label: str
    generated_label: str


# Swedish (sv) — native-authored and native-verifiable.
_SV = _Prompts(
    generate_instruction=(
        "Utifrån SVARET (och eventuell KONTEXT), formulera de frågor som "
        "svaret bäst besvarar. Ge tre kortfattade frågor på samma språk som "
        'svaret. Sätt dessutom "noncommittal" till sant om svaret är '
        'undvikande eller icke-informativt (till exempel "jag vet inte"), '
        "annars falskt."
    ),
    similarity_instruction=(
        "Bedöm hur väl varje GENERERAD FRÅGA motsvarar URSPRUNGSFRÅGAN i "
        "betydelse. Ge ett tal mellan 0 och 1 per fråga, i samma ordning "
        "(1 = samma fråga, 0 = orelaterad)."
    ),
    answer_label="SVAR",
    context_label="KONTEXT",
    question_label="URSPRUNGSFRÅGA",
    generated_label="GENERERADE FRÅGOR",
)

_PROMPTS: dict[str, _Prompts] = {"sv": _SV}

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
        try:
            return _PROMPTS[profile.code]
        except KeyError:
            raise NotImplementedError(
                f"AnswerRelevancy prompts are not yet authored for language "
                f"{profile.code!r}. Swedish (sv) is implemented; other languages "
                f"need native-authored, natively-reviewed prompts."
            ) from None

    def score(
        self, sample: EvalSample, *, judge: Judge, profile: LanguageProfile
    ) -> MetricResult:
        p = self._prompts(profile)

        context_block = "\n\n".join(f"- {c}" for c in sample.contexts) or "(ingen kontext)"
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
