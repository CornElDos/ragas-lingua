"""Faithfulness — the fraction of the answer's claims that the context supports.

Same shape as RAGAS faithfulness (extract atomic statements from the answer,
then judge each against the retrieved context, score = supported / total), but
the prompts are authored natively per language and carry the profile's
keep-language directive, so statements don't leak into English.
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
    extract_instruction: str
    verdict_instruction: str
    question_label: str
    answer_label: str
    context_label: str
    statements_label: str


# Swedish (sv) — native-authored and native-verifiable.
_SV = _Prompts(
    extract_instruction=(
        "Du är en noggrann faktagranskare. Dela upp SVARET i fristående, enkla "
        "påståenden. Varje påstående ska uttrycka exakt en verifierbar "
        "faktauppgift, vara begripligt på egen hand och återges på samma språk "
        "som svaret. Lägg inte till egna slutsatser eller tolkningar."
    ),
    verdict_instruction=(
        "Avgör för varje påstående om det går att härleda direkt ur KONTEXTEN. "
        'Sätt "supported" till sant endast om påståendet stöds direkt av '
        "kontexten, annars falskt. Gissa inte utifrån allmän kunskap – använd "
        "enbart kontexten."
    ),
    question_label="FRÅGA",
    answer_label="SVAR",
    context_label="KONTEXT",
    statements_label="PÅSTÅENDEN",
)

_PROMPTS: dict[str, _Prompts] = {"sv": _SV}

_EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "statements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The atomic factual statements found in the answer.",
        }
    },
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
        try:
            return _PROMPTS[profile.code]
        except KeyError:
            raise NotImplementedError(
                f"Faithfulness prompts are not yet authored for language "
                f"{profile.code!r}. Swedish (sv) is implemented; other languages "
                f"need native-authored, natively-reviewed prompts."
            ) from None

    def score(
        self, sample: EvalSample, *, judge: Judge, profile: LanguageProfile
    ) -> MetricResult:
        p = self._prompts(profile)

        # 1) Extract atomic statements from the answer, in the answer's language.
        system_extract = f"{p.extract_instruction}\n\n{profile.keep_language}"
        user_extract = f"{p.question_label}:\n{sample.question}\n\n{p.answer_label}:\n{sample.answer}"
        extracted = judge.structured(
            system=system_extract,
            user=user_extract,
            schema=_EXTRACT_SCHEMA,
            tool_name="statements",
        )
        statements = [s for s in extracted.get("statements", []) if s and s.strip()]
        if not statements:
            return MetricResult(
                name=self.name,
                score=float("nan"),
                details={"statements": [], "verdicts": [], "note": "no statements extracted"},
            )

        # 2) Judge each statement against the context.
        context_block = "\n\n".join(f"- {c}" for c in sample.contexts) or "(ingen kontext)"
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
        system_verdict = f"{p.verdict_instruction}\n\n{profile.keep_language}"
        user_verdict = f"{p.context_label}:\n{context_block}\n\n{p.statements_label}:\n{numbered}"
        judged = judge.structured(
            system=system_verdict,
            user=user_verdict,
            schema=_VERDICT_SCHEMA,
            tool_name="verdicts",
        )
        verdicts = judged.get("verdicts", [])

        # 3) Score = supported / total.
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
