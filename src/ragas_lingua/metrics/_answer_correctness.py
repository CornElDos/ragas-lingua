"""Answer correctness — how well the answer matches a reference answer.

Same structure as RAGAS answer_correctness: a factual F1 over statements
classified against the reference (TP / FP / FN) combined with a semantic
similarity, weighted (default 0.75 / 0.25). The classification and similarity
prompts are authored natively per language, so the comparison doesn't degrade
into English the way RAGAS's auto-translated prompts do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..dataset import EvalSample
from ..judge import Judge
from ..language import LanguageProfile
from .base import MetricResult


@dataclass(frozen=True)
class _Prompts:
    classify_instruction: str
    similarity_instruction: str
    answer_label: str
    reference_label: str


# Swedish (sv) — native-authored and native-verifiable.
_SV = _Prompts(
    classify_instruction=(
        "Du jämför SVARET mot FACIT (det korrekta svaret). Dela in de "
        "faktapåståenden som förekommer i tre grupper:\n"
        "- TP: påståenden i svaret som stöds av facit.\n"
        "- FP: påståenden i svaret som saknas i eller motsägs av facit.\n"
        "- FN: påståenden i facit som saknas i svaret.\n"
        "Återge varje påstående kort och på samma språk som texten."
    ),
    similarity_instruction=(
        "Bedöm hur väl SVARET stämmer överens med FACIT i sakinnehåll och "
        'betydelse. Sätt "similarity" till ett tal mellan 0 och 1, där 1 '
        "betyder att de i sak säger samma sak och 0 att de är helt olika."
    ),
    answer_label="SVAR",
    reference_label="FACIT",
)

_PROMPTS: dict[str, _Prompts] = {"sv": _SV}

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
    # (weight on factual F1, weight on semantic similarity); should sum to 1.
    weights: tuple[float, float] = (0.75, 0.25)

    def _prompts(self, profile: LanguageProfile) -> _Prompts:
        try:
            return _PROMPTS[profile.code]
        except KeyError:
            raise NotImplementedError(
                f"AnswerCorrectness prompts are not yet authored for language "
                f"{profile.code!r}. Swedish (sv) is implemented; other languages "
                f"need native-authored, natively-reviewed prompts."
            ) from None

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

        # Factual component: classify statements into TP / FP / FN, then F1.
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

        # Semantic component: an overall meaning-match score in [0, 1].
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
