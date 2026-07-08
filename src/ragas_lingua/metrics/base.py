"""Metric protocol, result type, and the per-language prompt loader hook."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..dataset import EvalSample
from ..judge import Judge
from ..language import LanguageProfile
from ..promptpacks import load_pack

_UNREVIEWED_WARNED: set[str] = set()


def get_metric_prompts(profile: LanguageProfile, metric_name: str) -> dict[str, str]:
    """Return a language's prompt section for a metric (field name -> string).

    Raises ``NotImplementedError`` if the language pack has no section for this
    metric, and warns once per process if the language is not native-reviewed.
    """
    pack = load_pack(profile.code)
    try:
        section = pack[metric_name]
    except KeyError:
        raise NotImplementedError(
            f"Language {profile.code!r} ({profile.english_name}) has no "
            f"'{metric_name}' prompt section."
        ) from None
    if not profile.reviewed and profile.code not in _UNREVIEWED_WARNED:
        _UNREVIEWED_WARNED.add(profile.code)
        warnings.warn(
            f"ragas-lingua: prompts for {profile.english_name} ({profile.code!r}) are "
            f"AUTO-GENERATED and not yet verified by a native speaker — treat scores as "
            f"provisional until reviewed.",
            stacklevel=3,
        )
    return section


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
