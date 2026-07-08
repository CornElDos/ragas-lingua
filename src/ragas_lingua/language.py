"""Language profiles — the core asset of ragas-lingua.

A profile carries the per-language, human-authored material that RAGAS lacks:
native-language judge directives (and, from M1, native instructions and
few-shot examples per metric). Profiles are plain data, so adding a language
is authoring, not coding.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageProfile:
    code: str  # ISO 639-1, e.g. "sv"
    name: str  # endonym, e.g. "Svenska"
    english_name: str  # "Swedish"
    # Appended to every judge prompt to stop the cross-language leakage that is
    # RAGAS's core multilingual failure. Written natively, by a speaker.
    keep_language: str


SWEDISH = LanguageProfile(
    code="sv",
    name="Svenska",
    english_name="Swedish",
    keep_language=(
        "Allt du extraherar och allt du skriver ska vara på svenska, samma "
        "språk som svaret. Översätt aldrig till engelska."
    ),
)

GERMAN = LanguageProfile(
    code="de",
    name="Deutsch",
    english_name="German",
    keep_language=(
        "Alles, was du extrahierst, und deine gesamte Ausgabe müssen auf "
        "Deutsch sein, in derselben Sprache wie die Antwort. Übersetze "
        "niemals ins Englische."
    ),
)

_PROFILES: dict[str, LanguageProfile] = {p.code: p for p in (SWEDISH, GERMAN)}


def get_profile(code: str) -> LanguageProfile:
    try:
        return _PROFILES[code.lower()]
    except KeyError:
        raise ValueError(
            f"No language profile for {code!r}. Available: {sorted(_PROFILES)}. "
            "Profiles are data — add one in ragas_lingua/language.py."
        ) from None


def available_languages() -> list[str]:
    return sorted(_PROFILES)
