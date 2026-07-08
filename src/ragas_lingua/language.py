"""Language profiles, built from the per-language prompt packs in ``prompts/``.

A language is one self-contained ``prompts/<code>.toml`` file. Adding a language
is adding a file — no code changes here. ``register_prompts_dir()`` lets you add
languages from your own directory. Swedish (sv) is native-verified; the rest are
auto-generated (``reviewed = false`` in their pack) until a native speaker checks
them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .promptpacks import available_language_codes, load_pack, register_prompts_dir

__all__ = [
    "LanguageProfile",
    "get_profile",
    "available_languages",
    "reviewed_languages",
    "register_prompts_dir",
    "SWEDISH",
    "GERMAN",
]


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    name: str  # endonym, e.g. "Svenska"
    english_name: str
    keep_language: str  # directive appended to every prompt (stops language leakage)
    reviewed: bool = False  # True only once a native speaker has verified the pack


def get_profile(code: str) -> LanguageProfile:
    data = load_pack(code.lower())
    return LanguageProfile(
        code=data["code"],
        name=data["name"],
        english_name=data["english_name"],
        keep_language=data["keep_language"],
        reviewed=bool(data.get("reviewed", False)),
    )


def available_languages() -> list[str]:
    return list(available_language_codes())


def reviewed_languages() -> list[str]:
    """Languages whose prompts a native speaker has verified."""
    return sorted(c for c in available_language_codes() if bool(load_pack(c).get("reviewed", False)))


# Convenience handles for the two verified/first-class languages.
SWEDISH = get_profile("sv")
GERMAN = get_profile("de")
