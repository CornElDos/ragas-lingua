"""Loads per-language prompt packs from TOML files.

Each language is one self-contained file, ``ragas_lingua/prompts/<code>.toml``,
with the language metadata, the keep-language directive, and one section per
metric. Adding a language means adding a file — no code changes. Point
:func:`register_prompts_dir` at your own directory to add languages from outside
the package (yours take precedence over the built-ins).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.10
    import tomli as tomllib

_BUILTIN_DIR = Path(__file__).parent / "prompts"
_EXTRA_DIRS: list[Path] = []

_REQUIRED_TOP = {"code", "name", "english_name", "keep_language"}


def register_prompts_dir(path: str | Path) -> None:
    """Register a directory of ``<code>.toml`` language packs (yours win over built-ins)."""
    _EXTRA_DIRS.append(Path(path))
    load_pack.cache_clear()
    available_language_codes.cache_clear()


def _search_dirs() -> list[Path]:
    # Extra dirs first so a user pack overrides a built-in of the same code.
    return [*reversed(_EXTRA_DIRS), _BUILTIN_DIR]


@lru_cache(maxsize=None)
def available_language_codes() -> tuple[str, ...]:
    codes: set[str] = set()
    for directory in _search_dirs():
        if directory.is_dir():
            codes.update(f.stem for f in directory.glob("*.toml"))
    return tuple(sorted(codes))


@lru_cache(maxsize=None)
def load_pack(code: str) -> dict:
    for directory in _search_dirs():
        path = directory / f"{code}.toml"
        if path.is_file():
            with path.open("rb") as fh:
                data = tomllib.load(fh)
            _validate(code, path, data)
            return data
    raise ValueError(
        f"No prompt pack for language {code!r}. "
        f"Available: {list(available_language_codes())}. "
        "Add a ragas_lingua/prompts/<code>.toml file, or register_prompts_dir(...)."
    )


def _validate(code: str, path: Path, data: dict) -> None:
    missing = _REQUIRED_TOP - data.keys()
    if missing:
        raise ValueError(f"Prompt pack {path} is missing keys: {sorted(missing)}")
    if data["code"] != code:
        raise ValueError(f"Prompt pack {path}: 'code' is {data['code']!r}, expected {code!r}")
