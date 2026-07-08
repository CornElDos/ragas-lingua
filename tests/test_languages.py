"""Language-pack architecture: discovery, coverage, review status, warnings."""

import warnings

import pytest

from ragas_lingua import available_languages, get_profile, reviewed_languages
from ragas_lingua.metrics.base import get_metric_prompts
from ragas_lingua.metrics import base as _base
from ragas_lingua.promptpacks import load_pack

_METRICS = ["faithfulness", "answer_correctness", "context_precision", "answer_relevancy"]


def test_sv_and_de_available_and_only_sv_reviewed():
    langs = available_languages()
    assert "sv" in langs
    assert "de" in langs
    assert reviewed_languages() == ["sv"]


@pytest.mark.parametrize("code", available_languages())
def test_every_pack_has_metadata_and_every_metric_section(code):
    pack = load_pack(code)
    for key in ("code", "name", "english_name", "keep_language"):
        assert key in pack, f"{code}: missing top-level '{key}'"
    for metric in _METRICS:
        assert metric in pack, f"{code}: missing '{metric}' section"


def test_unknown_language_raises_value_error():
    with pytest.raises(ValueError):
        get_profile("zz")


def test_unreviewed_language_warns():
    _base._UNREVIEWED_WARNED.discard("de")
    with pytest.warns(UserWarning, match="AUTO-GENERATED"):
        get_metric_prompts(get_profile("de"), "faithfulness")


def test_reviewed_language_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        get_metric_prompts(get_profile("sv"), "faithfulness")
