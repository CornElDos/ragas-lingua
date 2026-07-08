"""Swedish (sv) language-profile checks.

These verify language *consistency* — that the directive is Swedish and free of
English leakage, the exact bug class ragas-lingua exists to fix. Grammar itself
is native-verified by a Swedish speaker, not asserted here.
"""

from ragas_lingua import get_profile

# English tokens that must never appear in a Swedish directive (word-boundary matched).
_ENGLISH_LEAKAGE = {"the", "you", "must", "answer", "translate", "output", "same", "never"}


def _words(text: str) -> set[str]:
    return {w.strip(".,;:!?—-").lower() for w in text.split()}


def test_identity_fields():
    p = get_profile("sv")
    assert p.code == "sv"
    assert p.name == "Svenska"
    assert p.english_name == "Swedish"


def test_directive_is_swedish():
    kl = get_profile("sv").keep_language.lower()
    assert "svenska" in kl
    assert "engelska" in kl  # names the language it must not switch to


def test_directive_uses_swedish_characters():
    # Any of å/ä/ö present is strong evidence the text is actually Swedish.
    kl = get_profile("sv").keep_language
    assert any(ch in kl for ch in "åäöÅÄÖ")


def test_no_english_leakage():
    leaked = _words(get_profile("sv").keep_language) & _ENGLISH_LEAKAGE
    assert not leaked, f"English words leaked into the Swedish directive: {leaked}"
