"""German (de) language-profile checks.

These verify language *consistency* — that the directive is German and free of
English leakage. Grammar is NOT asserted here and, unlike Swedish, has not yet
been reviewed by a native German speaker: treat the German profile as
provisional (native-review wanted) until it has.
"""

from ragas_lingua import get_profile

_ENGLISH_LEAKAGE = {"the", "you", "must", "answer", "translate", "output", "same", "never", "and"}


def _words(text: str) -> set[str]:
    return {w.strip(".,;:!?—-").lower() for w in text.split()}


def test_identity_fields():
    p = get_profile("de")
    assert p.code == "de"
    assert p.name == "Deutsch"
    assert p.english_name == "German"


def test_directive_is_german():
    kl = get_profile("de").keep_language.lower()
    assert "deutsch" in kl
    assert "englische" in kl  # names the language it must not switch to


def test_directive_uses_german_markers():
    # German-specific letters or clearly German function words.
    kl = get_profile("de").keep_language
    has_special = any(ch in kl for ch in "äöüßÄÖÜ")
    has_words = any(w in _words(kl) for w in {"und", "die", "der", "auf", "was", "wie"})
    assert has_special or has_words


def test_no_english_leakage():
    leaked = _words(get_profile("de").keep_language) & _ENGLISH_LEAKAGE
    assert not leaked, f"English words leaked into the German directive: {leaked}"
