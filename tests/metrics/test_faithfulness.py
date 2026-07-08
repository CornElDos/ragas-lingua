import math

import pytest

from ragas_lingua import EvalSample, Faithfulness, FakeJudge, get_profile


def _sample():
    return EvalSample.from_dict(
        {
            "question": "Vad är Sveriges huvudstad och när grundades staden?",
            "answer": "Stockholm är Sveriges huvudstad och grundades år 1252.",
            "contexts": ["Stockholm är Sveriges huvudstad."],
        }
    )


def test_score_is_supported_over_total():
    judge = FakeJudge(
        responses=[
            {"statements": ["Stockholm är Sveriges huvudstad.", "Stockholm grundades år 1252."]},
            {
                "verdicts": [
                    {"statement": "Stockholm är Sveriges huvudstad.", "supported": True},
                    {"statement": "Stockholm grundades år 1252.", "supported": False},
                ]
            },
        ]
    )
    result = Faithfulness().score(_sample(), judge=judge, profile=get_profile("sv"))
    assert result.score == 0.5
    assert result.details["supported"] == 1
    assert result.details["total"] == 2


def test_extraction_prompt_is_swedish_and_native():
    judge = FakeJudge(responses=[{"statements": ["a"]}, {"verdicts": [{"statement": "a", "supported": True}]}])
    Faithfulness().score(_sample(), judge=judge, profile=get_profile("sv"))
    system_extract = judge.calls[0]["system"].lower()
    assert "svenska" in system_extract  # keep-language directive is present
    assert "faktagranskare" in system_extract or "påståenden" in system_extract


def test_full_support_scores_one():
    judge = FakeJudge(
        responses=[
            {"statements": ["Vatten består av väte och syre."]},
            {"verdicts": [{"statement": "Vatten består av väte och syre.", "supported": True}]},
        ]
    )
    result = Faithfulness().score(_sample(), judge=judge, profile=get_profile("sv"))
    assert result.score == 1.0


def test_no_statements_yields_nan():
    judge = FakeJudge(responses=[{"statements": []}])
    result = Faithfulness().score(_sample(), judge=judge, profile=get_profile("sv"))
    assert math.isnan(result.score)


def test_unauthored_language_raises_notimplemented():
    # German faithfulness prompts are not authored yet; must fail loudly, not guess.
    judge = FakeJudge(responses=[{"statements": ["x"]}])
    with pytest.raises(NotImplementedError):
        Faithfulness().score(_sample(), judge=judge, profile=get_profile("de"))
