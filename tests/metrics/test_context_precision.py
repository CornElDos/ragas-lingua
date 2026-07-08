import math

import pytest

from ragas_lingua import ContextPrecision, EvalSample, FakeJudge, get_profile


def _sample(contexts):
    return EvalSample.from_dict(
        {
            "question": "Vad är Sveriges huvudstad?",
            "answer": "Stockholm.",
            "contexts": contexts,
            "ground_truth": "Stockholm är Sveriges huvudstad.",
        }
    )


def test_average_precision_of_ranked_verdicts():
    # relevance [1, 0, 1] -> AP = (1/1 + 2/3) / 2 = 5/6
    judge = FakeJudge(
        responses=[
            {"verdicts": [{"relevant": True}, {"relevant": False}, {"relevant": True}]}
        ]
    )
    result = ContextPrecision().score(_sample(["a", "b", "c"]), judge=judge, profile=get_profile("sv"))
    assert math.isclose(result.score, 5 / 6, rel_tol=1e-9)


def test_all_relevant_scores_one():
    judge = FakeJudge(responses=[{"verdicts": [{"relevant": True}, {"relevant": True}]}])
    result = ContextPrecision().score(_sample(["a", "b"]), judge=judge, profile=get_profile("sv"))
    assert math.isclose(result.score, 1.0, rel_tol=1e-9)


def test_none_relevant_scores_zero():
    judge = FakeJudge(responses=[{"verdicts": [{"relevant": False}, {"relevant": False}]}])
    result = ContextPrecision().score(_sample(["a", "b"]), judge=judge, profile=get_profile("sv"))
    assert result.score == 0.0


def test_no_contexts_yields_nan():
    judge = FakeJudge(responses=[])
    result = ContextPrecision().score(_sample([]), judge=judge, profile=get_profile("sv"))
    assert math.isnan(result.score)
    assert judge.calls == []


def test_prompt_is_swedish_and_native():
    judge = FakeJudge(responses=[{"verdicts": [{"relevant": True}]}])
    ContextPrecision().score(_sample(["a"]), judge=judge, profile=get_profile("sv"))
    system = judge.calls[0]["system"].lower()
    assert "svenska" in system
    assert "kontext" in system


def test_unauthored_language_raises_notimplemented():
    judge = FakeJudge(responses=[{"verdicts": [{"relevant": True}]}])
    with pytest.raises(ValueError):
        ContextPrecision().score(_sample(["a"]), judge=judge, profile=get_profile("zz"))
