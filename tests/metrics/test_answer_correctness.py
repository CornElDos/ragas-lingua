import math

import pytest

from ragas_lingua import AnswerCorrectness, EvalSample, FakeJudge, get_profile


def _sample(ground_truth: str | None = "Stockholm är Sveriges huvudstad."):
    return EvalSample.from_dict(
        {
            "question": "Vad är Sveriges huvudstad?",
            "answer": "Sveriges huvudstad är Stockholm och staden grundades 1252.",
            "contexts": [],
            "ground_truth": ground_truth,
        }
    )


def test_combines_factual_f1_and_similarity_with_weights():
    # TP=2, FP=1, FN=1 -> F1 = 2 / (2 + 0.5*(1+1)) = 0.6667; similarity = 0.8
    # score = 0.75*0.6667 + 0.25*0.8 = 0.7
    judge = FakeJudge(
        responses=[
            {"TP": ["a", "b"], "FP": ["c"], "FN": ["d"]},
            {"similarity": 0.8},
        ]
    )
    result = AnswerCorrectness().score(_sample(), judge=judge, profile=get_profile("sv"))
    assert result.details["tp"] == 2
    assert math.isclose(result.details["f1"], 2 / 3, rel_tol=1e-9)
    assert math.isclose(result.score, 0.7, rel_tol=1e-9)


def test_perfect_match_scores_one():
    judge = FakeJudge(responses=[{"TP": ["a", "b"], "FP": [], "FN": []}, {"similarity": 1.0}])
    result = AnswerCorrectness().score(_sample(), judge=judge, profile=get_profile("sv"))
    assert math.isclose(result.score, 1.0, rel_tol=1e-9)


def test_missing_ground_truth_yields_nan():
    judge = FakeJudge(responses=[])  # judge must not be called
    result = AnswerCorrectness().score(_sample(ground_truth=None), judge=judge, profile=get_profile("sv"))
    assert math.isnan(result.score)
    assert judge.calls == []


def test_classification_prompt_is_swedish_and_native():
    judge = FakeJudge(responses=[{"TP": [], "FP": [], "FN": []}, {"similarity": 0.5}])
    AnswerCorrectness().score(_sample(), judge=judge, profile=get_profile("sv"))
    system = judge.calls[0]["system"].lower()
    assert "svenska" in system
    assert "facit" in system


def test_unauthored_language_raises_notimplemented():
    judge = FakeJudge(responses=[{"TP": [], "FP": [], "FN": []}, {"similarity": 1.0}])
    with pytest.raises(ValueError):
        AnswerCorrectness().score(_sample(), judge=judge, profile=get_profile("zz"))
