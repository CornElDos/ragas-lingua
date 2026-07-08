import math

import pytest

from ragas_lingua import AnswerRelevancy, EvalSample, FakeJudge, get_profile


def _sample():
    return EvalSample.from_dict(
        {
            "question": "Vad är Sveriges huvudstad?",
            "answer": "Sveriges huvudstad är Stockholm.",
            "contexts": ["Stockholm är Sveriges huvudstad."],
        }
    )


def test_score_is_mean_similarity_of_generated_questions():
    judge = FakeJudge(
        responses=[
            {"questions": ["Vilken stad är Sveriges huvudstad?", "Var ligger huvudstaden?", "Vad heter huvudstaden?"], "noncommittal": False},
            {"similarities": [0.9, 0.8, 0.7]},
        ]
    )
    result = AnswerRelevancy().score(_sample(), judge=judge, profile=get_profile("sv"))
    assert math.isclose(result.score, 0.8, rel_tol=1e-9)


def test_noncommittal_answer_scores_zero_without_second_call():
    judge = FakeJudge(responses=[{"questions": ["q"], "noncommittal": True}])
    result = AnswerRelevancy().score(_sample(), judge=judge, profile=get_profile("sv"))
    assert result.score == 0.0
    assert len(judge.calls) == 1  # no similarity call once noncommittal


def test_no_questions_yields_nan():
    judge = FakeJudge(responses=[{"questions": [], "noncommittal": False}])
    result = AnswerRelevancy().score(_sample(), judge=judge, profile=get_profile("sv"))
    assert math.isnan(result.score)


def test_generation_prompt_is_swedish_and_native():
    judge = FakeJudge(responses=[{"questions": ["q"], "noncommittal": False}, {"similarities": [1.0]}])
    AnswerRelevancy().score(_sample(), judge=judge, profile=get_profile("sv"))
    system = judge.calls[0]["system"].lower()
    assert "svenska" in system
    assert "frågor" in system


def test_unauthored_language_raises_notimplemented():
    judge = FakeJudge(responses=[{"questions": ["q"], "noncommittal": False}])
    with pytest.raises(NotImplementedError):
        AnswerRelevancy().score(_sample(), judge=judge, profile=get_profile("de"))
