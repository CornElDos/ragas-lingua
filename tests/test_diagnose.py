import math

from ragas_lingua import (
    EvaluationResult,
    MetricResult,
    Thresholds,
    diagnose,
    diagnose_all,
)


def _r(name: str, score: float, details: dict | None = None) -> MetricResult:
    return MetricResult(name=name, score=score, details=details or {})


def test_well_grounded_when_faithfulness_high():
    d = diagnose({"faithfulness": _r("faithfulness", 0.95)})
    assert d.label == "well_grounded"


def test_retrieval_gap_low_faithfulness_high_correctness():
    d = diagnose(
        {
            "faithfulness": _r(
                "faithfulness", 0.5, {"verdicts": [{"statement": "x", "supported": False}]}
            ),
            "answer_correctness": _r("answer_correctness", 0.9),
        }
    )
    assert d.label == "retrieval_gap"
    assert d.ungrounded_statements == ["x"]


def test_hallucination_low_faithfulness_low_correctness():
    d = diagnose(
        {
            "faithfulness": _r("faithfulness", 0.3),
            "answer_correctness": _r("answer_correctness", 0.2),
        }
    )
    assert d.label == "hallucination"


def test_partial_when_correctness_is_mixed():
    d = diagnose(
        {
            "faithfulness": _r("faithfulness", 0.6),
            "answer_correctness": _r("answer_correctness", 0.65),
        }
    )
    assert d.label == "partial"


def test_ungrounded_unverified_without_a_reference():
    d = diagnose({"faithfulness": _r("faithfulness", 0.4)})
    assert d.label == "ungrounded_unverified"


def test_flags_noisy_retrieval_and_off_topic():
    d = diagnose(
        {
            "faithfulness": _r("faithfulness", 0.95),
            "context_precision": _r("context_precision", 0.3),
            "answer_relevancy": _r("answer_relevancy", 0.2),
        }
    )
    assert d.label == "well_grounded"
    assert "noisy_retrieval" in d.flags
    assert "off_topic" in d.flags


def test_inconclusive_when_faithfulness_is_nan():
    d = diagnose({"faithfulness": _r("faithfulness", math.nan)})
    assert d.label == "inconclusive"


def test_custom_thresholds_shift_the_boundary():
    d = diagnose({"faithfulness": _r("faithfulness", 0.7)}, thresholds=Thresholds(high=0.6))
    assert d.label == "well_grounded"


def test_diagnose_all_over_an_evaluation_result():
    result = EvaluationResult(
        scores={"faithfulness": 0.625},
        per_sample=[
            {"faithfulness": _r("faithfulness", 0.95)},
            {
                "faithfulness": _r("faithfulness", 0.3),
                "answer_correctness": _r("answer_correctness", 0.2),
            },
        ],
    )
    labels = [d.label for d in diagnose_all(result)]
    assert labels == ["well_grounded", "hallucination"]


def test_summary_is_readable_and_descriptive_not_prescriptive():
    d = diagnose(
        {
            "faithfulness": _r("faithfulness", 0.5),
            "answer_correctness": _r("answer_correctness", 0.9),
        }
    )
    text = d.summary()
    assert "retrieval_gap" in text
    assert "faithfulness=0.50" in text
