"""Self-consistency confidence: mean + spread over repeated judge runs."""

import math

import pytest

from ragas_lingua import (
    ConfidenceBands,
    EvalDataset,
    FakeJudge,
    MetricConfidence,
    evaluate_with_confidence,
    score_with_confidence,
)
from ragas_lingua.language import get_profile
from ragas_lingua.metrics.base import MetricResult

SV = get_profile("sv")


class _DummyMetric:
    """One judge call per score(); returns whatever score the judge reports.

    Lets a test script the exact score of each run without going through a real
    metric's extract+verdict machinery.
    """

    name = "dummy"

    def score(self, sample, *, judge, profile):
        out = judge.structured(system="s", user="u", schema={"type": "object"})
        return MetricResult(name=self.name, score=float(out["score"]))


def _sample():
    return next(
        iter(
            EvalDataset.from_dicts(
                [{"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "a"}]
            )
        )
    )


def _judge(scores):
    return FakeJudge([{"score": s} for s in scores])


def test_reports_mean_and_spread():
    mc = score_with_confidence(
        _DummyMetric(), _sample(), judge=_judge([0.6, 0.8, 0.7]), profile=SV, runs=3
    )
    assert isinstance(mc, MetricConfidence)
    assert mc.mean == pytest.approx(0.7)
    assert mc.runs == 3
    assert mc.std == pytest.approx(0.081649, abs=1e-4)
    assert mc.confidence == "medium"


def test_high_confidence_when_runs_agree():
    mc = score_with_confidence(
        _DummyMetric(), _sample(), judge=_judge([0.9, 0.9, 0.9]), profile=SV, runs=3
    )
    assert mc.std == 0.0
    assert mc.confidence == "high"


def test_low_confidence_when_runs_disagree():
    mc = score_with_confidence(
        _DummyMetric(), _sample(), judge=_judge([0.2, 0.9, 0.5]), profile=SV, runs=3
    )
    assert mc.confidence == "low"


def test_nan_runs_are_dropped_before_aggregating():
    mc = score_with_confidence(
        _DummyMetric(), _sample(), judge=_judge([math.nan, 0.8]), profile=SV, runs=2
    )
    assert mc.runs == 1
    assert mc.mean == pytest.approx(0.8)


def test_all_nan_is_unknown():
    mc = score_with_confidence(
        _DummyMetric(), _sample(), judge=_judge([math.nan, math.nan]), profile=SV, runs=2
    )
    assert math.isnan(mc.mean)
    assert mc.runs == 0
    assert mc.confidence == "unknown"
    assert "n/a" in mc.summary()


def test_custom_bands_shift_the_label():
    # A spread that is "low" by default becomes "high" with wide bands.
    wide = ConfidenceBands(high_max_std=0.5, medium_max_std=0.9)
    mc = score_with_confidence(
        _DummyMetric(), _sample(), judge=_judge([0.2, 0.9, 0.5]), profile=SV, runs=3, bands=wide
    )
    assert mc.confidence == "high"


def test_runs_must_be_positive():
    with pytest.raises(ValueError):
        score_with_confidence(_DummyMetric(), _sample(), judge=_judge([0.5]), profile=SV, runs=0)


def test_warns_at_temperature_zero():
    judge = _judge([0.5, 0.5])
    judge.temperature = 0  # simulate a deterministic ClaudeJudge
    with pytest.warns(UserWarning, match="temperature"):
        score_with_confidence(_DummyMetric(), _sample(), judge=judge, profile=SV, runs=2)


def test_summary_shows_plus_minus_and_name():
    mc = score_with_confidence(
        _DummyMetric(), _sample(), judge=_judge([0.6, 0.7]), profile=SV, runs=2
    )
    text = mc.summary()
    assert "dummy" in text
    assert "+/-" in text
    assert "n=2" in text


def test_evaluate_with_confidence_matches_per_sample_shape():
    data = EvalDataset.from_dicts(
        [
            {"question": "q1", "answer": "a1", "contexts": ["c"], "ground_truth": "a1"},
            {"question": "q2", "answer": "a2", "contexts": ["c"], "ground_truth": "a2"},
        ]
    )
    # 2 samples x 1 metric x 2 runs = 4 scripted responses, in order.
    judge = _judge([0.6, 0.8, 0.4, 0.4])
    rows = evaluate_with_confidence(
        data, [_DummyMetric()], judge=judge, language="sv", runs=2
    )
    assert len(rows) == 2
    assert rows[0]["dummy"].mean == pytest.approx(0.7)
    assert rows[1]["dummy"].mean == pytest.approx(0.4)
    assert rows[1]["dummy"].confidence == "high"
