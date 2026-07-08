"""Concurrency preserves ordering and results (evaluate + the run_concurrent helper)."""

import pytest

from ragas_lingua import EvalDataset, FakeJudge, evaluate, evaluate_with_confidence
from ragas_lingua._concurrent import run_concurrent
from ragas_lingua.metrics.base import MetricResult


class _ScoreFromAnswer:
    """A metric whose score depends only on the sample, via one judge call.

    Because the score is derived from the sample (not from response ordering),
    the result is identical whether runs are sequential or interleaved — which
    is exactly what lets us assert concurrency didn't scramble anything.
    """

    name = "dummy"

    def score(self, sample, *, judge, profile):
        out = judge.structured(system="s", user=sample.answer, schema={"type": "object"})
        return MetricResult(name=self.name, score=float(out["score"]))


_SCORES = {"a": 0.1, "bb": 0.2, "ccc": 0.3}


def _judge():
    return FakeJudge(handler=lambda *, system, user, schema: {"score": _SCORES[user]})


def _data():
    return EvalDataset.from_dicts([{"question": "q", "answer": a} for a in _SCORES])


def test_run_concurrent_preserves_order():
    thunks = [lambda i=i: i * i for i in range(10)]
    assert run_concurrent(thunks, max_concurrency=4) == [i * i for i in range(10)]


def test_run_concurrent_propagates_exceptions():
    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        run_concurrent([lambda: 1, boom, lambda: 3], max_concurrency=3)


def test_run_concurrent_rejects_zero():
    with pytest.raises(ValueError):
        run_concurrent([lambda: 1], max_concurrency=0)


def test_evaluate_concurrent_matches_sequential():
    seq = evaluate(_data(), [_ScoreFromAnswer()], judge=_judge(), max_concurrency=1)
    par = evaluate(_data(), [_ScoreFromAnswer()], judge=_judge(), max_concurrency=4)
    assert seq.per_sample[0]["dummy"].score == pytest.approx(0.1)
    assert seq.per_sample[2]["dummy"].score == pytest.approx(0.3)
    assert par.to_dict() == pytest.approx(seq.to_dict())


def test_evaluate_with_confidence_concurrent_preserves_rows():
    rows = evaluate_with_confidence(
        _data(), [_ScoreFromAnswer()], judge=_judge(), runs=3, max_concurrency=4
    )
    assert [r["dummy"].mean for r in rows] == pytest.approx([0.1, 0.2, 0.3])
    # A deterministic judge -> zero spread -> high confidence.
    assert all(r["dummy"].confidence == "high" for r in rows)
