"""Judge protocol and evaluate() mechanics (language-agnostic).

Per-language quality checks live in tests/languages/.
"""

from dataclasses import dataclass

from ragas_lingua import EvalDataset, FakeJudge, MetricResult, evaluate
from ragas_lingua.judge import Judge


def test_fake_judge_satisfies_the_protocol():
    assert isinstance(FakeJudge(responses=[{}]), Judge)


def test_fake_judge_records_calls_and_pops_responses():
    judge = FakeJudge(responses=[{"score": 0.7}])
    out = judge.structured(system="sys", user="usr", schema={"type": "object"})
    assert out == {"score": 0.7}
    assert judge.calls[0]["user"] == "usr"


@dataclass
class _DummyMetric:
    name: str = "dummy"

    def score(self, sample, *, judge, profile):
        out = judge.structured(
            system=profile.keep_language,
            user=sample.answer,
            schema={"type": "object", "properties": {"score": {"type": "number"}}},
        )
        return MetricResult(name=self.name, score=float(out["score"]))


def test_evaluate_wires_judge_and_profile_and_averages():
    judge = FakeJudge(responses=[{"score": 1.0}, {"score": 0.0}])
    ds = EvalDataset.from_dicts(
        [
            {"question": "q1", "answer": "Stockholm"},
            {"question": "q2", "answer": "fel svar"},
        ]
    )
    result = evaluate(ds, [_DummyMetric()], judge=judge, language="sv")
    assert result["dummy"] == 0.5
    # The run language's keep-language directive reached the judge as the system prompt.
    assert "svenska" in judge.calls[0]["system"].lower()


def test_per_sample_language_overrides_run_default():
    judge = FakeJudge(responses=[{"score": 1.0}])
    ds = EvalDataset.from_dicts([{"question": "q", "answer": "Antwort", "language": "de"}])
    evaluate(ds, [_DummyMetric()], judge=judge, language="sv")
    # Sample declared German, so the German directive is used despite the sv default.
    assert "deutsch" in judge.calls[0]["system"].lower()
