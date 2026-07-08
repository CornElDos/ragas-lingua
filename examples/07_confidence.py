"""Judge confidence via self-consistency, offline with FakeJudge (no API key).

A single faithfulness number hides how sure the judge is. score_with_confidence
runs the metric several times and reports the mean plus the spread, so a
wobbling judge shows up as low confidence instead of a false-precise 0.67.

Here the scripted judge disagrees with itself across three runs (1.0, 0.5, 0.5)
-> mean 0.67 with a wide spread -> low confidence. With a real judge you get the
same effect by sampling above temperature 0: ClaudeJudge(temperature=0.7).

    python examples/07_confidence.py
"""

from ragas_lingua import EvalDataset, Faithfulness, FakeJudge, score_with_confidence
from ragas_lingua.language import get_profile

sample = next(
    iter(
        EvalDataset.from_dicts(
            [
                {
                    "question": "Vad kostar en ansökningsavgift i tvistemål?",
                    "answer": "Ansökningsavgiften är 900 kronor och betalas till tingsrätten.",
                    "contexts": ["Ansökningsavgiften i tvistemål är 900 kronor."],
                }
            ]
        )
    )
)

A = "Ansökningsavgiften är 900 kronor."
B = "Avgiften betalas till tingsrätten."


def main() -> None:
    # Three runs; the judge grounds both statements once, then wavers on one.
    judge = FakeJudge(
        responses=[
            {"statements": [A, B]},
            {"verdicts": [{"statement": A, "supported": True}, {"statement": B, "supported": True}]},
            {"statements": [A, B]},
            {"verdicts": [{"statement": A, "supported": True}, {"statement": B, "supported": False}]},
            {"statements": [A, B]},
            {"verdicts": [{"statement": A, "supported": False}, {"statement": B, "supported": True}]},
        ]
    )
    mc = score_with_confidence(
        Faithfulness(), sample, judge=judge, profile=get_profile("sv"), runs=3
    )
    print(mc.summary())
    print("individual runs:", mc.scores)


if __name__ == "__main__":
    main()
