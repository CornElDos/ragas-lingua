"""Run the whole flow offline with FakeJudge (no API key).

FakeJudge returns scripted judge responses, so you can see exactly what each
metric asks for and how the score is built, without calling a model. Faithfulness
makes two judge calls per sample: first it extracts statements, then it judges
each statement against the context.

    python examples/06_offline_with_fakejudge.py
"""

from ragas_lingua import EvalDataset, Faithfulness, FakeJudge, diagnose_all, evaluate

data = EvalDataset.from_dicts(
    [
        {
            "question": "Vad är Sveriges huvudstad?",
            "answer": "Stockholm är Sveriges huvudstad.",
            "contexts": ["Stockholm är huvudstad i Sverige."],
            "ground_truth": "Stockholm är Sveriges huvudstad.",
        },
    ]
)


def main() -> None:
    judge = FakeJudge(
        responses=[
            {"statements": ["Stockholm är Sveriges huvudstad."]},
            {"verdicts": [{"statement": "Stockholm är Sveriges huvudstad.", "supported": True}]},
        ]
    )
    result = evaluate(data, [Faithfulness()], judge=judge, language="sv")
    print("scores:", result.to_dict())
    for d in diagnose_all(result):
        print(d.summary())


if __name__ == "__main__":
    main()
