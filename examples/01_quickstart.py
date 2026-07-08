"""Quickstart: evaluate a small Swedish dataset with all four metrics.

    ANTHROPIC_API_KEY=... python examples/01_quickstart.py
"""

import os

from ragas_lingua import (
    AnswerCorrectness,
    AnswerRelevancy,
    ClaudeJudge,
    ContextPrecision,
    EvalDataset,
    Faithfulness,
    evaluate,
)

data = EvalDataset.from_dicts(
    [
        {
            "question": "Vad är Sveriges huvudstad?",
            "answer": "Stockholm är Sveriges huvudstad och landets största stad.",
            "contexts": [
                "Stockholm är huvudstad i Sverige.",
                "Stockholm är Sveriges folkrikaste stad.",
            ],
            "ground_truth": "Stockholm är Sveriges huvudstad.",
        },
    ]
)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this example.")
    result = evaluate(
        data,
        metrics=[Faithfulness(), AnswerCorrectness(), ContextPrecision(), AnswerRelevancy()],
        judge=ClaudeJudge(),
        language="sv",
    )
    for name, score in result.to_dict().items():
        print(f"{name:20} {score:.2f}")


if __name__ == "__main__":
    main()
