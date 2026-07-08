"""Diagnose: label each sample (retrieval gap vs hallucination vs grounded).

    ANTHROPIC_API_KEY=... python examples/02_diagnose.py
"""

import os

from ragas_lingua import (
    AnswerCorrectness,
    ClaudeJudge,
    ContextPrecision,
    EvalDataset,
    Faithfulness,
    diagnose_all,
    evaluate,
)

data = EvalDataset.from_dicts(
    [
        {  # grounded: the answer's claim is in the context
            "question": "Vad är Sveriges huvudstad?",
            "answer": "Stockholm är Sveriges huvudstad.",
            "contexts": ["Stockholm är huvudstad i Sverige."],
            "ground_truth": "Stockholm är Sveriges huvudstad.",
        },
        {  # correct but ungrounded: the year is right, the context doesn't contain it
            "question": "När grundades Stockholm?",
            "answer": "Stockholm grundades omkring år 1252.",
            "contexts": ["Stockholm är Sveriges huvudstad."],
            "ground_truth": "Stockholm grundades omkring 1252.",
        },
    ]
)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this example.")
    result = evaluate(
        data,
        metrics=[Faithfulness(), AnswerCorrectness(), ContextPrecision()],
        judge=ClaudeJudge(),
        language="sv",
    )
    for i, d in enumerate(diagnose_all(result)):
        print(f"\nSample {i}: {d.summary()}")
        for statement in d.ungrounded_statements:
            print(f"    ungrounded: {statement}")


if __name__ == "__main__":
    main()
