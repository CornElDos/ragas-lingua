"""Bring your own RAG: wire your pipeline's outputs into ragas-lingua.

Replace ``run_my_rag`` with your real retrieval + generation; everything else
stays. Never put real client or personal data in a committed file — keep such
data under ``_private/`` (git-ignored).

    ANTHROPIC_API_KEY=... python examples/03_bring_your_own_rag.py
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


def run_my_rag(question: str) -> dict:
    """Return the generated answer and the retrieved contexts for a question.

    Replace the body with your real pipeline:
        contexts = my_retriever.search(question)
        answer = my_llm.generate(question, contexts)
    """
    return {
        "answer": "Stockholm är Sveriges huvudstad.",
        "contexts": ["Stockholm är huvudstad i Sverige."],
    }


# Your evaluation questions, with optional gold answers for answer_correctness.
QUESTIONS = [
    {"question": "Vad är Sveriges huvudstad?", "ground_truth": "Stockholm är Sveriges huvudstad."},
]


def build_dataset() -> EvalDataset:
    rows = []
    for item in QUESTIONS:
        rag = run_my_rag(item["question"])
        rows.append(
            {
                "question": item["question"],
                "answer": rag["answer"],
                "contexts": rag["contexts"],
                "ground_truth": item.get("ground_truth"),
            }
        )
    return EvalDataset.from_dicts(rows)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this example.")
    result = evaluate(
        build_dataset(),
        metrics=[Faithfulness(), AnswerCorrectness(), ContextPrecision()],
        judge=ClaudeJudge(),
        language="sv",
    )
    print("scores:", result.to_dict())
    for d in diagnose_all(result):
        print(d.summary())


if __name__ == "__main__":
    main()
