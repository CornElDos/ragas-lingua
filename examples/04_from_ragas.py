"""Migrate from RAGAS: bring a RAGAS-style dataset straight over (no API key).

The dataset layer accepts both the classic RAGAS keys (question / answer /
contexts / ground_truth) and the v2 renames (user_input / response /
retrieved_contexts / reference), so it is a genuine drop-in.

    python examples/04_from_ragas.py
"""

from ragas_lingua import EvalDataset

# A RAGAS v2-style list of samples, as EvaluationDataset.to_list() would give you.
ragas_v2_rows = [
    {
        "user_input": "Vad är Sveriges huvudstad?",
        "response": "Stockholm är Sveriges huvudstad.",
        "retrieved_contexts": ["Stockholm är huvudstad i Sverige."],
        "reference": "Stockholm är Sveriges huvudstad.",
    },
]


def main() -> None:
    # Or straight from a live RAGAS dataset: EvalDataset.from_ragas(my_ragas_dataset)
    data = EvalDataset.from_dicts(ragas_v2_rows)
    sample = next(iter(data))
    print("question    :", sample.question)
    print("answer      :", sample.answer)
    print("contexts    :", sample.contexts)
    print("ground_truth:", sample.ground_truth)


if __name__ == "__main__":
    main()
