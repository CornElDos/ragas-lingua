"""RAGAS-compatible dataset types.

Accepts both the classic RAGAS field names (question/answer/contexts/
ground_truth) and the v2 renames (user_input/response/retrieved_contexts/
reference), so ragas-lingua drops into pipelines people already have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

# RAGAS field aliases -> canonical name.
_ALIASES = {
    "question": "question",
    "user_input": "question",
    "answer": "answer",
    "response": "answer",
    "contexts": "contexts",
    "retrieved_contexts": "contexts",
    "ground_truth": "ground_truth",
    "reference": "ground_truth",
}


@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    ground_truth: str | None = None
    language: str | None = None  # per-sample override of the run language

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "EvalSample":
        mapped: dict[str, Any] = {}
        for key, value in row.items():
            canon = _ALIASES.get(key)
            if canon is not None:
                mapped[canon] = value
        if "language" in row:
            mapped["language"] = row["language"]
        if "question" not in mapped or "answer" not in mapped:
            raise ValueError(
                "A sample needs a question/user_input and an answer/response; "
                f"got keys {sorted(row)}"
            )
        ctx = mapped.get("contexts") or []
        if isinstance(ctx, str):
            ctx = [ctx]
        mapped["contexts"] = list(ctx)
        return cls(**mapped)


@dataclass
class EvalDataset:
    samples: list[EvalSample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[EvalSample]:
        return iter(self.samples)

    @classmethod
    def from_dicts(cls, rows: Iterable[dict[str, Any]]) -> "EvalDataset":
        return cls([EvalSample.from_dict(r) for r in rows])

    @classmethod
    def from_ragas(cls, dataset: Any) -> "EvalDataset":
        """Accept a ragas EvaluationDataset, a HF Dataset, or any iterable of rows."""
        to_list = getattr(dataset, "to_list", None)
        if callable(to_list):
            return cls.from_dicts(to_list())
        return cls.from_dicts([dict(r) for r in dataset])
