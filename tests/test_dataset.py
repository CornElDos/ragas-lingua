import pytest

from ragas_lingua import EvalDataset, EvalSample


def test_from_dicts_canonical_keys():
    ds = EvalDataset.from_dicts(
        [
            {
                "question": "Vad är huvudstaden i Sverige?",
                "answer": "Stockholm",
                "contexts": ["Stockholm är Sveriges huvudstad."],
                "ground_truth": "Stockholm",
            }
        ]
    )
    assert len(ds) == 1
    sample = list(ds)[0]
    assert sample.question.startswith("Vad")
    assert sample.answer == "Stockholm"
    assert sample.contexts == ["Stockholm är Sveriges huvudstad."]
    assert sample.ground_truth == "Stockholm"


def test_from_dicts_ragas_v2_aliases():
    ds = EvalDataset.from_dicts(
        [
            {
                "user_input": "Was ist die Hauptstadt von Deutschland?",
                "response": "Berlin",
                "retrieved_contexts": ["Berlin ist die Hauptstadt Deutschlands."],
                "reference": "Berlin",
            }
        ]
    )
    sample = list(ds)[0]
    assert sample.question.startswith("Was")
    assert sample.answer == "Berlin"
    assert sample.contexts == ["Berlin ist die Hauptstadt Deutschlands."]
    assert sample.ground_truth == "Berlin"


def test_string_context_is_wrapped_in_list():
    sample = EvalSample.from_dict({"question": "q", "answer": "a", "contexts": "single"})
    assert sample.contexts == ["single"]


def test_per_sample_language_override():
    sample = EvalSample.from_dict({"question": "q", "answer": "a", "language": "de"})
    assert sample.language == "de"


def test_missing_required_field_raises():
    with pytest.raises(ValueError):
        EvalSample.from_dict({"question": "q"})
