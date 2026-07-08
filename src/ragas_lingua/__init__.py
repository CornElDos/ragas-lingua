"""ragas-lingua — language-aware RAG evaluation for Swedish, German and the Nordics.

A RAGAS-compatible evaluator that replaces RAGAS's fragile prompt auto-translation
with human-authored, language-native judge prompts and Claude as the LLM-judge.
"""

from .dataset import EvalDataset, EvalSample
from .evaluate import EvaluationResult, evaluate
from .judge import DEFAULT_JUDGE_MODEL, ClaudeJudge, FakeJudge, Judge
from .language import (
    GERMAN,
    SWEDISH,
    LanguageProfile,
    available_languages,
    get_profile,
)
from .metrics.base import Metric, MetricResult

__version__ = "0.1.0"

__all__ = [
    "EvalDataset",
    "EvalSample",
    "ClaudeJudge",
    "FakeJudge",
    "Judge",
    "DEFAULT_JUDGE_MODEL",
    "LanguageProfile",
    "get_profile",
    "available_languages",
    "SWEDISH",
    "GERMAN",
    "Metric",
    "MetricResult",
    "evaluate",
    "EvaluationResult",
]
