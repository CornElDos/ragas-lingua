"""ragas-lingua — multilingual RAG evaluation with language-native judge prompts.

A RAGAS-compatible evaluator that replaces RAGAS's fragile prompt auto-translation
with human-authored, language-native judge prompts and Claude as the LLM-judge.
Swedish and German first, any language next.
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
from .metrics import Faithfulness, Metric, MetricResult

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
    "Faithfulness",
    "evaluate",
    "EvaluationResult",
]
