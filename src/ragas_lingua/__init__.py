"""ragas-lingua — multilingual RAG evaluation with language-native judge prompts.

A RAGAS-compatible evaluator that replaces RAGAS's fragile prompt auto-translation
with human-authored, language-native judge prompts and Claude as the LLM-judge.
Swedish and German first, any language next.
"""

from .dataset import EvalDataset, EvalSample
from .diagnose import Diagnosis, Thresholds, diagnose, diagnose_all
from .evaluate import EvaluationResult, evaluate
from .judge import DEFAULT_JUDGE_MODEL, ClaudeJudge, FakeJudge, Judge
from .language import (
    GERMAN,
    SWEDISH,
    LanguageProfile,
    available_languages,
    get_profile,
    register_prompts_dir,
    reviewed_languages,
)
from .metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    Faithfulness,
    Metric,
    MetricResult,
)

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
    "reviewed_languages",
    "register_prompts_dir",
    "SWEDISH",
    "GERMAN",
    "Metric",
    "MetricResult",
    "Faithfulness",
    "AnswerCorrectness",
    "ContextPrecision",
    "AnswerRelevancy",
    "evaluate",
    "EvaluationResult",
    "diagnose",
    "diagnose_all",
    "Diagnosis",
    "Thresholds",
]
