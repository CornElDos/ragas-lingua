# ragas-lingua

[![CI](https://github.com/CornElDos/ragas-lingua/actions/workflows/ci.yml/badge.svg)](https://github.com/CornElDos/ragas-lingua/actions/workflows/ci.yml)

**Multilingual RAG evaluation with human-authored, language-native judge prompts — Swedish first, any language next.**

A [RAGAS](https://github.com/vibrantlabsai/ragas)-compatible evaluator that replaces
RAGAS's fragile prompt *auto-translation* with human-authored, language-native judge
prompts and Claude as the LLM-judge. Domain-general: works for any RAG application
(support, healthcare, finance, internal knowledge bases, legal) — not one industry.

> **Status: alpha.** The four core metrics below are implemented and unit-tested for
> Swedish. APIs may still change. Not on PyPI yet — install from source.

## Why

RAGAS evaluates non-English RAG by asking an LLM to translate its English judge prompts
into the target language and caching them (`adapt_prompts()`). That layer is unreliable —
statement extraction mixes English and the source language, and scores come out wrong.
It's a documented, recurring gap (e.g. RAGAS issues #1357 *German adapt bug*, #1298 Chinese,
#1949 evaluating other languages), and the RAGAS repo has stalled with 485 open issues.

`ragas-lingua` fixes it the honest way: **native prompts per language, not translated ones**,
plus a language-consistency guardrail that keeps extracted statements in the source language.

## Install

```bash
pip install git+https://github.com/CornElDos/ragas-lingua.git
export ANTHROPIC_API_KEY=...
```

Core dependencies are just `anthropic` and `pydantic` — **RAGAS is not required** (it's an
optional extra used only by the comparison benchmark).

## Quickstart

```python
from ragas_lingua import (
    EvalDataset, ClaudeJudge, evaluate,
    Faithfulness, AnswerCorrectness, ContextPrecision, AnswerRelevancy,
)

data = EvalDataset.from_dicts([
    {
        "question": "Vad är Sveriges huvudstad?",
        "answer": "Stockholm är Sveriges huvudstad.",
        "contexts": ["Stockholm är huvudstad i Sverige."],
        "ground_truth": "Stockholm är Sveriges huvudstad.",
    },
    # ... your app's RAG outputs
])

result = evaluate(
    data,
    metrics=[Faithfulness(), AnswerCorrectness(), ContextPrecision(), AnswerRelevancy()],
    judge=ClaudeJudge(),
    language="sv",
)
print(result.to_dict())
# {'faithfulness': 1.0, 'answer_correctness': 0.94, 'context_precision': 1.0, 'answer_relevancy': 0.88}
```

Already using RAGAS? Bring your dataset straight over — the dataset layer accepts both the
classic RAGAS keys (`question`, `answer`, `contexts`, `ground_truth`) and the v2 renames
(`user_input`, `response`, `retrieved_contexts`, `reference`):

```python
data = EvalDataset.from_ragas(my_ragas_dataset)
```

## Metrics

| Metric | What it measures | Needs |
| --- | --- | --- |
| `Faithfulness` | fraction of the answer's claims that the context supports | contexts |
| `AnswerCorrectness` | factual F1 vs the reference (TP/FP/FN) + semantic similarity | ground_truth |
| `ContextPrecision` | are the useful contexts ranked near the top (average precision) | contexts |
| `AnswerRelevancy` | does the answer actually address the question | — |

Each metric extracts/judges in the answer's own language and keeps RAGAS's scoring math;
only the prompts change. A metric run against a language whose prompts aren't authored yet
raises `NotImplementedError` rather than silently guessing.

## Diagnosis

A single low number can't tell you whether the tool or your RAG is at fault. `diagnose()`
labels each sample from the *combination* of metrics — it classifies, it doesn't advise:

```python
from ragas_lingua import evaluate, diagnose_all, ClaudeJudge, Faithfulness, AnswerCorrectness, ContextPrecision

result = evaluate(data, [Faithfulness(), AnswerCorrectness(), ContextPrecision()],
                  judge=ClaudeJudge(), language="sv")
for d in diagnose_all(result):
    print(d.summary())
# retrieval_gap [noisy_retrieval]  (faithfulness=0.28 answer_correctness=0.90 context_precision=0.40)
#   The answer is largely correct, but its claims are not supported by the retrieved context …
```

| Label | When | Meaning |
| --- | --- | --- |
| `well_grounded` | faithfulness high | grounded in the context |
| `retrieval_gap` | faithfulness low, answer_correctness high | correct but ungrounded (the support was not retrieved) |
| `hallucination` | faithfulness low, answer_correctness low | ungrounded *and* wrong |
| `partial_retrieval` | mixed correctness, ungrounded, low context_precision | the bottleneck is retrieval |
| `partial_generation` | mixed correctness, ungrounded, high context_precision | good context, but the answer does not ground in it |
| `partial` | mixed correctness, context_precision unknown | can't attribute yet |
| `ungrounded_unverified` | faithfulness low, no ground_truth | add a reference to disambiguate |

Plus flags (`noisy_retrieval`, `off_topic`) and the exact ungrounded statements as evidence.
Thresholds are configurable via `Thresholds(high=..., low=...)`.

## Confidence

An LLM judge is noisy — sample it twice and the score moves. Most eval tools
report a single number as if it were exact. `score_with_confidence` runs a metric
several times and reports the mean *and* how much it wobbled, so you know how far
to trust it:

```python
from ragas_lingua import ClaudeJudge, Faithfulness, score_with_confidence, get_profile

# sample above temperature 0 so the runs actually vary
mc = score_with_confidence(
    Faithfulness(), sample,
    judge=ClaudeJudge(temperature=0.7), profile=get_profile("sv"), runs=5,
)
print(mc.summary())
# faithfulness: 0.61 +/- 0.09 (medium confidence, n=5)
```

The spread maps to a `high` / `medium` / `low` label (cutoffs configurable via
`ConfidenceBands`). Low confidence means the judge disagreed with itself — the
number is soft, not a verdict. `evaluate_with_confidence(dataset, metrics, ...)`
does this for a whole dataset, returning one row per sample to line up with
`EvaluationResult.per_sample`. Runs that come back NaN are dropped; all-NaN reports
`unknown`. At `temperature=0` every run is identical, so it warns.

## How it works

- **`LanguageProfile`** — the core asset: per-language, human-authored judge material.
- **`ClaudeJudge`** — the Claude Messages API with tool-based structured output,
  `temperature=0` by default (raise it for confidence sampling). The judge is pluggable
  via a tiny `Judge` protocol, so you can swap in another model — including a **local**
  one, if your data can't leave your machine.
- **`metrics/`** — language-consistent reimplementations of the four core RAGAS metrics.

**Long answers?** `ClaudeJudge(max_tokens=...)` (default `8192`, or the
`RAGAS_LINGUA_JUDGE_MAX_TOKENS` env var) caps the judge's structured output — the
statement/verdict lists it produces, not your RAG input. If your RAG returns very long
answers or many contexts and results look truncated, raise it. It's a ceiling, not a
cost — you pay for tokens generated, not the limit — so a higher cap only helps.

## Languages

| Language | Code | Status |
| --- | --- | --- |
| Swedish | `sv` | native-authored & verified |
| German | `de` | auto-generated — native review wanted |
| Norwegian | `no` | auto-generated — native review wanted |
| Danish | `da` | auto-generated — native review wanted |
| Finnish | `fi` | auto-generated — **careful** native review needed |
| Icelandic | `is` | auto-generated — **careful** native review needed |

**Adding a language is dropping a file.** Each language is one self-contained
`ragas_lingua/prompts/<code>.toml` (metadata + the keep-language directive + a section per
metric) — no code changes. Point `register_prompts_dir("path")` at your own directory to
add languages from outside the package. Auto-generated packs carry `reviewed = false` and
emit a warning at runtime; see [`docs/AUTOGENERATED_TRANSLATIONS.md`](docs/AUTOGENERATED_TRANSLATIONS.md).

## Privacy

Running the metrics sends your evaluation data (questions, answers, contexts) to the judge —
by default the Claude API. If that data is sensitive (client or personal data), mind your
provider's data-processing terms, keep it out of public repos and logs, and consider
pointing the `Judge` protocol at a local model. Local eval data belongs in `_private/`
(git-ignored); public examples in this repo are synthetic or public-domain.

## Benchmarks

```bash
# our metric vs the human-labelled Swedish gold set
python bench/run_faithfulness.py

# head-to-head: ours vs RAGAS-default vs RAGAS-adapt(sv)  (needs the [ragas] extra)
python bench/compare_ragas.py
```

Both need `ANTHROPIC_API_KEY`. See [`ROADMAP.md`](ROADMAP.md) for what's next.

## Examples

Runnable scripts in [`examples/`](examples/):

| File | Shows | Needs API key |
| --- | --- | --- |
| `01_quickstart.py` | evaluate a dataset with all four metrics | yes |
| `02_diagnose.py` | metrics + `diagnose_all()` failure labels | yes |
| `03_bring_your_own_rag.py` | wire your RAG pipeline into `EvalDataset` | yes |
| `04_from_ragas.py` | migrate a RAGAS dataset via the field aliases | no |
| `05_add_a_language.py` | add a language with `register_prompts_dir()` | no |
| `06_offline_with_fakejudge.py` | run the whole flow offline with `FakeJudge` | no |
| `07_confidence.py` | `score_with_confidence()` — mean +/- spread over repeated runs | no |

## Development

```bash
uv pip install -e ".[dev]"
pytest -q
ruff check src tests bench examples
```

## License

MIT © Cornelii Sandberg
