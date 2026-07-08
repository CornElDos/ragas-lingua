# ragas-lingua

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

## How it works

- **`LanguageProfile`** — the core asset: per-language, human-authored judge material.
- **`ClaudeJudge`** — the Claude Messages API with tool-based structured output,
  `temperature=0`. The judge is pluggable via a tiny `Judge` protocol, so you can swap in
  another model — including a **local** one, if your data can't leave your machine.
- **`metrics/`** — language-consistent reimplementations of the four core RAGAS metrics.

## Languages

| Language | Code | Status |
| --- | --- | --- |
| Swedish | `sv` | native-authored & verified — all four metrics |
| German | `de` | language profile exists; **metric prompts not authored yet** (native review wanted) |

Adding a language is authoring prompts, not writing code. Each language has its own
consistency tests under `tests/languages/`.

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

## Development

```bash
uv pip install -e ".[dev]"
pytest -q
ruff check src tests bench
```

## License

MIT © Cornelii Sandberg
