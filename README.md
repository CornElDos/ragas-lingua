# ragas-lingua

**Multilingual RAG evaluation with human-authored, language-native judge prompts — Swedish and German first, any language next.**

A [RAGAS](https://github.com/vibrantlabsai/ragas)-compatible evaluator that replaces
RAGAS's fragile prompt *auto-translation* with human-authored, language-native judge
prompts and Claude as the LLM-judge. Domain-general: works for any RAG application
(support, healthcare, finance, internal knowledge bases, legal), not one industry.

> **Status: alpha (M0 — scaffold).** The architecture, dataset layer and judge are in
> place; the metrics land next (M1). Not yet ready for real evaluation runs.

## Why

RAGAS evaluates non-English RAG by asking an LLM to translate its English judge prompts
into the target language and caching them (`adapt_prompts()`). That layer is unreliable —
statement extraction mixes English and the source language, and scores come out wrong.
It's a documented, recurring gap (e.g. RAGAS issues #1357 *German adapt bug*, #1298 Chinese,
#1949 evaluating other languages), and the RAGAS repo has stalled with 485 open issues.

`ragas-lingua` fixes it the honest way: **native prompts per language, not translated ones**,
plus a language-consistency guardrail that keeps extracted statements in the source language.

## Quickstart

```python
from ragas_lingua import EvalDataset, ClaudeJudge, evaluate
# from ragas_lingua.metrics import Faithfulness   # coming in M1

data = EvalDataset.from_dicts([
    {
        "question": "Vad är Sveriges huvudstad?",
        "answer": "Stockholm är Sveriges huvudstad.",
        "contexts": ["Stockholm är huvudstad i Sverige."],
        "ground_truth": "Stockholm",
    },
])

# Already using RAGAS? Bring your dataset straight over:
# data = EvalDataset.from_ragas(my_ragas_dataset)

result = evaluate(data, metrics=[...], judge=ClaudeJudge(), language="sv")
print(result.to_dict())
```

The dataset layer already accepts both classic RAGAS keys (`question`, `answer`,
`contexts`, `ground_truth`) and the v2 renames (`user_input`, `response`,
`retrieved_contexts`, `reference`), so it's a genuine drop-in.

## How it works

- **`LanguageProfile`** — the core asset: per-language, human-authored judge material.
- **`ClaudeJudge`** — the Claude Messages API with tool-based structured output,
  `temperature=0`. Swap in your own model via the small `Judge` protocol.
- **`metrics/`** — language-consistent reimplementations of `faithfulness`,
  `answer_correctness`, `context_precision`, `answer_relevancy` (M1–M3). RAGAS's scoring
  math is kept; only the prompts change.

## Languages

| Language | Code | Quality |
| --- | --- | --- |
| Swedish | `sv` | native-authored & verified |
| German | `de` | provisional — **native review wanted** before production use |

Adding a language is authoring a `LanguageProfile`, not writing code. Each language has its
own consistency tests under `tests/languages/`.

## Development

```bash
uv pip install -e ".[dev]"
pytest -q
ruff check src tests
```

Local evaluation data goes in `_private/` (git-ignored) — **never commit client or personal
data**; public examples are synthetic or public-domain.

Roadmap: see [`ROADMAP.md`](ROADMAP.md).

## License

MIT © Cornelii Sandberg
