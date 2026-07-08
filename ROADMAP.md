# ragmål — language-aware RAG evaluation

**Project plan · draft v1**

> A Nordic-first, drop-in evaluator that fixes what RAGAS gets wrong on non-English
> text: hand-written, language-native judge prompts with Claude as the LLM-judge,
> instead of RAGAS's fragile auto-translation.

- **For:** Cornelii Sandberg
- **Stack:** Python · Claude API · uv
- **Effort:** project (days–weeks)
- **Blocker:** none — buildable today
- **Rendered plan:** ./plan.html · https://claude.ai/code/artifact/25028594-0347-49e8-8d3b-d25894c31965
- **RAGAS reference clone:** ~/Dokument/OpenSource/ragas-ref/

---

## 01 · The problem

RAGAS (`vibrantlabsai/ragas`, 14.7k★) evaluates non-English RAG by asking an LLM to
*auto-translate* its English judge prompts into the target language and caching them as
`{metric}_{language}.json` (`adapt_prompts()` in `src/ragas/prompt/mixin.py`). That
translation layer is where it breaks.

| Area | State | Evidence (open issues) |
|---|---|---|
| `adapt()` prompt translation | **Broken** | #1357 German adapt bug · #1925 `await` errors · #1298/#1296 Chinese adapt "no useful" |
| Statement extraction (faithfulness, answer_correctness) | **Broken** | mixes English & source language → wrong claims → mis-scored |
| Evaluating in other languages | **Unsolved** | #1949 "how to evaluate rags in other language" · #1938 "Language of the Testset" |
| Multilingual testset generation | **Broken** | #1769 "generated testset is empty" (KG-based generator) |
| Project maintenance | **Stalled** | no push since 2026-02-24 · 485 open issues · 11 open language tickets |

The stalled incumbent is *why we build our own* rather than sending a large PR into an
unreviewed queue — an open, unmet gap in the most-used RAG-eval tool.

## 02 · Thesis & approach

Don't translate English prompts — author judge prompts *natively* per language, run them
through a stronger judge, and enforce that extracted statements stay in the source
language. Ship it RAGAS-compatible so it drops into pipelines people already have.

- **Language-native prompts, not auto-translated.** Each metric gets human-authored
  instructions + few-shot examples per language (Swedish first), reviewed by a native
  speaker — *you*.
- **Claude as the judge.** Claude API via the Messages API with **tool-based structured
  output** for statement lists and verdicts; `temperature 0`; prompt caching on the long
  instructions to keep cost down. Judge stays pluggable.
- **Language-consistency guardrail.** Extraction is constrained to emit claims in the
  answer's own language, and a lightweight language-ID check flags cross-language leakage
  before scoring.
- **Reuse RAGAS's math, replace its prompts.** Keep the proven score formulas
  (TP/FP/FN → F1, precision@k) — the defect is linguistic, not arithmetic.

## 03 · Architecture

A thin library around one call: `evaluate(dataset, metrics, judge=Claude, language="sv")`.

- **LanguageProfile** — *the core asset.* Per-language pack: native instruction strings,
  few-shot examples, number/date norms, and the "keep output in source language"
  constraint. `sv` ships first; profiles are data, so adding `no`/`da`/`fi` is authoring,
  not coding.
- **ClaudeJudge** — judge adapter over the Claude Messages API: structured output via
  tools, temperature 0, caching, retries. Conforms to a small `Judge` protocol so users
  can swap in their own model.
- **metrics/** — language-consistent reimplementations: `faithfulness`,
  `answer_correctness`, `context_precision`, `answer_relevancy`. Each = native extraction
  prompt → judge → RAGAS-equivalent scoring.
- **ragas_compat** — accepts RAGAS-style samples (`question`, `answer`, `contexts`,
  `ground_truth`) and returns the same score keys — a genuine drop-in.
- **bench/** — reproducible comparison: our scores vs RAGAS-default vs RAGAS-adapt on the
  Swedish gold set, with human-agreement numbers.

## 04 · Metric scope — phased

| Metric | Why it breaks multilingually | Phase |
|---|---|---|
| `faithfulness` | statement extraction leaks into English → claims don't match source → false hallucination flags | P1 |
| `answer_correctness` | TP/FP/FN classification mis-fires when either side is mis-extracted | P1 |
| `context_precision` | per-chunk relevance verdicts degrade with translated prompts | P2 |
| `answer_relevancy` | generated "reverse questions" come out in the wrong language, skewing similarity | P2 |
| testset generation | KG-based generator yields empty/degenerate sets for non-English (#1769) | P3 (stretch) |

## 05 · Roadmap

Effort in focused days (ideal, part-time-friendly). Each milestone ends in something demoable.

| Milestone | Outcome | Effort |
|---|---|---|
| M0 · Scaffold | repo, `uv` setup, `Judge` protocol + ClaudeJudge, RAGAS-compat dataset IO, CI | 1–2d |
| M1 · faithfulness (sv) | native Swedish prompts + language guardrail + **gold set** + benchmark vs RAGAS | 2–3d |
| M2 · answer_correctness (sv) | statement classification, extends gold set & benchmark | 2d |
| M3 · precision + relevancy | the other two P2 metrics, sv | 2d |
| M4 · generalize + ship | add `no`/`da`/`fi` profiles, README, docs, **blog post**, RAGAS issue/PR link | 2–3d |
| M5 · testset gen | later — multilingual generation (hard, optional stretch) | stretch |

## 06 · Validation — the Swedish proof

The benchmark *is* the credibility. Show, with numbers, that native prompts beat RAGAS on Swedish.

- Build a small **gold set** (~50–100 items) in Swedish: answer/context pairs with
  known-supported and known-hallucinated claims, plus a few `no`/`da` items. Sources:
  synthetic + public-domain (Wikipedia, public court rulings) — **never** Vasa client material.
- Report three columns per metric: **ragmål** vs **RAGAS-default** vs **RAGAS-adapt(sv)**,
  scored against human labels — the LettuceDetect-style "F1 vs baseline" table.
- Success bar: ragmål correlates with human judgment on Swedish and beats both RAGAS modes;
  leakage guardrail catches cross-language extraction the others miss.

> **Privacy — non-negotiable.** Vasa data almost certainly contains client and personal
> data. Use it **only** for private sanity checks on your own machine; it must never be
> committed, logged, or shipped as an example. Everything public in the repo is synthetic
> or public-domain.

## 07 · Risks & guardrails

| Risk | Mitigation |
|---|---|
| Client/PII leaking into a public repo | strict rule above; `.gitignore`d `_private/`; pre-commit secret/PII scan |
| Judge cost & non-determinism | `temperature 0`, prompt caching, response caching in bench, documented per-run cost |
| Scope creep into testset generation | hard-park as M5; P1–P4 stand alone as a shippable tool |
| "Drop-in" drift as RAGAS changes | pin a tested RAGAS version; keep the compat adapter thin and tested |
| Only one language reviewer (you) | sv is native; for `fi`/others, mark profiles "community-review wanted" |

## 08 · Deliverables

- **repo** — Public MIT library; README leads with the problem + benchmark table, a
  10-line Swedish quickstart, and honest scope.
- **benchmark** — reproducible `bench/` showing ragmål vs RAGAS on Swedish.
- **write-up** — a short blog post / LinkedIn piece (the brand payload): "Why RAG
  evaluation is broken in Swedish, and a fix."
- **visibility** — one focused issue/PR to RAGAS referencing the lib.

## 09 · Decisions needed

1. **Name** — proposed **ragmål** (`rag` + Swedish *mål* = "measure/target", and "court
   case": a legal-tech wink). Alternatives: nordeval, lagom-eval.
2. **Languages after Swedish** — recommend `sv → no → da → fi` (fi is a good distant
   stress test but not your native → community-review).
3. **Standalone only, or + RAGAS PR?** — recommend standalone first, then a small
   companion issue/PR for reach.
4. **Default judge model** — a current Claude model (Sonnet-class), pluggable.
5. **License** — MIT (matches RAGAS, maximizes adoption) unless you want copyleft.

---

*ragmål · project plan v1 · grounded in the current `vibrantlabsai/ragas` source.
Next step on go: scaffold M0.*
