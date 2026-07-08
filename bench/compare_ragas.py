"""Head-to-head: ragas-lingua vs RAGAS (default and adapt(sv)) on the gold set.

Scores faithfulness three ways for every gold item and reports how closely each
tracks the human labels (mean absolute error + Pearson). The point is to show
that language-native prompts beat RAGAS's prompt auto-translation on Swedish.

Requires ANTHROPIC_API_KEY and the RAGAS extras:

    pip install "ragas-lingua[ragas]"

Heads-up: RAGAS pulls a heavy, version-sensitive langchain dependency tree and
in a clean environment often fails to import until langchain versions are
aligned. This script is resilient to that — the `ours` column always runs; each
RAGAS column is skipped with a clear message if RAGAS can't be imported or run.

    python bench/compare_ragas.py [path/to/gold.jsonl]
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from pathlib import Path

from ragas_lingua import ClaudeJudge, EvalSample, Faithfulness, get_profile
from ragas_lingua.judge import DEFAULT_JUDGE_MODEL

DEFAULT_GOLD = Path(__file__).parent / "gold" / "faithfulness_sv.jsonl"


def load_gold(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return cov / (sx * sy)


def score_ours(gold: list[dict], model: str) -> dict[str, float]:
    judge = ClaudeJudge(model=model)
    metric = Faithfulness()
    scores: dict[str, float] = {}
    for item in gold:
        sample = EvalSample.from_dict(item)
        profile = get_profile(item.get("language", "sv"))
        scores[item["id"]] = metric.score(sample, judge=judge, profile=profile).score
    return scores


def score_ragas(gold: list[dict], model: str, adapt_language: str | None) -> dict[str, float]:
    """Score with RAGAS faithfulness. adapt_language=None => default prompts."""
    from langchain_anthropic import ChatAnthropic
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness as RagasFaithfulness

    llm = LangchainLLMWrapper(ChatAnthropic(model=model, temperature=0))
    metric = RagasFaithfulness(llm=llm)

    async def run() -> dict[str, float]:
        if adapt_language:
            adapted = await metric.adapt_prompts(language=adapt_language, llm=llm)
            metric.set_prompts(**adapted)
        scores: dict[str, float] = {}
        for item in gold:
            sample = SingleTurnSample(
                user_input=item["question"],
                response=item["answer"],
                retrieved_contexts=item.get("contexts", []),
                reference=item.get("ground_truth"),
            )
            scores[item["id"]] = await metric.single_turn_ascore(sample)
        return scores

    return asyncio.run(run())


def main(argv: list[str]) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the comparison (it calls the Claude API).")
        return 1

    gold_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_GOLD
    gold = load_gold(gold_path)
    model = os.environ.get("RAGAS_LINGUA_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    print(f"Faithfulness comparison · {len(gold)} items · model={model}\n")

    columns: dict[str, dict[str, float]] = {"ours": score_ours(gold, model)}
    for label, adapt in (("ragas", None), ("ragas-adapt", "swedish")):
        try:
            columns[label] = score_ragas(gold, model, adapt)
        except Exception as exc:  # noqa: BLE001 - RAGAS import/runtime is fragile by design
            print(f"[skip] RAGAS column '{label}' unavailable: {type(exc).__name__}: {exc}")
            print("       Needs a working RAGAS env: pip install 'ragas-lingua[ragas]'")

    methods = list(columns)
    header = f"{'id':<10} {'gold':>5} " + " ".join(f"{m:>12}" for m in methods)
    print("\n" + header)
    print("-" * len(header))

    pairs: dict[str, list[tuple[float, float]]] = {m: [] for m in methods}
    for item in gold:
        g = float(item["gold_faithfulness"])
        cells = []
        for m in methods:
            p = columns[m].get(item["id"], float("nan"))
            cells.append(f"{p:>12.2f}" if not math.isnan(p) else f"{'nan':>12}")
            if not math.isnan(p):
                pairs[m].append((g, p))
        print(f"{item['id']:<10} {g:>5.2f} " + " ".join(cells))

    print("-" * len(header))
    for m in methods:
        if pairs[m]:
            gs = [g for g, _ in pairs[m]]
            ps = [p for _, p in pairs[m]]
            mae = sum(abs(g - p) for g, p in pairs[m]) / len(pairs[m])
            print(f"{m:<12} MAE {mae:.3f} · Pearson r {pearson(gs, ps):.3f} · scored {len(pairs[m])}/{len(gold)}")
    print("\nLower MAE and higher Pearson = closer to human judgment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
