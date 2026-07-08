"""Benchmark ragas-lingua faithfulness against a human-labelled gold set.

Runs the metric with a real ClaudeJudge over every item in the gold set and
reports how closely the predicted scores track the human `gold_faithfulness`
labels (mean absolute error + Pearson correlation).

Requires ANTHROPIC_API_KEY. Optionally set RAGAS_LINGUA_JUDGE_MODEL to pick the
judge model. Usage:

    python bench/run_faithfulness.py [path/to/gold.jsonl]
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

from ragas_lingua import ClaudeJudge, EvalSample, Faithfulness, get_profile

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


def main(argv: list[str]) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the benchmark (it calls the Claude API).")
        return 1

    gold_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_GOLD
    gold = load_gold(gold_path)
    judge = ClaudeJudge()
    metric = Faithfulness()

    print(f"Faithfulness benchmark · {len(gold)} items · judge={judge.model}\n")
    print(f"{'id':<10} {'domain':<11} {'gold':>5} {'pred':>6}  {'|err|':>5}")
    print("-" * 44)

    golds: list[float] = []
    preds: list[float] = []
    for item in gold:
        sample = EvalSample.from_dict(item)
        profile = get_profile(item.get("language", "sv"))
        result = metric.score(sample, judge=judge, profile=profile)
        g = float(item["gold_faithfulness"])
        p = result.score
        if math.isnan(p):
            print(f"{item['id']:<10} {item['domain']:<11} {g:>5.2f} {'nan':>6}  {'-':>5}")
            continue
        golds.append(g)
        preds.append(p)
        print(f"{item['id']:<10} {item['domain']:<11} {g:>5.2f} {p:>6.2f}  {abs(g - p):>5.2f}")

    if golds:
        mae = sum(abs(g - p) for g, p in zip(golds, preds)) / len(golds)
        r = pearson(golds, preds)
        print("-" * 44)
        print(f"scored {len(golds)}/{len(gold)} items · MAE {mae:.3f} · Pearson r {r:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
