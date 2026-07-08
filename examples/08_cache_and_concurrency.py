"""Cache judge calls and run them concurrently, offline with FakeJudge (no key).

Two knobs for evaluating at scale:

- CachingJudge memoises deterministic (temperature 0) judge calls, so re-running
  the same dataset costs nothing the second time. Pass path=... to persist it
  across processes.
- evaluate(..., max_concurrency=N) overlaps the I/O-bound judge calls in a
  thread pool. With a real ClaudeJudge that turns a serial crawl into N-at-a-time.

    python examples/08_cache_and_concurrency.py
"""

from ragas_lingua import CachingJudge, EvalDataset, Faithfulness, FakeJudge, evaluate

data = EvalDataset.from_dicts(
    [
        {
            "question": "Vad är Sveriges huvudstad?",
            "answer": "Stockholm är Sveriges huvudstad.",
            "contexts": ["Stockholm är huvudstad i Sverige."],
        },
    ]
)


def _handler(*, system, user, schema):
    # Faithfulness makes two calls: extract statements, then judge each one.
    if "statements" in schema.get("properties", {}):
        return {"statements": ["Stockholm är Sveriges huvudstad."]}
    return {"verdicts": [{"statement": "Stockholm är Sveriges huvudstad.", "supported": True}]}


def main() -> None:
    inner = FakeJudge(handler=_handler)
    judge = CachingJudge(inner)

    evaluate(data, [Faithfulness()], judge=judge, language="sv", max_concurrency=4)
    after_first = len(inner.calls)
    evaluate(data, [Faithfulness()], judge=judge, language="sv", max_concurrency=4)
    after_second = len(inner.calls)

    print(f"inner judge calls after run 1: {after_first}")
    print(f"inner judge calls after run 2: {after_second}  (unchanged = fully cached)")
    print("cache stats:", judge.stats())


if __name__ == "__main__":
    main()
