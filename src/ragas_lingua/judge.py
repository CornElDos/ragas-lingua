"""The LLM-as-judge layer.

`Judge` is a tiny protocol: given a system prompt, a user prompt and a JSON
schema, return a JSON object matching the schema. `ClaudeJudge` implements it
with the Anthropic Messages API using tool-based structured output. `FakeJudge`
is a deterministic stand-in for tests (no network, no API key).
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

# Override with the env var; see Anthropic's docs for the current Sonnet model id.
DEFAULT_JUDGE_MODEL = os.environ.get("RAGAS_LINGUA_JUDGE_MODEL", "claude-sonnet-4-5")

# Cap on the judge's structured OUTPUT (statement/verdict lists), not your RAG
# input. It's a ceiling, not a target — you're billed for tokens generated, not
# this limit — so a generous default just avoids truncation on long answers.
# Raise it (or set the env var) if you evaluate very long answers / many contexts.
DEFAULT_MAX_TOKENS = int(os.environ.get("RAGAS_LINGUA_JUDGE_MAX_TOKENS", "8192"))


@runtime_checkable
class Judge(Protocol):
    """An LLM-as-judge that returns structured JSON conforming to a schema."""

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        tool_name: str = "record",
    ) -> dict[str, Any]: ...


class ClaudeJudge:
    """Judge backed by the Anthropic Claude Messages API.

    Uses a forced tool call so the model must return a JSON object matching
    ``schema``. Deterministic (``temperature=0``). The ``anthropic`` package is
    imported lazily, so importing ragas-lingua never requires it.
    """

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        *,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._client: Any = None

    def _client_or_init(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - trivial guard
                raise ImportError(
                    "ClaudeJudge needs the `anthropic` package: pip install anthropic"
                ) from exc
            self._client = (
                anthropic.Anthropic(api_key=self._api_key)
                if self._api_key
                else anthropic.Anthropic()
            )
        return self._client

    def structured(
        self, *, system: str, user: str, schema: dict[str, Any], tool_name: str = "record"
    ) -> dict[str, Any]:
        client = self._client_or_init()
        tool = {
            "name": tool_name,
            "description": "Record the structured evaluation result.",
            "input_schema": schema,
        }
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user}],
        )
        # Critical for an eval tool: if the response was cut off at max_tokens,
        # the tool_use block's `input` is empty/partial. Returning it silently
        # would report a plausible-looking but wrong score (e.g. 0/N), which
        # reads as "the RAG hallucinated everything" when the judge was merely
        # truncated. Fail loudly instead.
        if getattr(message, "stop_reason", None) == "max_tokens":
            raise RuntimeError(
                f"Judge response was truncated at max_tokens={self.max_tokens}; the "
                f"structured result is incomplete. Raise it via "
                f"ClaudeJudge(max_tokens=...) or the RAGAS_LINGUA_JUDGE_MAX_TOKENS env var."
            )
        for block in message.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                return dict(block.input)
        raise RuntimeError("Judge returned no tool_use block")


class FakeJudge:
    """Deterministic judge for tests: pops scripted responses or calls a handler."""

    def __init__(self, responses: list[dict[str, Any]] | None = None, handler: Any = None) -> None:
        self._responses = list(responses or [])
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def structured(
        self, *, system: str, user: str, schema: dict[str, Any], tool_name: str = "record"
    ) -> dict[str, Any]:
        self.calls.append({"system": system, "user": user, "schema": schema})
        if self._handler is not None:
            return self._handler(system=system, user=user, schema=schema)
        if self._responses:
            return self._responses.pop(0)
        raise AssertionError("FakeJudge ran out of scripted responses")
