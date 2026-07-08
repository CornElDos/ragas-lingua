"""ClaudeJudge behaviour, exercised with a fake Anthropic client (no network).

The important case: a response truncated at max_tokens must raise, not silently
return a partial/empty structured result — otherwise a truncated judge looks
like a real 0.0 evaluation score.
"""

import pytest

from ragas_lingua import ClaudeJudge


class _Block:
    def __init__(self, type_, name, input_):
        self.type = type_
        self.name = name
        self.input = input_


class _Message:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class _Messages:
    def __init__(self, message):
        self._message = message
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._message


class _Client:
    def __init__(self, message):
        self.messages = _Messages(message)


def _judge_returning(message: _Message) -> ClaudeJudge:
    judge = ClaudeJudge()
    judge._client = _Client(message)  # bypass the real Anthropic client
    return judge


def test_raises_on_max_tokens_truncation():
    # Claude cut the tool call off; input is empty/partial.
    judge = _judge_returning(_Message("max_tokens", [_Block("tool_use", "record", {})]))
    with pytest.raises(RuntimeError, match="truncated"):
        judge.structured(system="s", user="u", schema={"type": "object"})


def test_returns_tool_input_on_normal_stop():
    judge = _judge_returning(_Message("tool_use", [_Block("tool_use", "record", {"score": 1})]))
    assert judge.structured(system="s", user="u", schema={"type": "object"}) == {"score": 1}


def test_raises_when_no_tool_use_block():
    judge = _judge_returning(_Message("end_turn", [_Block("text", None, None)]))
    with pytest.raises(RuntimeError, match="no tool_use block"):
        judge.structured(system="s", user="u", schema={"type": "object"})


def test_temperature_defaults_to_zero_and_is_forwarded():
    judge = _judge_returning(_Message("tool_use", [_Block("tool_use", "record", {"ok": 1})]))
    assert judge.temperature == 0.0
    judge.structured(system="s", user="u", schema={"type": "object"})
    assert judge._client.messages.last_kwargs["temperature"] == 0.0


def test_temperature_override_is_forwarded():
    judge = ClaudeJudge(temperature=0.7)
    judge._client = _Client(_Message("tool_use", [_Block("tool_use", "record", {"ok": 1})]))
    judge.structured(system="s", user="u", schema={"type": "object"})
    assert judge._client.messages.last_kwargs["temperature"] == 0.7
