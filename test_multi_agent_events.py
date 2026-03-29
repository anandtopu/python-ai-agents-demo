from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from typing import Any

try:
    import openai as _openai  # noqa: F401
except ModuleNotFoundError as e:  # pragma: no cover
    raise unittest.SkipTest("openai not installed; skipping multi_agent event tests") from e

from multi_agent import Agent, Orchestrator


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction

    def model_dump(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function", "function": {"name": self.function.name, "arguments": self.function.arguments}}


@dataclass
class _FakeMessage:
    content: str | None = None
    tool_calls: list[_FakeToolCall] | None = None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeUsage:
    prompt_tokens: int = 1
    completion_tokens: int = 1
    total_tokens: int = 2

    def model_dump(self) -> dict[str, Any]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens}


@dataclass
class _FakeResp:
    choices: list[_FakeChoice]
    usage: _FakeUsage | None = None


class _FakeChatCompletions:
    def __init__(self, responses: list[_FakeResp]) -> None:
        self._responses = responses
        self._idx = 0

    def create(self, **_kwargs: Any) -> _FakeResp:
        r = self._responses[self._idx]
        self._idx += 1
        return r


class _FakeChat:
    def __init__(self, responses: list[_FakeResp]) -> None:
        self.completions = _FakeChatCompletions(responses)


class _FakeOpenAI:
    def __init__(self, responses: list[_FakeResp]) -> None:
        self.chat = _FakeChat(responses)


class TestMultiAgentEvents(unittest.TestCase):
    def test_tool_call_event_flow(self) -> None:
        tool_call = _FakeToolCall(
            id="tc1",
            function=_FakeFunction(name="calculator", arguments=json.dumps({"expression": "1+2"})),
        )

        responses = [
            _FakeResp(choices=[_FakeChoice(message=_FakeMessage(content="", tool_calls=[tool_call]))], usage=_FakeUsage()),
            _FakeResp(choices=[_FakeChoice(message=_FakeMessage(content="done", tool_calls=None))], usage=_FakeUsage()),
        ]

        events: list[dict[str, Any]] = []
        client = _FakeOpenAI(responses)
        orch = Orchestrator(client, "fake", on_event=lambda ev: events.append(ev))

        agent = Agent(name="A", system_prompt="test")
        thread = orch.start_thread(agent)
        result = orch.ask(thread, "compute")

        self.assertEqual(result.assistant_text, "done")

        types = [e.get("type") for e in events]
        self.assertIn("user_message", types)
        self.assertIn("llm_request", types)
        self.assertIn("llm_response", types)
        self.assertIn("tool_call", types)
        self.assertIn("tool_result", types)

        tool_results = [e for e in events if e.get("type") == "tool_result"]
        self.assertTrue(tool_results)
        self.assertEqual(tool_results[0]["name"], "calculator")
        self.assertIn("duration_ms", tool_results[0])
