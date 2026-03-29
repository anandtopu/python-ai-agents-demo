from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from openai import OpenAI

from context_engineering import ContextEngineer
from tools import TOOL_SPECS, run_tool


def _tools_for_openai() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.json_schema,
            },
        }
        for spec in TOOL_SPECS
    ]


@dataclass(frozen=True)
class Agent:
    name: str
    system_prompt: str


class TurnResult:
    assistant_text: str
    messages: list[dict[str, Any]]


class ToolLoopRunner:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._on_event = on_event

    def _emit(self, event: dict[str, Any]) -> None:
        if not self._on_event:
            return
        self._on_event(event)

    def run(self, messages: list[dict[str, Any]]) -> TurnResult:
        agent_name = _agent_name_from_thread(messages)
        while True:
            self._emit({"type": "llm_request", "agent": agent_name})
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=_tools_for_openai(),
                tool_choice="auto",
            )

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if tool_calls:
                self._emit(
                    {
                        "type": "assistant_message",
                        "agent": agent_name,
                        "content": msg.content or "",
                        "tool_calls": [tc.model_dump() for tc in tool_calls],
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [tc.model_dump() for tc in tool_calls],
                    }
                )

                for tc in tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")

                    self._emit(
                        {
                            "type": "tool_call",
                            "agent": agent_name,
                            "tool_call_id": tc.id,
                            "name": name,
                            "args": args,
                        }
                    )
                    result = run_tool(name, args)

                    self._emit(
                        {
                            "type": "tool_result",
                            "agent": agent_name,
                            "tool_call_id": tc.id,
                            "name": name,
                            "result": result,
                        }
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": json.dumps(result),
                        }
                    )

                continue

            assistant_text = msg.content or ""
            self._emit({"type": "assistant_message", "agent": agent_name, "content": assistant_text})
            messages.append({"role": "assistant", "content": assistant_text})
            return TurnResult(assistant_text=assistant_text, messages=messages)


class Orchestrator:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        context_engineer: ContextEngineer | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._runner = ToolLoopRunner(client=client, model=model, on_event=on_event)
        self._context_engineer = context_engineer

    def start_thread(self, agent: Agent) -> list[dict[str, Any]]:
        thread: list[dict[str, Any]] = [
            {"role": "system", "content": f"[{agent.name}] {agent.system_prompt}"},
        ]
        if self._context_engineer:
            thread.extend(self._context_engineer.inject_system_context())
        return thread

    def add_handoff(self, thread: list[dict[str, Any]], handoff_text: str) -> None:
        thread.append({"role": "user", "content": handoff_text})

    def ask(self, thread: list[dict[str, Any]], user_text: str) -> TurnResult:
        agent_name = _agent_name_from_thread(thread)
        self._runner._emit({"type": "user_message", "agent": agent_name, "content": user_text})
        thread.append({"role": "user", "content": user_text})
        return self._runner.run(thread)


def _agent_name_from_thread(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "unknown"
    first = messages[0]
    content = str(first.get("content", ""))
    if content.startswith("["):
        end = content.find("]")
        if end > 1:
            return content[1:end]
    return "unknown"
