from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from openai import OpenAI

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


@dataclass(frozen=True)
class TurnResult:
    assistant_text: str
    messages: list[dict[str, Any]]


class ToolLoopRunner:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._on_event = on_event

    def run(self, messages: list[dict[str, Any]]) -> TurnResult:
        while True:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=_tools_for_openai(),
                tool_choice="auto",
            )

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if tool_calls:
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
                    result = run_tool(name, args)

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
            messages.append({"role": "assistant", "content": assistant_text})
            return TurnResult(assistant_text=assistant_text, messages=messages)


class Orchestrator:
    def __init__(self, client: OpenAI, model: str) -> None:
        self._runner = ToolLoopRunner(client=client, model=model)

    def start_thread(self, agent: Agent) -> list[dict[str, Any]]:
        return [{"role": "system", "content": f"[{agent.name}] {agent.system_prompt}"}]

    def ask(self, thread: list[dict[str, Any]], user_text: str) -> TurnResult:
        thread.append({"role": "user", "content": user_text})
        return self._runner.run(thread)
