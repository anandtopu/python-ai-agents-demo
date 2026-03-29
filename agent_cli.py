from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_SPECS, run_tool
from multi_agent_demo import run_demo


_HISTORY_PATH = Path("agent_history.json")
_SESSIONS_DIR = Path("sessions")


def _sanitize_session_name(name: str) -> str:
    cleaned = "".join(ch for ch in name.strip() if ch.isalnum() or ch in {"-", "_"})
    return cleaned[:64]


def _session_path(name: str) -> Path:
    safe = _sanitize_session_name(name)
    if not safe:
        raise ValueError("Invalid session name")
    return _SESSIONS_DIR / f"{safe}.json"


def _list_sessions() -> list[str]:
    if not _SESSIONS_DIR.exists():
        return []
    names: list[str] = []
    for p in _SESSIONS_DIR.glob("*.json"):
        names.append(p.stem)
    names.sort()
    return names


def _load_history(default_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not _HISTORY_PATH.exists():
        return default_messages
    try:
        data = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return default_messages


def _save_history(messages: list[dict[str, Any]]) -> None:
    _HISTORY_PATH.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_session(messages: list[dict[str, Any]], name: str) -> Path:
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_path(name)
    path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_session(name: str) -> list[dict[str, Any]]:
    path = _session_path(name)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Invalid session format")
    return data


def _summarize_history(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    keep_last: int = 12,
) -> list[dict[str, Any]]:
    if len(messages) <= keep_last + 1:
        return messages

    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    if not system_msg:
        system_msg = {
            "role": "system",
            "content": "You are a helpful CLI agent. If a tool helps, call it. Be concise.",
        }

    tail = messages[-keep_last:]
    head = messages[1:-keep_last]
    compact_input = json.dumps(head, ensure_ascii=False)

    summary_resp = client.chat.completions.create(
        model=model,
        messages=[
            system_msg,
            {
                "role": "user",
                "content":
                "Summarize the following prior conversation messages into a compact memory for future context. "
                "Keep key user preferences, goals, decisions, and any important facts. Be concise.\n\n"
                + compact_input,
            },
        ],
        tools=_tools_for_openai(),
        tool_choice="none",
    )

    summary_text = summary_resp.choices[0].message.content or ""

    summarized_messages: list[dict[str, Any]] = [
        system_msg,
        {"role": "system", "content": f"Conversation summary (memory):\n{summary_text}"},
    ]
    summarized_messages.extend(tail)
    return summarized_messages


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


def main() -> None:
    load_dotenv(override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing OPENAI_API_KEY. Ensure you have a .env file with OPENAI_API_KEY set. "
            "(Note: this app loads .env with override=True.)"
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    default_messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a helpful CLI agent. If a tool helps, call it. Be concise.",
        }
    ]

    messages = _load_history(default_messages)

    print("First Agent (Python) — type 'exit' to quit")
    print(
        "Commands: /reset, /save <name>, /load <name>, /sessions, /delete <name>, /summary, /ma <demo>"
    )

    while True:
        user_text = input("> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        if user_text.lower() == "/reset":
            messages = default_messages.copy()
            _save_history(messages)
            print("History cleared.")
            continue

        if user_text.lower().startswith("/save"):
            parts = user_text.split(maxsplit=1)
            if len(parts) != 2:
                print("Usage: /save <name>")
                continue
            try:
                path = _save_session(messages, parts[1])
                print(f"Saved session: {path.as_posix()}")
            except (OSError, ValueError) as e:
                print(f"Save failed: {e}")
            continue

        if user_text.lower().startswith("/load"):
            parts = user_text.split(maxsplit=1)
            if len(parts) != 2:
                print("Usage: /load <name>")
                continue
            try:
                loaded = _load_session(parts[1])
                messages = loaded
                _save_history(messages)
                print("Session loaded.")
            except (OSError, json.JSONDecodeError, ValueError) as e:
                print(f"Load failed: {e}")
            continue

        if user_text.lower() == "/sessions":
            sessions = _list_sessions()
            if not sessions:
                print("No saved sessions.")
            else:
                print("Saved sessions:")
                for s in sessions:
                    print(f"- {s}")
            continue

        if user_text.lower().startswith("/delete"):
            parts = user_text.split(maxsplit=1)
            if len(parts) != 2:
                print("Usage: /delete <name>")
                continue
            try:
                path = _session_path(parts[1])
                path.unlink(missing_ok=False)
                print("Session deleted.")
            except FileNotFoundError:
                print("Delete failed: session not found")
            except (OSError, ValueError) as e:
                print(f"Delete failed: {e}")
            continue

        if user_text.lower() == "/summary":
            try:
                messages = _summarize_history(client, model, messages)
                _save_history(messages)
                print("History summarized.")
            except Exception as e:
                print(f"Summary failed: {e}")
            continue

        if user_text.lower().startswith("/ma"):
            parts = user_text.split(maxsplit=1)
            if len(parts) != 2:
                print(
                    "Usage: /ma <demo>  (demos: research | code_review | debate | context | cse | lc_structured | lg_hitl | complex | context_limits | langsmith)"
                )
                continue
            try:
                result = run_demo(client, model, parts[1])
                print(f"Multi-agent demo complete: {result.demo}")
                for k, v in result.outputs.items():
                    print(f"\n--- {k.upper()} ---\n{v}\n")
            except Exception as e:
                print(f"Multi-agent demo failed: {e}")
            continue

        messages.append({"role": "user", "content": user_text})
        _save_history(messages)

        while True:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=_tools_for_openai(),
                tool_choice="auto",
            )

            choice = resp.choices[0]
            msg = choice.message

            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [tc.model_dump() for tc in tool_calls],
                    }
                )
                _save_history(messages)

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
                    _save_history(messages)

                continue

            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=_tools_for_openai(),
                tool_choice="none",
                stream=True,
            )

            assistant_text_parts: list[str] = []
            for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                chunk = getattr(delta, "content", None)
                if chunk:
                    assistant_text_parts.append(chunk)
                    print(chunk, end="", flush=True)

            print("")
            assistant_text = "".join(assistant_text_parts)
            messages.append({"role": "assistant", "content": assistant_text})
            _save_history(messages)
            break


if __name__ == "__main__":
    main()
