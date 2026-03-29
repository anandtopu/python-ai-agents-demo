from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable


@dataclass
class ContextItem:
    kind: str
    text: str
    tags: set[str] = field(default_factory=set)
    source: str | None = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")


class ContextStore:
    def __init__(self) -> None:
        self._items: list[ContextItem] = []

    def add(self, kind: str, text: str, *, tags: Iterable[str] = (), source: str | None = None) -> None:
        self._items.append(ContextItem(kind=kind, text=text, tags=set(tags), source=source))

    def items(self) -> list[ContextItem]:
        return list(self._items)

    def as_compact_text(self, *, max_items: int = 30) -> str:
        tail = self._items[-max_items:]
        lines: list[str] = []
        for it in tail:
            tag_str = "" if not it.tags else f" tags={sorted(it.tags)}"
            src_str = "" if not it.source else f" source={it.source}"
            lines.append(f"- ({it.kind}){tag_str}{src_str}: {it.text}")
        return "\n".join(lines)

    def retrieve(self, query: str, *, k: int = 8) -> list[ContextItem]:
        q_tokens = _tokenize(query)
        scored: list[tuple[int, int, ContextItem]] = []
        for idx, it in enumerate(self._items):
            t_tokens = _tokenize(it.text) | set(it.tags)
            score = len(q_tokens & t_tokens)
            if score <= 0:
                continue
            scored.append((score, idx, it))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [it for _, _, it in scored[:k]]


def _tokenize(text: str) -> set[str]:
    raw = "".join(ch.lower() if (ch.isalnum() or ch in {"_", "-"}) else " " for ch in text)
    return {t for t in raw.split() if len(t) >= 3}


@dataclass(frozen=True)
class HandoffPacket:
    from_agent: str
    to_agent: str
    goal: str
    context: dict[str, Any]

    def to_user_message(self) -> str:
        return "HANDOFF_PACKET_JSON:\n" + json.dumps(
            {
                "from": self.from_agent,
                "to": self.to_agent,
                "goal": self.goal,
                "context": self.context,
            },
            ensure_ascii=False,
            indent=2,
        )


class ContextEngineer:
    def __init__(self, store: ContextStore | None = None) -> None:
        self.store = store or ContextStore()

    def inject_system_context(self) -> list[dict[str, Any]]:
        text = self.store.as_compact_text()
        if not text.strip():
            return []
        return [
            {
                "role": "system",
                "content": "Shared working context (use as authoritative; if conflicts, ask to clarify):\n" + text,
            }
        ]

    def add_user_goal(self, text: str) -> None:
        self.store.add("goal", text, tags={"goal"}, source="user")

    def add_decision(self, text: str, *, source: str | None = None) -> None:
        self.store.add("decision", text, tags={"decision"}, source=source)

    def add_artifact(self, text: str, *, path: str | None = None, source: str | None = None) -> None:
        tags = {"artifact"}
        if path:
            tags.add(path)
        self.store.add("artifact", text, tags=tags, source=source)

    def retrieve_text(self, query: str, *, k: int = 8) -> str:
        items = self.store.retrieve(query, k=k)
        if not items:
            return ""
        lines: list[str] = []
        for it in items:
            tag_str = "" if not it.tags else f" tags={sorted(it.tags)}"
            src_str = "" if not it.source else f" source={it.source}"
            lines.append(f"- ({it.kind}){tag_str}{src_str}: {it.text}")
        return "\n".join(lines)

    def make_handoff(
        self,
        *,
        from_agent: str,
        to_agent: str,
        goal: str,
        retrieve_query: str | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> HandoffPacket:
        retrieved = self.retrieve_text(retrieve_query or goal)
        ctx: dict[str, Any] = {
            "retrieved_context": retrieved,
            "store_snapshot": self.store.as_compact_text(max_items=25),
        }
        if extra_context:
            ctx["extra"] = extra_context
        return HandoffPacket(from_agent=from_agent, to_agent=to_agent, goal=goal, context=ctx)
