from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol


def _utc_ts() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


class EventSink(Protocol):
    def write(self, event: dict[str, Any]) -> None: ...


@dataclass
class InMemorySink:
    events: list[dict[str, Any]]

    def write(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class JsonlFileSink:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False)
        with self._path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")


class Tracer:
    def __init__(
        self,
        *,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        sinks: list[EventSink] | None = None,
    ) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        self._seq = 0
        self._sinks = sinks or []

    def emit(self, event: dict[str, Any]) -> None:
        self._seq += 1
        enriched = {
            "run_id": self.run_id,
            "seq": self._seq,
            "ts": _utc_ts(),
            "meta": self.metadata,
            **event,
        }
        for s in self._sinks:
            s.write(enriched)

    @staticmethod
    def default_trace_path(run_id: str) -> Path:
        return Path("traces") / f"{run_id}.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events
