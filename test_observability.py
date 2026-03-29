from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from observability import InMemorySink, JsonlFileSink, Tracer, read_jsonl


class TestObservability(unittest.TestCase):
    def test_tracer_enriches_events(self) -> None:
        events: list[dict] = []
        tracer = Tracer(metadata={"demo": "x"}, sinks=[InMemorySink(events)])
        tracer.emit({"type": "hello", "value": 1})
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertIn("run_id", ev)
        self.assertIn("seq", ev)
        self.assertIn("ts", ev)
        self.assertEqual(ev["meta"]["demo"], "x")
        self.assertEqual(ev["type"], "hello")

    def test_jsonl_sink_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trace.jsonl"
            sink = JsonlFileSink(path)
            tracer = Tracer(run_id="rid", sinks=[sink])
            tracer.emit({"type": "a"})
            tracer.emit({"type": "b"})
            events = read_jsonl(path)
            self.assertEqual([e["type"] for e in events], ["a", "b"])
