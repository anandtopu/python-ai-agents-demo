from __future__ import annotations

import json
import unittest

from context_engineering import ContextEngineer, ContextStore


class TestContextEngineering(unittest.TestCase):
    def test_store_add_and_compact_text(self) -> None:
        store = ContextStore()
        store.add("goal", "Ship feature X", tags={"goal"}, source="user")
        store.add("decision", "Use sqlite", tags={"decision"}, source="team")
        text = store.as_compact_text()
        self.assertIn("Ship feature X", text)
        self.assertIn("Use sqlite", text)
        self.assertIn("goal", text)
        self.assertIn("decision", text)

    def test_retrieve_prefers_overlap(self) -> None:
        store = ContextStore()
        store.add("note", "Intervals should merge adjacency", tags={"interval", "merge"})
        store.add("note", "Unrelated build logs", tags={"ci", "lint"})
        items = store.retrieve("merge intervals", k=5)
        self.assertTrue(items)
        self.assertIn("Intervals", items[0].text)

    def test_handoff_packet_is_valid_json(self) -> None:
        ce = ContextEngineer()
        ce.add_user_goal("Decide adjacency merge")
        ce.add_decision("Adjacency is merged", source="trusted")
        packet = ce.make_handoff(from_agent="A", to_agent="B", goal="Decide", retrieve_query="adjacency")
        msg = packet.to_user_message()
        self.assertTrue(msg.startswith("HANDOFF_PACKET_JSON:"))
        payload = msg.split("\n", 1)[1]
        obj = json.loads(payload)
        self.assertEqual(obj["from"], "A")
        self.assertEqual(obj["to"], "B")
        self.assertIn("retrieved_context", obj["context"])
