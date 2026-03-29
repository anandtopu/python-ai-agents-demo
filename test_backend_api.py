from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError as e:  # pragma: no cover
    raise unittest.SkipTest("fastapi not installed; skipping backend API tests") from e

try:
    import openai as _openai  # noqa: F401
except ModuleNotFoundError as e:  # pragma: no cover
    raise unittest.SkipTest("openai not installed; skipping backend API tests") from e

import backend.main as api


class TestBackendAPI(unittest.TestCase):
    def test_list_demos(self) -> None:
        client = TestClient(api.app)
        r = client.get("/api/demos")
        self.assertEqual(r.status_code, 200)
        demos = r.json()["demos"]
        self.assertIn("context", demos)
        self.assertIn("complex", demos)

    def test_run_and_runs_endpoints_with_mock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            traces_dir = Path(td) / "traces"
            traces_dir.mkdir(parents=True, exist_ok=True)

            def _fake_trace_path(run_id: str) -> Path:
                return traces_dir / f"{run_id}.jsonl"

            class _FakeResult:
                def __init__(self) -> None:
                    self.demo = "fake"
                    self.outputs = {"x": "y"}

            def _fake_run_demo(_client: Any, _model: str, _demo: str, on_event=None):
                if on_event:
                    on_event({"type": "assistant_message", "agent": "X", "content": "hi"})
                return _FakeResult()

            with patch.object(api.Tracer, "default_trace_path", side_effect=_fake_trace_path):
                with patch.object(api, "run_demo", side_effect=_fake_run_demo):
                    with patch.dict("os.environ", {"OPENAI_API_KEY": "k", "OPENAI_MODEL": "m"}):
                        client = TestClient(api.app)
                        r = client.post("/api/run", json={"demo": "context", "evaluate": False})
                        self.assertEqual(r.status_code, 200)
                        body = r.json()
                        self.assertIn("run_id", body)
                        self.assertTrue((traces_dir / f"{body['run_id']}.jsonl").exists())

                        r2 = client.get("/api/runs")
                        self.assertEqual(r2.status_code, 200)
                        runs = r2.json()["runs"]
                        self.assertTrue(runs)

                        r3 = client.get(f"/api/runs/{body['run_id']}")
                        self.assertEqual(r3.status_code, 200)
                        self.assertEqual(r3.json()["run_id"], body["run_id"])
