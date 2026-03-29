from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from agentic_eval import evaluate_text
from multi_agent_demo import run_demo
from observability import InMemorySink, JsonlFileSink, Tracer, read_jsonl


class RunRequest(BaseModel):
    demo: str = Field(..., description="Demo name, e.g. 'context', 'lc_structured', 'lg_hitl'")
    evaluate: bool = Field(default=True)


class EvalRequest(BaseModel):
    demo: str = Field(..., description="Demo name")
    trials: int = Field(default=3, ge=1, le=20)


app = FastAPI(title="Python AI Agents Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/demos")
def list_demos() -> dict[str, Any]:
    return {
        "demos": [
            "research",
            "code_review",
            "debate",
            "context",
            "lc_structured",
            "lg_hitl",
            "complex",
            "context_limits",
        ]
    }


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    root = Path("traces")
    root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for p in root.glob("*.jsonl"):
        try:
            stat = p.stat()
        except OSError:
            continue
        runs.append({"run_id": p.stem, "path": p.as_posix(), "mtime": stat.st_mtime})
    runs.sort(key=lambda x: x["mtime"], reverse=True)
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    path = Tracer.default_trace_path(run_id)
    return {"run_id": run_id, "events": read_jsonl(path)}


@app.post("/api/run")
def run_demo_api(req: RunRequest) -> dict[str, Any]:
    load_dotenv(override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing OPENAI_API_KEY")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    client = OpenAI(api_key=api_key, base_url=base_url)

    events: list[dict[str, Any]] = []
    tracer = Tracer(metadata={"demo": req.demo})
    trace_path = Tracer.default_trace_path(tracer.run_id)
    tracer = Tracer(
        run_id=tracer.run_id,
        metadata={"demo": req.demo},
        sinks=[InMemorySink(events), JsonlFileSink(trace_path)],
    )
    tracer.emit({"type": "run_start", "demo": req.demo})

    try:
        result = run_demo(client, model, req.demo, on_event=tracer.emit)
    except Exception as e:
        tracer.emit({"type": "run_error", "error": str(e)})
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        tracer.emit({"type": "run_end", "demo": req.demo})

    evaluation: dict[str, Any] | None = None
    if req.evaluate:
        text = ""
        if isinstance(result.outputs.get("explainer"), str):
            text = str(result.outputs.get("explainer"))
        elif isinstance(result.outputs.get("final"), str):
            text = str(result.outputs.get("final"))
        else:
            # Fall back to a compact stringified view.
            text = str(result.outputs)

        evaluation = evaluate_text(
            rubric="Clarity, correctness, and actionability. Penalize missing steps or vague claims.",
            task=f"Run demo '{result.demo}' and present its outputs.",
            text=text,
            model=model,
        ).raw

    return {
        "run_id": tracer.run_id,
        "trace_path": trace_path.as_posix(),
        "demo": result.demo,
        "events": events,
        "outputs": result.outputs,
        "evaluation": evaluation,
    }


@app.post("/api/eval")
def eval_demo_api(req: EvalRequest) -> dict[str, Any]:
    load_dotenv(override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing OPENAI_API_KEY")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)

    scores: list[int] = []
    passes: int = 0
    last_outputs: dict[str, Any] | None = None

    for _ in range(req.trials):
        try:
            result = run_demo(client, model, req.demo)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        last_outputs = result.outputs
        text = ""
        if isinstance(result.outputs.get("explainer"), str):
            text = str(result.outputs.get("explainer"))
        elif isinstance(result.outputs.get("final"), str):
            text = str(result.outputs.get("final"))
        else:
            text = str(result.outputs)

        ev = evaluate_text(
            rubric="Clarity, correctness, and actionability. Penalize missing steps or vague claims.",
            task=f"Run demo '{result.demo}' and present its outputs.",
            text=text,
            model=model,
        ).score

        scores.append(int(ev.overall_score))
        if ev.pass_fail:
            passes += 1

    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "demo": req.demo,
        "trials": req.trials,
        "avg_score": avg,
        "scores": scores,
        "pass_rate": passes / req.trials,
        "last_outputs": last_outputs,
    }
