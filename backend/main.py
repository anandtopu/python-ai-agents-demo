from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from agentic_eval import evaluate_text
from multi_agent_demo import run_demo


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
        ]
    }


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

    try:
        result = run_demo(client, model, req.demo, on_event=lambda ev: events.append(ev))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

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
