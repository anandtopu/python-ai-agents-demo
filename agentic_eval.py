from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class EvalScore(BaseModel):
    rubric: str = Field(..., description="Name of the rubric")
    overall_score: int = Field(..., ge=0, le=10)
    strengths: list[str]
    weaknesses: list[str]
    suggested_improvements: list[str]
    pass_fail: bool


@dataclass(frozen=True)
class EvalResult:
    score: EvalScore
    raw: dict[str, Any]


def evaluate_text(*, rubric: str, task: str, text: str, model: str | None = None) -> EvalResult:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    llm = ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), api_key=api_key, base_url=base_url)
    judge = llm.with_structured_output(EvalScore)

    prompt = (
        "You are an evaluator. Score the assistant output using the given rubric. "
        "Be strict and consistent. Use the full 0-10 scale.\n\n"
        f"RUBRIC: {rubric}\n"
        f"TASK: {task}\n\n"
        "ASSISTANT OUTPUT:\n"
        f"{text}\n"
    )

    score = judge.invoke(prompt)
    return EvalResult(score=score, raw=score.model_dump())
