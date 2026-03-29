from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from multi_agent import Agent, Orchestrator


@dataclass(frozen=True)
class DemoResult:
    demo: str
    outputs: dict[str, Any]


def run_demo(client: OpenAI, model: str, demo: str) -> DemoResult:
    demo = demo.strip().lower()

    if demo in {"research", "research_writer_critic", "rwc"}:
        return _demo_research_writer_critic(client, model)

    if demo in {"code_review", "review"}:
        return _demo_code_review(client, model)

    if demo in {"debate", "debate_consensus"}:
        return _demo_debate_consensus(client, model)

    raise ValueError("Unknown demo. Try: research | code_review | debate")


def _main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python multi_agent_demo.py <demo>  (research | code_review | debate)")

    demo = sys.argv[1]
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    client = OpenAI(api_key=api_key, base_url=base_url)
    result = run_demo(client, model, demo)

    print(f"Demo complete: {result.demo}")
    for k, v in result.outputs.items():
        print(f"\n--- {k.upper()} ---\n{v}\n")


def _demo_research_writer_critic(client: OpenAI, model: str) -> DemoResult:
    orch = Orchestrator(client, model)

    researcher = Agent(
        name="Researcher",
        system_prompt=(
            "You gather facts from the web using web_get_text. "
            "Output: a short bullet list of verified points with citations as URLs."
        ),
    )
    writer = Agent(
        name="Writer",
        system_prompt=(
            "You write a short structured explanation based only on the provided research bullets. "
            "If asked to write to a file, use sandbox_write into workspace_sandbox."
        ),
    )
    critic = Agent(
        name="Critic",
        system_prompt=(
            "You review the draft for clarity, correctness, and missing pieces. "
            "Output: actionable feedback and a revised outline."
        ),
    )

    topic = "Artificial Intelligence (AI): definition, key subfields, and major concerns"

    r_thread = orch.start_thread(researcher)
    r = orch.ask(
        r_thread,
        "Research this topic using web_get_text and return 6-10 bullets with URLs: " + topic,
    )

    w_thread = orch.start_thread(writer)
    w = orch.ask(
        w_thread,
        "Using only these research bullets, write a 250-400 word explainer with headings.\n\n"
        f"RESEARCH BULLETS:\n{r.assistant_text}",
    )

    c_thread = orch.start_thread(critic)
    c = orch.ask(
        c_thread,
        "Critique this draft and suggest improvements.\n\nDRAFT:\n" + w.assistant_text,
    )

    w2 = orch.ask(
        w_thread,
        "Revise the explainer using the critic feedback. Then save final to sandbox_write path 'multi_agent/research_ai.md'.\n\n"
        f"CRITIC FEEDBACK:\n{c.assistant_text}",
    )

    return DemoResult(
        demo="research_writer_critic",
        outputs={
            "research": r.assistant_text,
            "draft": w.assistant_text,
            "critic": c.assistant_text,
            "final": w2.assistant_text,
        },
    )


def _demo_code_review(client: OpenAI, model: str) -> DemoResult:
    orch = Orchestrator(client, model)

    author = Agent(
        name="Author",
        system_prompt=(
            "You are a developer. You can create files in workspace_sandbox using sandbox_write. "
            "Prefer small, readable functions."
        ),
    )
    reviewer = Agent(
        name="Reviewer",
        system_prompt=(
            "You do code review. Identify bugs, edge cases, style issues, and suggest specific fixes. "
            "Be concrete."
        ),
    )
    fixer = Agent(
        name="Fixer",
        system_prompt=(
            "You apply the review suggestions by editing files in workspace_sandbox using sandbox_write. "
            "When rewriting, output the full corrected file content."
        ),
    )

    a_thread = orch.start_thread(author)
    a = orch.ask(
        a_thread,
        "Create a Python file 'multi_agent/bad_math.py' in the sandbox that implements a function mean(nums) but has 2 subtle bugs. "
        "Also add a small main block that prints mean([1,2,3]).",
    )

    r_thread = orch.start_thread(reviewer)
    r = orch.ask(
        r_thread,
        "Review the code produced below. Point out the bugs and propose a fixed version.\n\nCODE:\n" + a.assistant_text,
    )

    f_thread = orch.start_thread(fixer)
    f = orch.ask(
        f_thread,
        "Apply the reviewer suggestions. Write the corrected file to sandbox_write path 'multi_agent/bad_math.py'.\n\n"
        f"REVIEW:\n{r.assistant_text}",
    )

    return DemoResult(
        demo="code_review",
        outputs={"author": a.assistant_text, "review": r.assistant_text, "fix": f.assistant_text},
    )


def _demo_debate_consensus(client: OpenAI, model: str) -> DemoResult:
    orch = Orchestrator(client, model)

    pro = Agent(
        name="Pro",
        system_prompt="Argue in favor. Use clear premises and be concise.",
    )
    con = Agent(
        name="Con",
        system_prompt="Argue against. Use clear premises and be concise.",
    )
    judge = Agent(
        name="Judge",
        system_prompt=(
            "Synthesize both sides into a balanced conclusion and a decision rubric. "
            "Output: pros, cons, and a practical recommendation."
        ),
    )

    motion = "Should teams adopt multi-agent LLM systems for internal automation this quarter?"

    p_thread = orch.start_thread(pro)
    p = orch.ask(p_thread, "Debate motion: " + motion)

    c_thread = orch.start_thread(con)
    c = orch.ask(c_thread, "Debate motion: " + motion)

    j_thread = orch.start_thread(judge)
    j = orch.ask(
        j_thread,
        "Given the arguments below, produce a balanced conclusion and a rubric for deciding.\n\n"
        f"PRO:\n{p.assistant_text}\n\nCON:\n{c.assistant_text}",
    )

    return DemoResult(
        demo="debate_consensus",
        outputs={"pro": p.assistant_text, "con": c.assistant_text, "judge": j.assistant_text},
    )


if __name__ == "__main__":
    _main()
