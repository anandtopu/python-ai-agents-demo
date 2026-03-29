from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

from openai import OpenAI

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from context_engineering import ContextEngineer
from multi_agent import Agent, Orchestrator


@dataclass(frozen=True)
class DemoResult:
    demo: str
    outputs: dict[str, Any]


def run_demo(
    client: OpenAI,
    model: str,
    demo: str,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> DemoResult:
    demo = demo.strip().lower()

    if demo in {"research", "research_writer_critic", "rwc"}:
        return _demo_research_writer_critic(client, model, on_event=on_event)

    if demo in {"code_review", "review"}:
        return _demo_code_review(client, model, on_event=on_event)

    if demo in {"debate", "debate_consensus"}:
        return _demo_debate_consensus(client, model, on_event=on_event)

    if demo in {"context", "context_engineering", "cse"}:
        return _demo_context_engineering_cse(client, model, on_event=on_event)

    if demo in {"lc_structured", "langchain_structured", "structured"}:
        return _demo_langchain_structured(client, model)

    if demo in {"lg_hitl", "langgraph_hitl", "hitl"}:
        return _demo_langgraph_hitl(client, model)

    raise ValueError("Unknown demo. Try: research | code_review | debate | context | lc_structured | lg_hitl")


def _main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python multi_agent_demo.py <demo>  (research | code_review | debate | context | lc_structured | lg_hitl)"
        )

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


def _demo_research_writer_critic(client: OpenAI, model: str, on_event: Any = None) -> DemoResult:
    orch = Orchestrator(client, model, on_event=on_event)

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


def _demo_code_review(client: OpenAI, model: str, on_event: Any = None) -> DemoResult:
    orch = Orchestrator(client, model, on_event=on_event)

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


def _demo_debate_consensus(client: OpenAI, model: str, on_event: Any = None) -> DemoResult:
    orch = Orchestrator(client, model, on_event=on_event)

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


def _demo_context_engineering_cse(client: OpenAI, model: str, on_event: Any = None) -> DemoResult:
    ce = ContextEngineer()
    ce.add_user_goal(
        "Create a small CSE-style demo: debug an algorithm implementation, add tests, and write a short explanation. "
        "Prefer minimal, readable Python in a single module."
    )
    ce.add_decision("All code artifacts must be written to workspace_sandbox under 'cse_demo/'.", source="demo")
    ce.add_decision("Use only standard library for tests (unittest).", source="demo")

    orch = Orchestrator(client, model, context_engineer=ce, on_event=on_event)

    implementer = Agent(
        name="Implementer",
        system_prompt=(
            "You are a software engineer. Implement the requested code. "
            "When asked to write files, use sandbox_write into workspace_sandbox. "
            "Keep the code small and clear."
        ),
    )
    tester = Agent(
        name="Tester",
        system_prompt=(
            "You write focused tests. Use unittest. "
            "When asked to write files, use sandbox_write into workspace_sandbox."
        ),
    )
    explainer = Agent(
        name="Explainer",
        system_prompt=(
            "You explain the final solution concisely: what was wrong, what changed, complexity, and how to run tests."
        ),
    )

    buggy_spec = (
        "Write a Python module 'cse_demo/intervals.py' that merges overlapping intervals. "
        "Intentionally include a subtle bug around adjacency (e.g., [1,2] and [3,4]) and sorting edge cases. "
        "Also include a small main block printing a sample merge."
    )

    i_thread = orch.start_thread(implementer)
    i1 = orch.ask(i_thread, buggy_spec)

    ce.add_artifact(
        "Created initial buggy implementation at cse_demo/intervals.py (expected to have adjacency/sorting issues).",
        path="cse_demo/intervals.py",
        source="implementer",
    )

    handoff_to_tester = ce.make_handoff(
        from_agent="Implementer",
        to_agent="Tester",
        goal="Write tests that expose the bug(s) in the interval merging implementation.",
        retrieve_query="interval merge adjacency sorting unittest cse_demo/intervals.py",
        extra_context={"module_path": "cse_demo/intervals.py"},
    )

    t_thread = orch.start_thread(tester)
    orch.add_handoff(t_thread, handoff_to_tester.to_user_message())
    t1 = orch.ask(
        t_thread,
        "Using the handoff packet, write a test file 'cse_demo/test_intervals.py' that fails on the buggy behavior. "
        "Then write it to the sandbox.",
    )

    ce.add_artifact(
        "Added tests at cse_demo/test_intervals.py to expose merge edge cases.",
        path="cse_demo/test_intervals.py",
        source="tester",
    )

    handoff_back_to_implementer = ce.make_handoff(
        from_agent="Tester",
        to_agent="Implementer",
        goal="Fix the interval merge implementation so tests pass, without changing the public API.",
        retrieve_query="failing tests interval merge adjacency fix",
        extra_context={"test_path": "cse_demo/test_intervals.py", "module_path": "cse_demo/intervals.py"},
    )

    orch.add_handoff(i_thread, handoff_back_to_implementer.to_user_message())
    i2 = orch.ask(
        i_thread,
        "Fix 'cse_demo/intervals.py' so it correctly merges overlapping intervals and (by decision) treats adjacent intervals as merged. "
        "Ensure it sorts safely and handles empty input. Write the full corrected file to the sandbox.",
    )

    ce.add_decision("Adjacency is merged: treat end >= next_start - 1 as continuous only if intervals are integer endpoints.", source="implementer")

    handoff_to_explainer = ce.make_handoff(
        from_agent="Implementer",
        to_agent="Explainer",
        goal="Explain the final behavior and how context engineering helped coordinate multi-agent work.",
        retrieve_query="goals decisions artifacts intervals tests handoff",
        extra_context={"artifacts": ["cse_demo/intervals.py", "cse_demo/test_intervals.py"]},
    )

    e_thread = orch.start_thread(explainer)
    orch.add_handoff(e_thread, handoff_to_explainer.to_user_message())
    e1 = orch.ask(
        e_thread,
        "Using the handoff packet, explain: bug(s), fix, complexity, and how to run tests with run_shell.",
    )

    return DemoResult(
        demo="context_engineering_cse",
        outputs={
            "implementer_initial": i1.assistant_text,
            "tester": t1.assistant_text,
            "implementer_fix": i2.assistant_text,
            "explainer": e1.assistant_text,
            "shared_context_snapshot": ce.store.as_compact_text(max_items=50),
        },
    )


class _BugReport(BaseModel):
    title: str = Field(..., description="Short bug title")
    severity: str = Field(..., description="one of: low, medium, high")
    suspected_root_cause: str = Field(..., description="Hypothesis about what caused it")
    repro_steps: list[str] = Field(..., description="Steps to reproduce")
    expected: str = Field(..., description="Expected behavior")
    actual: str = Field(..., description="Actual behavior")
    fix_plan: list[str] = Field(..., description="Concrete steps to fix")


def _demo_langchain_structured(_client: OpenAI, model: str) -> DemoResult:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
    structured = llm.with_structured_output(_BugReport)

    prompt = (
        "You are doing a code review triage. Produce a structured bug report.\n\n"
        "Scenario: A function merge_intervals(intervals) sometimes returns unsorted output and fails to merge adjacent "
        "intervals like [1,2] and [3,4] when the team expects adjacency to merge. "
        "Assume integer endpoints and that intervals should be merged if they overlap OR are adjacent."
    )

    report = structured.invoke(prompt)
    return DemoResult(demo="lc_structured", outputs={"bug_report": report.model_dump()})


def _demo_langgraph_hitl(_client: OpenAI, model: str) -> DemoResult:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    State = dict[str, Any]

    def plan_node(_state: State) -> dict[str, Any]:
        llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
        plan = (
            llm.invoke("Propose a 3-step plan to fix a failing unit test in a small Python project. Keep it short.")
            .content
        )
        return {"plan": plan, "status": "pending"}

    def approval_node(state: State) -> Command[str]:
        decision = interrupt({"question": "Approve this plan?", "plan": state.get("plan", "")})
        return Command(goto="execute" if decision else "cancel")

    def execute_node(_state: State) -> dict[str, Any]:
        return {"status": "approved_and_executed"}

    def cancel_node(_state: State) -> dict[str, Any]:
        return {"status": "rejected"}

    builder = StateGraph(State)
    builder.add_node("plan", plan_node)
    builder.add_node("approval", approval_node)
    builder.add_node("execute", execute_node)
    builder.add_node("cancel", cancel_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "approval")
    builder.add_edge("execute", END)
    builder.add_edge("cancel", END)

    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "demo-hitl-1"}}

    first = graph.invoke({} , config=config)
    interrupts = first.get("__interrupt__", [])

    resumed = graph.invoke(Command(resume=True), config=config)

    return DemoResult(
        demo="lg_hitl",
        outputs={
            "interrupt_payload": [getattr(i, "value", None) for i in interrupts],
            "result": dict(resumed),
        },
    )


if __name__ == "__main__":
    _main()
