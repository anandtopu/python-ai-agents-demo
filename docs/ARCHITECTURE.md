# Architecture

## Overview

This repository is a teaching/demo project that explores multi-agent patterns and “context engineering” with a small Python CLI agent, multi-agent orchestration demos, and a web UI.

Key goals:

- Provide runnable demos that illustrate agent orchestration patterns.
- Demonstrate context engineering primitives (shared context store, retrieval, and handoffs).
- Demonstrate observability and evaluation patterns.

## High-level components

### CLI agent

- `agent_cli.py` is an interactive CLI loop that:
  - keeps a message history on disk
  - supports sessions
  - supports tool calling via OpenAI Chat Completions
  - can execute multi-agent demos via `/ma <demo>`

### Multi-agent runtime

- `multi_agent.py` contains:
  - `Agent` definitions (system prompt + name)
  - `ToolLoopRunner` that runs the LLM tool-calling loop
  - `Orchestrator` that manages “threads” (per-agent message lists)

The orchestrator supports an `on_event` hook used by observability/UI.

### Demos

- `multi_agent_demo.py` hosts a set of runnable demos.

### Context engineering

- `context_engineering.py` implements:
  - `ContextStore` (shared working context)
  - lightweight retrieval
  - `HandoffPacket` for explicit agent-to-agent handoffs

### Observability

- `observability.py` implements a lightweight tracer:
  - enriches events with `run_id`, timestamps, and sequence numbers
  - supports in-memory + JSONL file sinks

### Evaluation

- `agentic_eval.py` demonstrates a judge-based “agentic evaluation” pattern using structured outputs.

### Web app

- Backend: `backend/main.py` (FastAPI) exposes endpoints to run demos and fetch traces.
- Frontend: `frontend/` (Vite/React) visualizes the event timeline and outputs.

## Data flow

### Tool-calling loop

1. User (or demo) appends a user message to a thread.
2. `ToolLoopRunner` calls Chat Completions with tool schemas.
3. If tool calls are returned:
   - tools are executed locally via `tools.py`
   - results are appended as tool messages
   - the loop continues
4. If the assistant returns normal content, the turn ends.

### Observability

- The `on_event` callback records:
  - user messages
  - LLM request/response timings + usage when available
  - tool call + tool result + tool timings
- `backend/main.py` persists the trace to `traces/<run_id>.jsonl`.

## Notes

This is a demo codebase. In production, you would typically:

- use durable persistence for traces/checkpoints
- add authentication to the web API
- enforce stricter sandboxing for tools
- pin dependency versions and use lockfiles
