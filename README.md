# Python AI Agents Demo

This is a teaching/demo repository for building **multi-agent** and **agentic workflows** in Python.

It includes:

- A minimal **CLI chat agent** with tool calling
- A set of **multi-agent demos** (handoffs, debate, reviewer/fixer loops, complex workflows)
- **Context engineering** primitives (shared context store, retrieval, handoff packets)
- **Agentic evaluation** (judge-style scoring with structured output)
- **Observability/tracing** (event timeline, persisted JSONL traces)
- A **web UI** (FastAPI backend + React/Vite frontend) to visualize multi-agent runs

## Requirements

- Python 3.10+ (3.11+ recommended)
- Node.js 18+ (for the web UI)

## Technology stack

### Python / LLM

- **OpenAI Python SDK** (`openai`)
  - OpenAI-compatible Chat Completions (supports `OPENAI_BASE_URL`)
- **LangChain** (`langchain`, `langchain-openai`)
  - Structured outputs demo (`lc_structured`)
  - Optional LangSmith tracing demo (`langsmith`)
- **LangGraph** (`langgraph`)
  - Human-in-the-loop interrupts demo (`lg_hitl`)
- **LangSmith** (`langsmith`)
  - Production-style tracing enablement via env vars

### Orchestration / tooling

- **Custom tool-calling loop** (`multi_agent.py`) with local tools (`tools.py`)
- **Context engineering** (`context_engineering.py`)
- **Observability** (`observability.py`) + backend trace persistence
- **Vector DB / RAG** (`vectordb.py`)
  - FAISS (`faiss-cpu`) via `langchain-community`

### Web app

- **FastAPI** backend (`backend/main.py`)
- **React + Vite + TypeScript** frontend (`frontend/`)

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Configure environment variables:

- Copy `.env.example` to `.env`
- Set `OPENAI_API_KEY`
- Optionally set:
  - `OPENAI_MODEL` (default: `gpt-4o-mini`)
  - `OPENAI_BASE_URL` (for OpenAI-compatible providers)

Example `.env`:

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=https://your-openai-compatible-host/v1
```

## Run the CLI agent

```powershell
python agent_cli.py
```

### CLI commands

- `/reset`
  - Clears the current chat history.
- `/save <name>`
  - Saves current history to `sessions/<name>.json`.
- `/load <name>`
  - Loads `sessions/<name>.json` as the active history.
- `/sessions`
  - Lists saved sessions.
- `/delete <name>`
  - Deletes `sessions/<name>.json`.
- `/summary`
  - Compresses older history into a compact memory message while keeping recent turns.
- `/ma <demo>`
  - Runs a multi-agent demo: `research`, `code_review`, `debate`, `context`, `lc_structured`, `lg_hitl`, `complex`, `context_limits`, or `langsmith`.

## Tools available to the agent

These are implemented in `tools.py` and are exposed to the model via tool-calling.

### Web

- `http_get(url, timeout_s=10)`
  - Returns status + a short preview.
- `web_get_text(url, timeout_s=15, max_chars=12000)`
  - Fetches a page and extracts readable text for summarization.

### Sandboxed file tools (safe)

All file operations are restricted to `./workspace_sandbox`.

- `sandbox_list(path=".", recursive=False, max_entries=200)`
- `sandbox_read(path, max_chars=12000)`
- `sandbox_write(path, content, append=False, create_dirs=True)`

### Project search

- `project_search(query, root=".", max_results=50, case_sensitive=False)`
  - Finds matching lines with line numbers.

### Restricted shell tool

- `run_shell(command, args=[], timeout_s=10, max_output_chars=8000)`
  - Allowlist only: `python`, `pip`, `git` (and `.exe` variants).

## Multi-agent demos

Multi-agent logic lives in:

- `multi_agent.py` (Agent + Orchestrator + tool-loop runner)
- `multi_agent_demo.py` (practice demos)

Current demos:

- `research` (researcher → writer → critic)
- `code_review` (author → reviewer → fixer)
- `debate` (pro → con → judge)
- `context` / `cse` (context engineering handoffs)
- `lc_structured` (LangChain structured output)
- `lg_hitl` (LangGraph interrupts / HITL)
- `complex` (planner → implement → tests → verify/repair loop → critic) + optional VectorDB retrieval
- `context_limits` (context poisoning/distraction/confusion/clash + mitigation)
- `langsmith` (LangSmith production-style tracing enablement)

### Run from the CLI

Inside `python agent_cli.py`:

```text
/ma research
/ma code_review
/ma debate
/ma context
/ma lc_structured
/ma lg_hitl
/ma complex
/ma context_limits
/ma langsmith
```

### Run directly

```powershell
python multi_agent_demo.py research
python multi_agent_demo.py code_review
python multi_agent_demo.py debate
python multi_agent_demo.py context
python multi_agent_demo.py lc_structured
python multi_agent_demo.py lg_hitl
python multi_agent_demo.py complex
python multi_agent_demo.py context_limits
python multi_agent_demo.py langsmith
```

## Project structure

```text
.
  agent_cli.py
  tools.py
  multi_agent.py
  multi_agent_demo.py
  context_engineering.py
  vectordb.py
  agentic_eval.py
  observability.py
  backend/
    main.py
    README.md
  frontend/
    src/
    README.md
  docs/
    ARCHITECTURE.md
    RUNBOOK.md
  scripts/
    run_cli.ps1
    run_backend.ps1
  tests (unittest)
    test_*.py
```

## LangSmith (production tracing demo)

There is a demo (`/ma langsmith`) that shows how production projects typically enable LangSmith tracing using environment variables.

To record traces in LangSmith, set:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=python-ai-agents-demo
```

Then run:

```text
/ma langsmith
```

If these variables are not set, the demo will print setup instructions instead of failing.

## Observability and traces

- The backend writes JSONL traces to `traces/<run_id>.jsonl`.
- The web UI can load past runs via `GET /api/runs`.

## Tests

Run:

```powershell
python -m unittest -q
```

## Docs

- `docs/ARCHITECTURE.md`
- `docs/RUNBOOK.md`

Artifacts written by demos will appear under:

- `workspace_sandbox/multi_agent/...`

## Web UI (FastAPI + React)

This repo includes a small web app that:

- Runs multi-agent demos on the backend
- Shows a timeline of outputs
- Optionally runs an agentic evaluation (judge rubric) and displays the score

### Start the backend

1. Install Python deps:

```powershell
pip install -r requirements.txt
```

2. Run the API server:

```powershell
uvicorn backend.main:app --reload --port 8000
```

Backend endpoints:

- `GET http://127.0.0.1:8000/api/demos`
- `POST http://127.0.0.1:8000/api/run`

### Start the frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Then open:

- `http://127.0.0.1:5173`

## Files created at runtime

These are intentionally gitignored:

- `.env` (secrets)
- `agent_history.json` (local chat history)
- `sessions/` (saved sessions)
- `workspace_sandbox/` (sandboxed artifacts)

## Security notes

- Do not commit secrets. Keep your API key only in `.env`.
- If you ever paste a key into a tracked file or terminal log, rotate it in your provider dashboard.
