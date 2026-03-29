# First Agent (Python)

A minimal Python CLI agent that uses an OpenAI(-compatible) Chat Completions API with:

- Tool calling (web fetch, sandboxed files, project search, restricted shell)
- Persistent chat history (`agent_history.json`)
- Saved sessions (`sessions/*.json`)
- History summarization (`/summary`)
- Multi-agent orchestration demos (`/ma <demo>`)

## Requirements

- Python 3.10+ (3.11+ recommended)

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
  - Runs a multi-agent demo: `research`, `code_review`, or `debate`.

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

### Run from the CLI

Inside `python agent_cli.py`:

```text
/ma research
/ma code_review
/ma debate
```

### Run directly

```powershell
python multi_agent_demo.py research
python multi_agent_demo.py code_review
python multi_agent_demo.py debate
```

Artifacts written by demos will appear under:

- `workspace_sandbox/multi_agent/...`

## Files created at runtime

These are intentionally gitignored:

- `.env` (secrets)
- `agent_history.json` (local chat history)
- `sessions/` (saved sessions)
- `workspace_sandbox/` (sandboxed artifacts)

## Security notes

- Do not commit secrets. Keep your API key only in `.env`.
- If you ever paste a key into a tracked file or terminal log, rotate it in your provider dashboard.
