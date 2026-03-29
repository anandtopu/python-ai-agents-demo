# Runbook

## CLI

```powershell
python agent_cli.py
```

Common commands:

- `/ma <demo>`
- `/summary`
- `/save <name>` / `/load <name>`

## Tests

```powershell
python -m unittest -q
```

## Web app

### Backend

```powershell
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open:

- http://127.0.0.1:5173

## Traces

The backend writes JSONL traces to:

- `traces/<run_id>.jsonl`

You can list runs via:

- `GET /api/runs`

and fetch a run via:

- `GET /api/runs/{run_id}`
