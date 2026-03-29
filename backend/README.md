# Backend (FastAPI)

## Endpoints

- `GET /api/demos`
- `POST /api/run`
- `POST /api/eval`
- `GET /api/runs`
- `GET /api/runs/{run_id}`

## Run

```powershell
python -m pip install -r ..\requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

## Notes

- Traces are persisted as JSONL under `../traces/`.
- CORS is currently wide-open for local demo use.
