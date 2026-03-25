# DataOps Environment API

A local environment for simulating data cleaning and pipeline optimization tasks with a structured API.

## What this project provides

- A FastAPI backend with task lifecycle endpoints (`/reset`, `/step`, `/state`, `/grader`, `/baseline`).
- A deterministic baseline runner for automated task execution.
- OpenEnv-compatible metadata and endpoint mapping (`openenv.yaml`).
- A lightweight Vite + React shell for local app hosting.

## Repository layout

- `app.py` — FastAPI service entrypoint and route handlers.
- `open_env.py` — Environment interfaces, task mechanics, scoring, and baseline agent logic.
- `run_baseline.py` — Standalone script to run baseline evaluation against a running service.
- `openenv.yaml` — Environment definition and endpoint mapping.
- `health_check.md` — Manual verification checklist.
- `src/` — Frontend bootstrapping files.

## Requirements

### Backend
- Python 3.11+
- Dependencies in `requirements.txt`

### Frontend (optional)
- Node.js 20+
- npm

## Quick start

### 1) Install backend dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Start the API service

```bash
python app.py
```

The API listens on `http://0.0.0.0:3000`.

### 3) Run health checks

```bash
curl -X GET http://localhost:3000/health
curl -X GET http://localhost:3000/tasks
```

### 4) Run the baseline evaluator

```bash
python run_baseline.py
```

## API reference

### `GET /health`
Returns service status.

### `GET /tasks`
Returns:
- available tasks (`easy`, `medium`, `hard`)
- JSON schema for valid actions

### `POST /reset?difficulty=<easy|medium|hard>`
Creates a fresh environment state for the selected difficulty.

### `POST /step`
Applies one action and returns:
- updated observation
- reward payload (`score`, `delta`, `done`, `breakdown`)

### `GET /state`
Returns the current observation without mutating state.

### `POST /grader`
Computes and returns the current total score in `[0, 1]`.

### `POST /baseline`
Executes the internal baseline policy over all task difficulties.

## Example `step` request

```bash
curl -X POST http://localhost:3000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "fill_missing",
    "parameters": { "column": "region", "value": "unknown" }
  }'
```

## Frontend development

Install and run:

```bash
npm install
npm run dev
```

The development server runs on `http://localhost:3000` by default.

## Validation

If the OpenEnv CLI is installed:

```bash
openenv validate openenv.yaml
```

## Troubleshooting

- **Port already in use**: stop the conflicting process or change the service port.
- **Import errors**: confirm the virtual environment is active and dependencies are installed.
- **No baseline output**: ensure the API is running before executing `run_baseline.py`.
