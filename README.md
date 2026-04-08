---
title: Cloud Cost Optimizer
emoji: "☁️"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Cloud Cost Optimizer

Cloud Cost Optimizer is a real-world OpenEnv reinforcement learning environment where an agent manages a simulated AWS fleet and learns to reduce infrastructure spend without breaking service-level objectives.

The environment models a 20-instance fleet over 72 simulated hours. The agent can resize instances, schedule off-peak scale-down windows, convert eligible workloads to spot, reserve stable workloads, and restore interrupted spot capacity. This is meant to feel like an actual cloud FinOps and reliability task rather than a toy control problem.

## Why this aligns with the hackathon requirements

- Real-world task: cloud infrastructure cost optimization with SLA risk, workload timing, and pricing tradeoffs.
- OpenEnv spec: typed models live in `models.py`, the environment implements `reset()`, `step()`, and `state()` in `server/environment.py`, and `openenv.yaml` defines tasks, spaces, reward, and infrastructure metadata.
- Three tasks: easy, medium, and hard tasks are declared in `openenv.yaml` and backed by deterministic graders in `tasks/`.
- Dense reward: each step rewards hourly savings and penalizes SLA breaches and unsafe interruption handling.
- Baseline inference: `inference.py` runs all three tasks with reproducible seed `42` and emits `[START]`, `[STEP]`, and `[END]` logs.
- Docker deploy: the repo includes a root `Dockerfile` that serves the FastAPI/OpenEnv app on port `7860`, matching Hugging Face Spaces Docker expectations.

## Environment design

Episode length:
- 72 steps
- 1 step = 1 simulated hour

Fleet composition:
- 5 clearly overprovisioned instances for the easy task
- 5 schedule-sensitive instances with predictable peak windows for the medium task
- 4 spot-friendly candidates for the hard task
- 6 stable anchors that should usually be left alone

Workload patterns:
- `web_api` peaks around 2pm
- `data_pipeline` peaks between 2am and 6am
- `batch_job` is bursty and noisy
- `stateless` is safe for spot
- `database` should remain stable and never move to spot

## Action space

The action schema is defined in [`models.py`](./models.py) and documented in [`openenv.yaml`](./openenv.yaml).

- `resize(instance_id, new_type)`: change an instance type
- `schedule(instance_id, schedule_off, schedule_on)`: create an off-peak schedule
- `convert_spot(instance_id)`: move a workload to spot pricing
- `reserve(instance_id)`: move a workload to reserved pricing
- `restore(instance_id)`: restore an interrupted spot instance to on-demand
- `noop()`: do nothing for the current hour

## Observation space

Each observation includes:

- `hour`
- `hour_of_day`
- `instances`
- `total_cost_so_far`
- `sla_violations`
- `savings_vs_baseline`
- `scheduled_actions`

Each instance snapshot includes pricing model, workload type, utilization, interruption state, headroom, and accumulated SLA breaches.

## Reward function

The environment uses a dense reward with partial progress signals:

- positive reward for hourly savings versus the on-demand baseline
- negative reward for new SLA violations
- negative reward for unsafe or unhandled spot interruptions

This means agents get useful feedback every step instead of only at the end of the episode.

## Tasks and graders

Task 1, easy:
- Identify the 5 heavily overprovisioned instances (`i-001` to `i-005`)
- Downsize them without touching the stable anchors (`i-015` to `i-020`)

Task 2, medium:
- Learn workload timing patterns
- Use schedule actions to capture off-peak savings without causing SLA breaches

Task 3, hard:
- Blend on-demand, spot, and reserved pricing across the portfolio
- Restore interrupted spot instances promptly
- Balance efficiency with operational risk

Task grader entry points:
- [`tasks/task1_grader.py`](./tasks/task1_grader.py)
- [`tasks/task2_grader.py`](./tasks/task2_grader.py)
- [`tasks/task3_grader.py`](./tasks/task3_grader.py)

LLM-based rubric helpers also exist in [`rubrics.py`](./rubrics.py) for richer trajectory evaluation.

## API endpoints

- `GET /`: health check
- `GET /health`: secondary health endpoint
- `GET /web`: interactive browser UI for manual reset/step/state testing
- `GET /docs`: Swagger docs
- `POST /reset`: start a new episode
- `POST /step`: take an action
- `GET /state`: get current episode state
- `WS /ws`: WebSocket interface
- `GET /tools`: tool discovery for OpenEnv-style integrations

## Local setup

### 1. Create and activate the environment

```powershell
cd C:\Users\DELL\Downloads\cloud_cost_env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install uv
uv pip install fastapi "uvicorn[standard]" pydantic openai openenv-core httpx websockets gradio
```

### 2. Run the server locally

```powershell
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Then open:

- [http://localhost:7860/](http://localhost:7860/)
- [http://localhost:7860/web](http://localhost:7860/web)
- [http://localhost:7860/docs](http://localhost:7860/docs)

## Step 4: test `inference.py` locally

If you do not have an API key yet, the script now falls back to a deterministic heuristic so it still completes locally and prints the required structured logs.

Windows PowerShell:

```powershell
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:HF_TOKEN="your_openai_api_key"
.\.venv\Scripts\python.exe inference.py
```

Mac/Linux shell:

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_openai_api_key
python inference.py
```

Expected behavior:
- the script completes without crashing
- it prints `[START]`, `[STEP]`, and `[END]` logs for each task
- it prints a final `[SUMMARY]`

## Step 5: test Docker locally

Build:

```powershell
docker build -t cloud-cost-optimizer .
```

Run:

```powershell
docker run -p 7860:7860 cloud-cost-optimizer
```

Then test:

```powershell
curl http://localhost:7860/
curl http://localhost:7860/health
```

If those return a healthy JSON response, the container is in the same shape Hugging Face Spaces will use.

## Deploy to Hugging Face Spaces

1. Create a new Hugging Face Space.
2. Choose `Docker` as the Space SDK.
3. Upload this repository as-is, or push it from git.
4. In the Space settings, add secrets:
   - `API_BASE_URL=https://api.openai.com/v1`
   - `MODEL_NAME=gpt-4o-mini`
   - `HF_TOKEN=<your OpenAI-compatible API key>`
5. Wait for the Docker build to finish.
6. Open the Space URL and verify:
   - `/` returns status JSON
   - `/web` loads the interactive UI
   - `/docs` opens the API docs

## Notes for submission

- Replace the `author` field in [`openenv.yaml`](./openenv.yaml) with your actual Hugging Face username before final submission if you want it to reflect your account.
- If you want the strongest leaderboard result, add a real API key before running `inference.py`; otherwise the heuristic fallback is useful for validation but not necessarily for best scores.
