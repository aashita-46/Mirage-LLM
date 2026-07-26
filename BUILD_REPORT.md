# Mirage build report

## Delivered

A full-stack recruiter-facing Mirage application with a React/Vite frontend and a
FastAPI-compatible Python backend. It includes API-backed analysis, token uncertainty,
meaning-level sample clusters, experimental MirageScore, model self-verification,
adversarial prompt stress testing, and a curated benchmark that computes AUROC, ECE,
Brier score, calibration bins, and per-record outcomes at run time.

The default mode is clearly labelled **Cached demonstration**. Values are generated
deterministically from repository-defined records and calculated by the backend; they
are never presented as live model inference or external benchmark performance.

## Important files

- `src/App.tsx` — complete product experience and interaction flows
- `src/components/` — gauge, token heatmap, and coordinated analysis view
- `src/styles.css` — responsive research-lab visual system
- `api/index.py` — API, uncertainty mathematics, stress engine, and benchmark
- `tests/test_api.py` — backend mathematical and API tests
- `README.md` — architecture, methodology, configuration, run modes, and limitations
- `vercel.json` — Vite plus Python serverless deployment
- `Dockerfile`, `api.Dockerfile`, `docker-compose.yml` — container workflow

## Run locally

```text
npm install
python -m pip install -r requirements.txt uvicorn pytest httpx
python -m uvicorn api.index:app --reload --port 8000
npm run dev
```

Open `http://localhost:5173`.

## Real local model and Groq

The provider contract and environment variables are documented in `.env.example` and
`README.md`. Install PyTorch/Transformers/NLI dependencies separately before enabling
a local provider; model weights are never downloaded at app startup. For Groq, set
`GROQ_API_KEY` only in a server-side secret store and configure `GROQ_MODEL`. The
deployed demonstration does not require or expose a key.

## Verification completed

- `python -m pytest -q` — **9 passed**
- `npm test` — **2 passed**
- `npm run build` — **passed**, 1,937 modules transformed
- `npm audit --audit-level=high` — **0 vulnerabilities**
- API health smoke test — HTTP success, `cached_demo`
- Browser analysis flow — 14 token cells, 6 samples, no console errors
- Browser stress flow — 5 variant results
- Browser benchmark flow — 4 computed metrics, 12 result records
- Mobile check — navigation breakpoint active; no horizontal overflow

Docker files were authored, but the Docker Desktop daemon was not running on this
machine, so container execution was not claimed as verified.

## Known limitations / deferred

- The hosted deployment uses deterministic cached analysis, not live Hugging Face
  inference. The current machine has no system Python or detected NVIDIA runtime.
- Bidirectional neural NLI, Hugging Face logits, persistent SQLite leaderboard,
  true SSE token streaming, provider-side Groq calls, external dataset adapters,
  pause/resume benchmark jobs, and learned calibration remain production extensions.
- The curated benchmark verifies metric plumbing and UI behavior; it is not a model
  performance claim.

## Exact verification commands

```text
npm install
npm run build
npm test
npm audit --audit-level=high
<bundled-python> -m pip install -r requirements.txt uvicorn pytest httpx
<bundled-python> -m pytest -q
<bundled-python> -m uvicorn api.index:app --host 127.0.0.1 --port 8000
npm run dev -- --host 127.0.0.1
```

## Production deployment

**https://mirage-llm.vercel.app**

Vercel deployment `dpl_Ae3GzsR3qYx6oXhfD24paS7dfUsg` reports **Ready**.
Production smoke tests confirmed:

- Landing page: HTTP 200 with the expected Mirage title
- `GET /api/v1/health`: HTTP 200, `cached_demo`
- `POST /api/v1/analyse`: 14 token records and 6 semantic samples returned

### Mobile optimisation verification

The production interface was hardened for 320–390 px phone widths with safe-area
insets, 44–50 px touch targets, an accessible collapsible navigation menu, stacked
analysis controls, wrapped metadata, mobile benchmark result cards, and narrow-screen
metric layouts. Browser checks confirmed no horizontal document overflow at either
breakpoint, menu close-on-navigation, working analysis/benchmark flows, and zero
rendered error states.
