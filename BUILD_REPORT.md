# Mirage research-platform build report

## Outcome

Mirage is now a local-first LLM reliability evaluation platform. It separates
uncertainty signals from labelled correctness, preserves raw cached provider outputs,
computes calibration and selective-prediction metrics, persists versioned experiments
in SQLite, and exposes the results through a responsive research interface.

The implementation was completed and verified locally before the later authorised
GitHub publication and Vercel production deployment.

## Architecture changes

- `api/core.py` contains versioned Pydantic contracts, provider capabilities,
  deterministic scoring, signal calculation, metrics, failure categorisation,
  persistence, overrides, and export.
- `api/index.py` is the HTTP boundary with structured errors and routes for datasets,
  models, playground analysis, experiments, comparisons, overrides, and reports.
- `data/datasets/mirage-starter.json` is an 18-example, multi-domain demonstration
  dataset. Cached responses are explicitly labelled and are not a live benchmark.
- `scripts/evaluate.py` runs the same pipeline without the browser.
- The React application is organised into Overview, Playground, Experiments,
  Datasets, Compare, Calibration, Failures, Reports, Methodology, and Settings views.

## Features completed

- JSON and JSONL dataset validation, preview, filtering, upload, and version metadata
- deterministic experiment IDs, schema version 2.0, timestamps, and model metadata
- provider capability flags; unsupported log probabilities remain unavailable
- acceptable-answer matching, normalised exact match, token F1, numeric tolerance,
  date normalisation, and explicit unanswerable evaluation
- configurable multi-response grouping with an honestly labelled lexical fallback
- semantic entropy, response consistency, self-verification uncertainty, and an
  experimental weighted Mirage Risk Score with contribution traces
- AUROC, AUPRC, ECE, Brier score, NLL, reliability bins, risk-coverage, and
  signal-comparison calculations with validity and small-sample warnings
- SQLite migrations and storage for raw outputs, aggregate metrics, errors, and
  human overrides without overwriting automated labels
- partial-failure-aware experiment runs and per-example error states
- failure taxonomy and slices for confidently wrong and uncertain-correct examples
- JSON, CSV, Markdown, and HTML exports
- desktop and mobile research UI with capability-aware controls and meaningful states
- methodology and limitation documentation in the application and README

## Reproducible starter run

Research question: Which available uncertainty signal best separates correct and
incorrect factual answers on the included demonstration dataset?

The locally executed cached-provider run produced experiment
`exp_6bc242fcbb015c76` over 18 labelled examples:

- accuracy: 0.778
- combined-risk AUROC: 0.589
- combined-risk AUPRC: 0.339
- ECE: 0.434
- Brier score: 0.315

These are pipeline outputs from repository-labelled cached responses, not claims about
a live model or a definitive benchmark. Calibration is explicitly flagged as unstable
because the dataset contains fewer than 30 examples.

## Verification

- backend: `16 passed`
- frontend: `5 passed`
- TypeScript production build: passed
- lint: passed
- npm audit: zero vulnerabilities
- CLI starter experiment: completed, 18 examples, zero provider failures
- API health: schema 2.0, SQLite, local research mode
- desktop browser: experiment creation and result rendering passed
- phone browser: 390 × 844 viewport, no horizontal overflow
- signal controls, comparison navigation, deletion controls, and human overrides:
  browser-verified

## Run commands

```powershell
npm install
& "<bundled-python>" -m pip install -r requirements.txt
& "<bundled-python>" -m uvicorn api.index:app --host 127.0.0.1 --port 8000
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173`.

Run the CLI experiment with:

```powershell
& "<bundled-python>" scripts/evaluate.py --config config/starter-experiment.json
```

## Known limitations

- The included provider is cached demonstration data; it performs no live inference.
- Token uncertainty is disabled because cached outputs contain no provider-native
  token distributions. No values are estimated.
- Semantic clustering uses a lexical Jaccard fallback, not NLI or embeddings.
- Self-verification fields are cached provider metadata, not ground truth.
- Retrieval faithfulness, streaming, pause/cancel, learned calibration, and live
  OpenAI-compatible/Ollama providers remain extension points.
- The starter dataset is intentionally small; its metrics are useful for validating
  the pipeline, not drawing scientific conclusions.
- Docker configuration was updated but the container runtime was not available for
  execution on this machine.

## Recommended next experiments

1. Add an OpenAI-compatible or Ollama provider and capture native capability metadata.
2. Replace lexical clustering with a frozen embedding or bidirectional NLI evaluator.
3. Run at least 200 independently labelled examples with stratified domain slices.
4. Compare prompt, temperature, retrieval, and model variants under identical data.
5. Fit Platt or isotonic calibration on a training split and evaluate on a held-out
   split only.
6. Measure coverage at target accuracy and review-budget outcomes with confidence
   intervals.
