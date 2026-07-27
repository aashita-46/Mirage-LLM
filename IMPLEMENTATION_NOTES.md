# Mirage research-platform implementation checklist

## Audit findings (2026-07-27)

- The current FastAPI backend is a single module with no configuration, storage, or
  provider boundaries.
- `tokenise()` invents log-probabilities and entropy for demo answers. These values
  cannot be presented as model uncertainty and must be removed.
- Benchmark correctness and risk values are hard-coded tuples. They demonstrate UI
  plumbing but are not a reproducible evaluation experiment.
- Semantic cluster IDs are manually assigned; no semantic evaluator is run.
- Analysis IDs use random UUIDs and saved records have no schema version.
- There is no dataset abstraction, validation, upload, experiment persistence,
  human-label preservation, comparison, failure taxonomy, or export layer.
- AUROC and Brier are present, but AUPRC and risk-coverage/selective prediction are
  missing. Small-sample calibration warnings are missing.
- The frontend is polished and responsive, but its navigation and positioning still
  describe a hallucination detector rather than an evaluation platform.
- Several source/documentation strings contain mojibake and must be normalised.
- No secrets are tracked. `.env.example`, Docker, tests, and build scripts exist.
- Local CPU execution is practical; no NVIDIA runtime is available. Large local model
  downloads are not required for this iteration.

## Incremental implementation checklist

- [x] Replace generated-looking demo token scores with explicit `unavailable` signal
  capability; never synthesise provider log-probabilities.
- [x] Add central settings, structured errors/logging, deterministic IDs, and schema
  version `2.0`.
- [x] Add versioned Pydantic/TypeScript dataset and experiment contracts.
- [x] Add JSON/JSONL dataset validation and a multi-domain demonstration dataset.
- [x] Add deterministic and cached-demo provider capability abstractions.
- [x] Add correctness evaluators: normalised match, aliases, token F1, dates, and
  numeric tolerance.
- [x] Add semantic grouping abstraction with an honestly labelled lexical fallback.
- [x] Add ECE, Brier, AUROC, AUPRC, reliability bins, risk coverage, selective
  accuracy, latency percentiles, and signal comparison.
- [x] Add SQLite persistence, migration table, human overrides, raw-output
  traceability, duplicate/delete/rerun-friendly experiment records.
- [x] Add experiment runner with partial failures and truthful run states.
- [x] Add JSON/CSV/Markdown/HTML export and a CLI evaluation command.
- [x] Reposition and rebuild the frontend around Playground, Experiments, Datasets,
  Compare, Calibration, Failures, Reports, Settings, and Methodology.
- [x] Add dataset upload/preview, experiment configuration, comparison, threshold
  analysis, failure slices, raw-record inspection, and exports.
- [x] Expand backend/frontend/integration tests and verify locally.
- [x] Rewrite README and BUILD_REPORT with reproducibility commands and limitations.

GitHub publication and Vercel deployment were authorised after the local review.

## Post-review corrections

- Experiment signal checkboxes now change the saved configuration and at least one
  available signal is required.
- Human overrides now feed aggregate metrics, signal comparison, failure categories,
  and CSV exports through the same effective label.
- Experiment detail and comparison navigation are separate and directly accessible.
- Repeated dataset uploads receive incremented versions instead of overwriting the
  original manifest.
- The UI exposes experiment deletion and correctness override controls.

## Research iteration audit and implementation (2026-07-27)

- Corrected tie-dependent average precision and versioned classification metrics as
  `2.1`.
- Replaced the sentence-initial-capital entity heuristic with a narrower
  multi-token-capital extraction rule.
- Added live Ollama and generic OpenAI-compatible providers with redaction,
  capabilities, bounded retries, timeouts, usage, and partial-error traces.
- Added genuine local embedding clustering through Ollama `all-minilm`; lexical
  Jaccard remains an explicit fallback.
- Added Mirage Reliability Set v1 from pinned TruthfulQA and SQuAD 2.0 gold records,
  plus an honest validation report. No generated question was marked verified.
- Added bootstrap intervals, paired bootstrap signal differences, stratified Platt
  scaling, and isotonic calibration.
- Added incremental per-example saves, resume, failed-example reruns, and stale
  configuration separation through deterministic IDs.
- Added the Findings API/page and a non-generative Markdown research-report builder.
- Completed a genuine three-example `phi:latest` Ollama smoke experiment. It is not
  presented as the official 200-example study.
- Docker execution remains unverified because Docker Desktop is not running.
- The official three-model, 200-example research configuration exists but is not
  claimed as completed.
# Hardening iteration

- Metric definitions and eligibility now come from a central registry; invalid
  metrics remain null with an explicit reason.
- Classification, calibration, risk-coverage, semantic-entropy and composite-risk
  definitions were versioned after tie, missing-value and traceability changes.
- Lexical and embedding clustering now use deterministic connected components.
- Dataset-content and canonical configuration fingerprints are stored in new runs.
- Human overrides are append-only and continue to preserve automated labels.
- Bootstrap intervals store requested, valid and rejected resamples.
- Official Findings/report output requires completed, live, non-demonstration runs;
  research-v1 conclusions additionally require three 200-example model runs.
- Existing pre-hardening experiments remain readable but are reported as legacy
  provenance rather than silently upgraded.
