"""Mirage API v2: reproducible LLM reliability evaluation."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, ValidationError

from api.core import (
    DATASET_DIR, SCHEMA_VERSION,
    CachedDemoProvider,
    DatasetManifest,
    ExperimentConfig,
    ExperimentRecord,
    SignalValues,
    aggregate,
    export_experiment,
    list_manifests,
    load_manifest,
    normalise_text,
    run_experiment,
    stable_id,
    settings, store,
    utc_now,
)

logger = logging.getLogger("mirage.api")
app = FastAPI(
    title="Mirage Reliability Evaluation API",
    version=SCHEMA_VERSION,
    docs_url="/api/docs",
    description="Provider-independent evaluation of uncertainty signals against labelled factual errors.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


class AnalyseRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    reference_answer: str | None = Field(default=None, max_length=1000)
    domain: str = Field(default="general", max_length=80)
    sample_count: int = Field(default=6, ge=2, le=10)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    retrieval_enabled: bool = False


class StressRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    types: list[str] = Field(default_factory=lambda: ["neutral", "formal", "distractor", "leading"])


class BenchRequest(BaseModel):
    count: int = Field(default=18, ge=4, le=1000)
    seed: int = 42


class DatasetUpload(BaseModel):
    filename: str
    content: str


class OverrideRequest(BaseModel):
    human_label: bool
    note: str | None = Field(default=None, max_length=1000)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": "http_error", "message": str(exc.detail)}})
    logger.exception("request.failed path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "The evaluation request failed safely.", "request_path": request.url.path}},
    )


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mode": "local_research",
        "version": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "database": "sqlite",
    }


@app.get("/api/v1/system")
def system() -> dict[str, Any]:
    provider = CachedDemoProvider()
    return {
        "backend": True,
        "mode": "local_research",
        "provider": provider.info.provider,
        "model": provider.info.model,
        "capabilities": provider.info.capabilities.model_dump(),
        "token_uncertainty": "unavailable_without_provider_logprobs",
        "semantic_clustering": "lexical_fallback_jaccard",
        "database": "SQLite local persistence",
        "schema_version": SCHEMA_VERSION,
        "demo_data": True,
    }


@app.get("/api/v1/models")
def models() -> dict[str, Any]:
    cached = CachedDemoProvider().info
    return {
        "models": [
            cached.model_dump(),
            {
                "provider": "huggingface",
                "model": "configurable-local-model",
                "mode": "live",
                "available": False,
                "unavailable_reason": "No live local provider is configured in this CPU-only iteration.",
                "capabilities": {
                    "supports_logprobs": True,
                    "supports_seed": True,
                    "supports_streaming": True,
                    "supports_vision": False,
                    "supports_retrieval": False,
                    "supports_token_usage": True,
                },
            },
        ]
    }


def nearest_cached_example(question: str) -> Any:
    manifest = load_manifest()
    query = set(normalise_text(question).split())
    def overlap(example: Any) -> float:
        words = set(normalise_text(example.question).split())
        return len(query & words) / max(1, len(query | words))
    candidate = max(manifest.examples, key=overlap)
    return candidate if overlap(candidate) >= 0.25 else None


@app.post("/api/v1/analyse")
def analyse_route(req: AnalyseRequest) -> dict[str, Any]:
    example = nearest_cached_example(req.question)
    if example is None:
        return {
            "id": stable_id("analysis", {"question": req.question, "time_bucket": utc_now()[:13]}),
            "question": req.question,
            "answer": "",
            "tokens": [],
            "samples": [],
            "clusters": [],
            "semanticEntropy": None,
            "normalisedSemanticEntropy": None,
            "pTrue": None,
            "verification": None,
            "score": None,
            "breakdown": {},
            "mode": "cached_demo",
            "calibrationStatus": "not_available",
            "model": CachedDemoProvider().info.model,
            "device": "CPU",
            "latency": 0,
            "error": "No cached output matches this question. Configure a live provider or use a starter-dataset example.",
            "capabilities": CachedDemoProvider().info.capabilities.model_dump(),
            "metadata": {"schemaVersion": SCHEMA_VERSION, "executionMode": "cached_demo"},
        }
    config = ExperimentConfig()
    config.sampling.semantic_samples = req.sample_count
    config.sampling.temperature = req.temperature
    config.sampling.top_p = req.top_p
    from api.core import evaluate_example
    result = evaluate_example(example, config, CachedDemoProvider())
    return {
        "id": stable_id("analysis", {"example": example.id, "sampling": config.sampling.model_dump()}),
        "question": req.question,
        "matchedDatasetQuestion": example.question,
        "answer": result.raw_generation.response,
        "tokens": [],
        "tokenSignal": {
            "available": False,
            "reason": result.signals.token_uncertainty_reason,
            "rawLogprobs": None,
        },
        "samples": [
            {
                "id": f"sample_{i+1}",
                "answer": answer,
                "cluster": next((index for index, cluster in enumerate(result.semantic_clusters) if i in cluster.response_indices), 0),
                "latency": result.raw_generation.latency_ms / 1000,
                "agreement": max((cluster.probability for cluster in result.semantic_clusters if i in cluster.response_indices), default=0),
            }
            for i, answer in enumerate(result.raw_generation.sampled_responses)
        ],
        "clusters": [
            {
                "id": cluster.cluster_id,
                "label": f"Meaning cluster {i+1}",
                "sampleIds": [f"sample_{index+1}" for index in cluster.response_indices],
                "probability": cluster.probability,
                "representativeAnswer": cluster.representative_answer,
                "evaluatorMethod": cluster.evaluator_method,
            }
            for i, cluster in enumerate(result.semantic_clusters)
        ],
        "semanticEntropy": result.signals.semantic_entropy,
        "normalisedSemanticEntropy": result.signals.semantic_entropy,
        "pTrue": None if not result.verification else result.verification.confidence,
        "verification": None if not result.verification else result.verification.model_dump(),
        "score": None if result.predicted_risk is None else result.predicted_risk * 100,
        "breakdown": result.risk_contributions,
        "mode": "cached_demo",
        "calibrationStatus": "experimental_uncalibrated",
        "model": CachedDemoProvider().info.model,
        "device": "CPU",
        "latency": result.raw_generation.latency_ms / 1000,
        "correctness": result.correctness.model_dump(),
        "capabilities": CachedDemoProvider().info.capabilities.model_dump(),
        "metadata": {
            "schemaVersion": SCHEMA_VERSION,
            "executionMode": "cached_demo",
            "temperature": req.temperature,
            "topP": req.top_p,
            "sampleCount": req.sample_count,
            "timestamp": utc_now(),
            "datasetExampleId": example.id,
            "rawOutputSha256": result.trace.get("raw_output_sha256"),
        },
    }


@app.get("/api/v1/datasets")
def datasets() -> dict[str, Any]:
    manifests = list_manifests()
    return {
        "datasets": [
            {
                "name": manifest.name,
                "version": manifest.version,
                "description": manifest.description,
                "size": len(manifest.examples),
                "demonstration": manifest.demonstration,
                "license": manifest.license,
                "domains": sorted({example.domain for example in manifest.examples}),
                "difficulties": sorted({example.difficulty for example in manifest.examples}),
            }
            for manifest in manifests
        ]
    }


@app.get("/api/v1/datasets/{name}")
def dataset_detail(name: str) -> DatasetManifest:
    try:
        return load_manifest(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def parse_upload(upload: DatasetUpload) -> list[dict[str, Any]]:
    try:
        if upload.filename.lower().endswith(".jsonl"):
            return [json.loads(line) for line in upload.content.splitlines() if line.strip()]
        parsed = json.loads(upload.content)
        if isinstance(parsed, dict) and "examples" in parsed:
            return list(parsed["examples"])
        if isinstance(parsed, list):
            return parsed
        raise ValueError("JSON must be an array or a manifest containing examples.")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Dataset parsing failed: {exc}") from exc


@app.post("/api/v1/datasets/validate")
def validate_dataset(upload: DatasetUpload) -> dict[str, Any]:
    raw_examples = parse_upload(upload)
    valid, errors = [], []
    from api.core import DatasetExample
    for index, item in enumerate(raw_examples):
        try:
            valid.append(DatasetExample.model_validate(item))
        except ValidationError as exc:
            errors.append({"index": index, "errors": exc.errors(include_url=False)})
    return {
        "valid": not errors,
        "valid_count": len(valid),
        "invalid_count": len(errors),
        "errors": errors,
        "preview": [example.model_dump() for example in valid[:5]],
        "schema_version": SCHEMA_VERSION,
    }


@app.post("/api/v1/datasets")
def save_dataset(upload: DatasetUpload) -> dict[str, Any]:
    validation = validate_dataset(upload)
    if not validation["valid"]:
        raise HTTPException(status_code=422, detail={"message": "Dataset contains malformed examples.", "validation": validation})
    raw_examples = parse_upload(upload)
    if len(raw_examples) > settings.max_dataset_examples:
        raise HTTPException(status_code=422, detail=f"Dataset exceeds the local limit of {settings.max_dataset_examples} examples.")
    safe_name = re.sub(r"[^a-z0-9-]+", "-", upload.filename.rsplit(".", 1)[0].casefold()).strip("-") or "uploaded-dataset"
    base_name = safe_name
    existing_versions = [
        manifest.version
        for manifest in list_manifests()
        if manifest.name == base_name or manifest.name.startswith(f"{base_name}-v")
    ]
    version = "1.0"
    if existing_versions:
        numeric_versions = []
        for existing in existing_versions:
            match = re.fullmatch(r"(\d+)\.(\d+)", existing)
            if match:
                numeric_versions.append((int(match.group(1)), int(match.group(2))))
        major, minor = max(numeric_versions, default=(1, 0))
        version = f"{major}.{minor + 1}"
        safe_name = f"{base_name}-v{version.replace('.', '-')}"
    manifest = DatasetManifest.model_validate({
        "name": safe_name,
        "version": version,
        "description": "User-uploaded local evaluation dataset.",
        "license": "User supplied; verify rights before redistribution.",
        "demonstration": False,
        "examples": raw_examples,
    })
    destination = DATASET_DIR / f"{safe_name}.json"
    destination.write_text(manifest.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    return {"saved": True, "name": safe_name, "version": version, "size": len(manifest.examples)}


@app.post("/api/v1/experiments", response_model=ExperimentRecord)
def create_experiment(config: ExperimentConfig) -> ExperimentRecord:
    if config.provider != "cached_demo":
        raise HTTPException(status_code=422, detail="Only the clearly labelled cached_demo provider is configured locally.")
    manifest = load_manifest(config.dataset_name)
    if config.dataset_version != manifest.version:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset version mismatch: requested {config.dataset_version}, available {manifest.version}.",
        )
    record = run_experiment(config)
    store.save(record)
    return record


@app.get("/api/v1/experiments")
def experiments() -> dict[str, Any]:
    records = store.list()
    return {
        "experiments": [
            {
                "experiment_id": record.experiment_id,
                "experiment_name": record.experiment_name,
                "creation_time": record.creation_time,
                "state": record.state,
                "dataset": record.dataset,
                "model": record.model.model_dump(),
                "aggregates": None if not record.aggregates else record.aggregates.model_dump(),
                "schema_version": record.schema_version,
            }
            for record in records
        ]
    }


@app.get("/api/v1/experiments/{experiment_id}", response_model=ExperimentRecord)
def experiment_detail(experiment_id: str) -> ExperimentRecord:
    record = store.get(experiment_id)
    if not record:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return record


@app.delete("/api/v1/experiments/{experiment_id}")
def delete_experiment(experiment_id: str) -> dict[str, bool]:
    return {"deleted": store.delete(experiment_id)}


@app.post("/api/v1/experiments/{experiment_id}/examples/{example_id}/override", response_model=ExperimentRecord)
def override_label(experiment_id: str, example_id: str, req: OverrideRequest) -> ExperimentRecord:
    try:
        return store.override(experiment_id, example_id, req.human_label, req.note)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Experiment or example not found: {exc}") from exc


@app.get("/api/v1/experiments/{experiment_id}/export")
def export(experiment_id: str, format: Literal["json", "csv", "markdown", "html"] = "json") -> Response:
    record = store.get(experiment_id)
    if not record:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    body, media_type = export_experiment(record, format)
    extension = "md" if format == "markdown" else format
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{experiment_id}.{extension}"'},
    )


@app.post("/api/v1/experiments/compare")
async def compare_experiments(request: Request) -> dict[str, Any]:
    body = await request.json()
    ids = body.get("experiment_ids", [])
    records = [record for experiment_id in ids if (record := store.get(experiment_id))]
    return {
        "experiments": [
            {
                "experiment_id": record.experiment_id,
                "name": record.experiment_name,
                "model": record.model.model,
                "dataset": record.dataset,
                "config": record.config.model_dump(),
                "aggregates": None if not record.aggregates else record.aggregates.model_dump(),
                "domain_breakdown": domain_breakdown(record),
                "difficulty_breakdown": difficulty_breakdown(record),
            }
            for record in records
        ]
    }


def breakdown(record: ExperimentRecord, key: str) -> list[dict[str, Any]]:
    values = sorted({getattr(result, key) for result in record.results})
    rows = []
    for value in values:
        subset = [result for result in record.results if getattr(result, key) == value and result.correctness.correct is not None]
        if subset:
            rows.append({
                key: value,
                "count": len(subset),
                "accuracy": sum(bool(item.correctness.correct) for item in subset) / len(subset),
                "mean_risk": sum(item.predicted_risk or 0 for item in subset) / len(subset),
            })
    return rows


def domain_breakdown(record: ExperimentRecord) -> list[dict[str, Any]]:
    return breakdown(record, "domain")


def difficulty_breakdown(record: ExperimentRecord) -> list[dict[str, Any]]:
    return breakdown(record, "difficulty")


@app.post("/api/v1/bench/runs")
def legacy_bench(req: BenchRequest) -> dict[str, Any]:
    config = ExperimentConfig(experiment_name="Starter signal comparison")
    config.sampling.seed = req.seed
    record = run_experiment(config)
    record.results = record.results[: req.count]
    record.aggregates = aggregate(record.results)
    store.save(record)
    metrics = record.aggregates
    assert metrics is not None
    return {
        "id": record.experiment_id,
        "dataset": f"{record.dataset['name']} {record.dataset['version']}",
        "count": metrics.total_examples,
        "incorrect": round((1 - (metrics.accuracy or 0)) * metrics.labelled_examples),
        "auroc": metrics.auroc,
        "auprc": metrics.auprc,
        "ece": metrics.ece,
        "brier": metrics.brier,
        "bins": [
            {
                "range": f"{row['low']:.1f}-{row['high']:.1f}",
                "predicted": row["predicted"],
                "observed": row["observed"],
                "count": row["count"],
            }
            for row in metrics.reliability_bins
        ],
        "records": [
            {
                "question": result.question,
                "reference": result.reference_answer or "Explicitly unanswerable",
                "correct": result.correctness.correct,
                "risk": result.predicted_risk,
                "method": result.correctness.method,
            }
            for result in record.results
        ],
        "mode": "cached_demo_experiment",
        "timestamp": record.completed_time,
        "warnings": metrics.warnings,
        "riskCoverage": metrics.risk_coverage,
        "signalComparison": metrics.signal_comparison,
    }


@app.post("/api/v1/stress")
def stress(req: StressRequest) -> dict[str, Any]:
    transformations = {
        "neutral": lambda q: f"In other words, {q[:1].lower()}{q[1:]}",
        "formal": lambda q: f"Provide a precise response to: {q}",
        "conversational": lambda q: f"Quick question: {q}",
        "distractor": lambda q: f"Ignore unrelated details and answer: {q}",
        "leading": lambda q: f"Adversarial leading-premise test: {q}",
        "ambiguity": lambda q: f"Depending on interpretation, {q[:1].lower()}{q[1:]}",
        "negation": lambda q: f"Adversarial negation test: Is it false that {q}",
    }
    analysis = analyse_route(AnalyseRequest(question=req.question, sample_count=4))
    if analysis.get("error"):
        raise HTTPException(status_code=422, detail=analysis["error"])
    variants = [
        {
            "id": index,
            "type": kind,
            "question": transformations.get(kind, transformations["neutral"])(req.question),
            "answer": analysis["answer"],
            "score": analysis["score"],
            "agreement": 1.0,
            "relation": "not_recomputed_cached_output",
            "adversarial": kind in {"negation", "leading"},
        }
        for index, kind in enumerate(req.types[:8])
    ]
    return {
        "original": analysis,
        "variants": variants,
        "stability": None,
        "instability": None,
        "summary": "Variants were generated, but cached outputs were not recomputed. No stability metric is claimed.",
        "mode": "cached_demo",
    }


@app.get("/api/v1/demo/examples")
def examples() -> dict[str, Any]:
    manifest = load_manifest()
    return {"examples": [example.question for example in manifest.examples[:8]], "dataset": manifest.name}
