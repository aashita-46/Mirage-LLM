"""Versioned, provider-independent evaluation core for Mirage."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import sqlite3
import statistics
import subprocess
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

SCHEMA_VERSION = "2.0"
METRIC_VERSIONS = {
    "classification": "2.2",
    "calibration": "2.1",
    "risk_coverage": "2.1",
    "semantic_entropy": "2.2",
    "mirage_risk_score": "2.1",
}
METRIC_REGISTRY: dict[str, dict[str, Any]] = {
    "accuracy": {"version": "2.2", "direction": "higher_is_better", "range": [0.0, 1.0], "minimum_samples": 1, "requires_binary_labels": True, "requires_both_classes": False, "description": "Fraction of effective correctness labels that are correct."},
    "auroc": {"version": "2.1", "direction": "higher_is_better", "range": [0.0, 1.0], "minimum_samples": 2, "requires_binary_labels": True, "requires_both_classes": True, "description": "Pairwise ranking probability for incorrect versus correct examples; ties receive half credit."},
    "auprc": {"version": "2.1", "direction": "higher_is_better", "range": [0.0, 1.0], "minimum_samples": 1, "requires_binary_labels": True, "requires_both_classes": False, "description": "Average precision for the incorrect class using tied-score thresholds."},
    "ece": {"version": "2.1", "direction": "lower_is_better", "range": [0.0, 1.0], "minimum_samples": 1, "requires_binary_labels": True, "requires_both_classes": False, "description": "Weighted absolute difference between predicted error risk and observed error frequency in fixed-width bins."},
    "brier": {"version": "2.1", "direction": "lower_is_better", "range": [0.0, 1.0], "minimum_samples": 1, "requires_binary_labels": True, "requires_both_classes": False, "description": "Mean squared error between bounded predicted risk and binary error labels."},
    "negative_log_likelihood": {"version": "2.1", "direction": "lower_is_better", "range": [0.0, None], "minimum_samples": 1, "requires_binary_labels": True, "requires_both_classes": False, "description": "Binary log loss with probabilities clipped only for numerical stability."},
    "semantic_entropy": {"version": "2.2", "direction": "higher_is_more_uncertain", "range": [0.0, 1.0], "minimum_samples": 2, "requires_binary_labels": False, "requires_both_classes": False, "description": "Normalised Shannon entropy over semantic response clusters."},
    "numerical_variance": {"version": "2.1", "direction": "higher_is_more_uncertain", "range": [0.0, None], "minimum_samples": 2, "requires_binary_labels": False, "requires_both_classes": False, "description": "Population variance of extracted numeric answers; unavailable with fewer than two numeric responses."},
    "mirage_risk_score": {"version": "2.1", "direction": "higher_is_more_error_risk", "range": [0.0, 1.0], "minimum_samples": 1, "requires_binary_labels": False, "requires_both_classes": False, "description": "Experimental weighted composite of available bounded uncertainty signals; not a probability unless calibrated out of sample."},
}
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "datasets"


class Settings(BaseModel):
    environment: str = os.getenv("MIRAGE_ENV", "development")
    mode: str = os.getenv("MIRAGE_MODE", "local_research")
    database_path: Path = Path(os.getenv(
        "MIRAGE_DATABASE_PATH",
        "/tmp/mirage.db" if os.getenv("VERCEL") else DATA_DIR / "mirage.db",
    ))
    log_level: str = os.getenv("MIRAGE_LOG_LEVEL", "INFO")
    max_dataset_examples: int = int(os.getenv("MIRAGE_MAX_DATASET_EXAMPLES", "1000"))
    schema_version: str = SCHEMA_VERSION


settings = Settings()
DB_PATH = settings.database_path

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("mirage")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


class DatasetExample(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(min_length=1, max_length=120)
    question: str = Field(min_length=3, max_length=4000)
    reference_answer: str | None = None
    acceptable_answers: list[str] = Field(default_factory=list)
    unanswerable: bool = False
    answerability: Literal["answerable", "unanswerable", "false_premise", "ambiguous"] = "answerable"
    domain: str = "general"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    source: str
    question_type: str = "factual"
    source_name: str = ""
    source_reference: str = ""
    source_date: str = ""
    verification_status: Literal["verified", "pending"] = "pending"
    verification_notes: str = ""
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reference_answer")
    @classmethod
    def reference_required(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not value.strip():
            return None
        return value


class DatasetManifest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    name: str
    version: str
    description: str
    license: str = "Repository demonstration data"
    demonstration: bool = True
    source_revisions: dict[str, str] = Field(default_factory=dict)
    build_script_version: str | None = None
    build_timestamp: str | None = None
    examples: list[DatasetExample]


class ProviderCapabilities(BaseModel):
    supports_logprobs: bool = False
    supports_seed: bool = True
    supports_streaming: bool = False
    supports_parallel_samples: bool = False
    supports_structured_output: bool = False
    supports_vision: bool = False
    supports_retrieval: bool = False
    supports_token_usage: bool = True


class ModelInfo(BaseModel):
    provider: str
    model: str
    version: str | None = None
    mode: Literal["live", "cached_demo", "test_mock"]
    capabilities: ProviderCapabilities


class SamplingConfig(BaseModel):
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    max_tokens: int = Field(default=180, ge=8, le=2048)
    seed: int = 42
    semantic_samples: int = Field(default=6, ge=2, le=10)
    timeout_seconds: float = Field(default=120, ge=1, le=600)
    retry_count: int = Field(default=2, ge=0, le=8)
    max_parallel_requests: int = Field(default=1, ge=1, le=16)


class SignalConfig(BaseModel):
    semantic_entropy: bool = True
    response_consistency: bool = True
    self_verification: bool = True
    token_uncertainty: bool = False
    retrieval_faithfulness: bool = False
    clustering_method: Literal["lexical_fallback", "embedding", "nli", "embedding_nli_hybrid"] = "lexical_fallback"
    embedding_model: str = "all-minilm"
    similarity_threshold: float = Field(default=0.78, ge=0, le=1)
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "semantic_entropy": 0.45,
            "response_inconsistency": 0.30,
            "self_verification_uncertainty": 0.25,
        }
    )


class EvaluatorConfig(BaseModel):
    methods: list[str] = Field(
        default_factory=lambda: ["acceptable_answer", "normalised_exact_match", "token_f1"]
    )
    numeric_tolerance: float = Field(default=0.01, ge=0)
    partial_credit_threshold: float = Field(default=0.78, ge=0, le=1)
    llm_judge_enabled: bool = False


class ExperimentConfig(BaseModel):
    experiment_name: str = Field(default="Starter reliability study", min_length=2, max_length=160)
    dataset_name: str = "mirage-starter"
    dataset_version: str = "1.0"
    provider: str = "cached_demo"
    model: str = "mirage/cached-research-samples"
    prompt_template: str = "{question}"
    system_prompt: str = "Answer concisely. If the premise is false or unknowable, say so."
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    signals: SignalConfig = Field(default_factory=SignalConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    retrieval_enabled: bool = False
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
    verified_only: bool = True
    max_examples: int | None = Field(default=None, ge=1, le=5000)
    bootstrap_resamples: int = Field(default=1000, ge=0, le=10000)
    bootstrap_seed: int = 42
    experiment_group: str | None = None


class GenerationRecord(BaseModel):
    response: str
    sampled_responses: list[str]
    token_logprobs: list[float] | None = None
    token_entropies: list[float] | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float
    estimated_cost: float | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    retry_count: int = 0


class SemanticCluster(BaseModel):
    cluster_id: str
    response_indices: list[int]
    responses: list[str]
    size: int
    representative_answer: str
    probability: float
    evaluator_method: str


class VerificationClaim(BaseModel):
    claim: str
    status: Literal["supported", "unsupported", "uncertain"]


class VerificationResult(BaseModel):
    verdict: Literal["supported", "uncertain", "contradicted"]
    confidence: float = Field(ge=0, le=1)
    reason: str
    claims: list[VerificationClaim] = Field(default_factory=list)
    source: str


class CorrectnessResult(BaseModel):
    correct: bool | None
    score: float | None
    exact_match: float | None
    token_f1: float | None
    method: str
    reason: str
    error_type: str
    automated_label: bool | None
    human_label: bool | None = None
    human_override_at: str | None = None
    human_note: str | None = None
    human_override_history: list[dict[str, Any]] = Field(default_factory=list)
    partial_credit_threshold: float | None = None


class SignalValues(BaseModel):
    token_uncertainty: float | None = None
    token_uncertainty_reason: str | None = None
    semantic_entropy: float | None = None
    response_consistency: float | None = None
    self_verification_uncertainty: float | None = None
    retrieval_unsupported_claim_rate: float | None = None
    exact_response_consistency: float | None = None
    answer_switch_rate: float | None = None
    numerical_variance: float | None = None
    entity_disagreement_rate: float | None = None


class ExampleResult(BaseModel):
    example_id: str
    question: str
    reference_answer: str | None
    acceptable_answers: list[str]
    domain: str
    difficulty: str
    source: str
    tags: list[str]
    raw_generation: GenerationRecord
    semantic_clusters: list[SemanticCluster]
    verification: VerificationResult | None
    signals: SignalValues
    predicted_risk: float | None
    risk_contributions: dict[str, float]
    risk_trace: dict[str, Any] = Field(default_factory=dict)
    correctness: CorrectnessResult
    failure_types: list[str]
    retrieval_context: list[dict[str, Any]] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
    error_state: str | None = None


class AggregateMetrics(BaseModel):
    total_examples: int
    labelled_examples: int
    failed_examples: int
    accuracy: float | None
    exact_match: float | None
    token_f1: float | None
    auroc: float | None
    auprc: float | None
    ece: float | None
    brier: float | None
    negative_log_likelihood: float | None
    mean_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    average_input_tokens: float | None
    average_output_tokens: float | None
    total_estimated_cost: float | None
    reliability_bins: list[dict[str, Any]]
    risk_coverage: list[dict[str, float]]
    signal_comparison: list[dict[str, Any]]
    warnings: list[str]
    metric_status: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ExperimentRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    experiment_id: str
    experiment_name: str
    creation_time: str
    completed_time: str | None = None
    state: Literal["pending", "running", "partially_complete", "completed", "cancelled", "failed"]
    dataset: dict[str, Any]
    model: ModelInfo
    config: ExperimentConfig
    git_commit: str | None = None
    results: list[ExampleResult]
    aggregates: AggregateMetrics | None = None
    metric_versions: dict[str, str] = Field(default_factory=lambda: dict(METRIC_VERSIONS))
    research_analysis: dict[str, Any] = Field(default_factory=dict)
    configuration_fingerprint: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


def normalise_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = value.replace("’", "'").replace("“", '"').replace("”", '"')
    value = re.sub(r"(?<=\d),(?=\d)", "", value)
    value = re.sub(r"[^\w\s.\-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .")


def token_f1(prediction: str, reference: str) -> float:
    pred = normalise_text(prediction).split()
    ref = normalise_text(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def extract_number(value: str) -> float | None:
    match = re.search(r"-?(?:\d+(?:,\d{3})*|\d*\.\d+)(?:[eE][+-]?\d+)?", value)
    return float(match.group(0).replace(",", "")) if match else None


def numeric_match(prediction: str, reference: str, tolerance: float) -> bool | None:
    pred, ref = extract_number(prediction), extract_number(reference)
    if pred is None or ref is None:
        return None
    pred_pct, ref_pct = "%" in prediction, "%" in reference
    if pred_pct != ref_pct:
        return False
    units = r"\b(km|m|cm|mm|kg|g|mg|lb|mph|km/h|°c|°f|usd|eur)\b"
    pred_unit = re.search(units, prediction.casefold())
    ref_unit = re.search(units, reference.casefold())
    if pred_unit and ref_unit and pred_unit.group(1) != ref_unit.group(1):
        return False
    allowed = max(tolerance, abs(ref) * tolerance)
    return abs(pred - ref) <= allowed


def correctness(example: DatasetExample, answer: str, config: EvaluatorConfig) -> CorrectnessResult:
    candidates = [x for x in [example.reference_answer, *example.acceptable_answers] if x]
    answer_norm = normalise_text(answer)
    if example.answerability in {"unanswerable", "ambiguous"} or example.unanswerable:
        valid_markers = ("cannot be determined", "insufficient information", "not enough information", "does not provide enough", "no valid answer")
        evasive_only = answer_norm in {"i am not sure", "maybe", "it depends", "unknown"}
        ok = any(marker in answer_norm for marker in valid_markers) and not evasive_only
        return CorrectnessResult(
            correct=ok, score=float(ok), exact_match=None, token_f1=None,
            method="unanswerable_marker", reason="Checked for an explicit abstention or premise rejection.",
            error_type="none" if ok else "unanswerable_question_answered", automated_label=ok,
            partial_credit_threshold=config.partial_credit_threshold,
        )
    if example.answerability == "false_premise":
        rejection = any(marker in answer_norm for marker in ("false premise", "premise is incorrect", "did not", "does not exist", "no such"))
        invented = any(normalise_text(candidate) in answer_norm for candidate in candidates) if candidates else False
        ok = rejection and not invented
        return CorrectnessResult(
            correct=ok, score=float(ok), exact_match=None, token_f1=None,
            method="false_premise_rejection",
            reason="Required an explicit rejection of the premise without inventing the requested answer.",
            error_type="none" if ok else "false_premise_accepted", automated_label=ok,
            partial_credit_threshold=config.partial_credit_threshold,
        )
    exact = any(answer_norm == normalise_text(candidate) for candidate in candidates)
    alias = any(normalise_text(candidate) in answer_norm for candidate in candidates)
    f1 = max((token_f1(answer, candidate) for candidate in candidates), default=0.0)
    numeric = any(numeric_match(answer, candidate, config.numeric_tolerance) is True for candidate in candidates)
    ok = exact or alias or numeric or f1 >= config.partial_credit_threshold
    error_type = "none" if ok else classify_error(answer, example.reference_answer or "")
    return CorrectnessResult(
        correct=ok, score=max(float(exact or alias or numeric), f1), exact_match=float(exact),
        token_f1=f1, method="deterministic_composite",
        reason="Evaluated with aliases, normalised matching, token F1, and numerical tolerance.",
        error_type=error_type, automated_label=ok, partial_credit_threshold=config.partial_credit_threshold,
    )


def classify_error(answer: str, reference: str) -> str:
    a_num, r_num = extract_number(answer), extract_number(reference)
    if a_num is not None and r_num is not None and a_num != r_num:
        return "wrong_number"
    if re.search(r"\b(?:19|20)\d{2}\b", answer) and re.search(r"\b(?:19|20)\d{2}\b", reference):
        return "wrong_date"
    if any(word[:1].isupper() for word in answer.split()) and any(word[:1].isupper() for word in reference.split()):
        return "wrong_entity"
    return "incorrect_answer"


def lexical_similarity(a: str, b: str) -> float:
    left, right = set(normalise_text(a).split()), set(normalise_text(b).split())
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def semantic_clusters(responses: list[str], threshold: float = 0.42) -> list[SemanticCluster]:
    matrix = [[lexical_similarity(a, b) for b in responses] for a in responses]
    linked = [[i == j or matrix[i][j] >= threshold for j in range(len(responses))] for i in range(len(responses))]
    from api.semantics import connected_components
    groups = connected_components(linked, responses)
    return [
        SemanticCluster(
            cluster_id=f"cluster_{idx + 1}",
            response_indices=group,
            responses=[responses[i] for i in group],
            size=len(group),
            representative_answer=min(
                (responses[i] for i in group),
                key=lambda text: (-sum(matrix[responses.index(text)][j] for j in group), normalise_text(text), text),
            ),
            probability=len(group) / len(responses),
            evaluator_method="lexical_fallback_jaccard",
        )
        for idx, group in enumerate(groups)
    ]


def entropy_from_clusters(clusters: list[SemanticCluster], sample_count: int) -> float | None:
    if sample_count <= 1:
        return None
    raw = -sum(c.probability * math.log(c.probability) for c in clusters if c.probability)
    return raw / math.log(sample_count)


def consistency_signals(responses: list[str], clusters: list[SemanticCluster]) -> dict[str, float | None]:
    normalised = [normalise_text(x) for x in responses]
    concentration = max((c.probability for c in clusters), default=0.0)
    numbers = [n for n in (extract_number(x) for x in responses) if n is not None]
    ignored = {"The", "This", "That", "There", "It", "A", "An", "Answer"}
    entities = [
        {entity for entity in re.findall(r"\b(?:[A-Z][\w.-]*)(?:\s+[A-Z][\w.-]*)*\b", x) if entity not in ignored}
        for x in responses
    ]
    entity_union = set().union(*entities) if entities else set()
    entity_intersection = set.intersection(*entities) if entities else set()
    return {
        "response_consistency": concentration,
        "exact_response_consistency": max(Counter(normalised).values(), default=0) / max(1, len(responses)),
        "answer_switch_rate": 1 - concentration,
        "numerical_variance": statistics.pvariance(numbers) if len(numbers) > 1 else None,
        "entity_disagreement_rate": None if not entity_union else 1 - (len(entity_intersection) / len(entity_union)),
    }


def combine_risk(signals: SignalValues, weights: dict[str, float]) -> tuple[float | None, dict[str, float], dict[str, Any]]:
    values = {
        "semantic_entropy": signals.semantic_entropy,
        "response_inconsistency": None if signals.response_consistency is None else 1 - signals.response_consistency,
        "self_verification_uncertainty": signals.self_verification_uncertainty,
        "token_uncertainty": signals.token_uncertainty,
        "retrieval_unsupported_claim_rate": signals.retrieval_unsupported_claim_rate,
    }
    active = {name: (value, max(0.0, weights.get(name, 0.0))) for name, value in values.items() if value is not None and weights.get(name, 0) > 0}
    total = sum(weight for _, weight in active.values())
    if not active or total == 0:
        return None, {}, {"version": METRIC_VERSIONS["mirage_risk_score"], "warnings": ["No enabled, available, positively weighted signals."], "raw_values": values, "configured_weights": weights, "effective_weights": {}}
    contributions = {name: value * weight / total for name, (value, weight) in active.items()}
    effective = {name: weight / total for name, (_, weight) in active.items()}
    return min(1.0, max(0.0, sum(contributions.values()))), contributions, {
        "version": METRIC_VERSIONS["mirage_risk_score"], "enabled_signals": sorted(active),
        "raw_values": values, "normalised_values": {name: value for name, (value, _) in active.items()},
        "configured_weights": weights, "effective_weights": effective,
        "direction": {name: "higher_is_more_error_risk" for name in active},
        "contributions": contributions, "warnings": [],
    }


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    thresholds = sorted(set(scores), reverse=True)
    previous_recall, area = 0.0, 0.0
    for threshold in thresholds:
        predicted = [score >= threshold for score in scores]
        true_positive = sum(label and selected for label, selected in zip(labels, predicted))
        selected_count = sum(predicted)
        recall = true_positive / positives
        precision = true_positive / selected_count if selected_count else 1.0
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def calibration(labels: list[int], scores: list[float], bins: int = 10) -> tuple[float | None, float | None, float | None, list[dict[str, Any]]]:
    if not labels or len(labels) != len(scores) or any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        return None, None, None, []
    rows, ece = [], 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        values = [(s, y) for s, y in zip(scores, labels) if low <= s < high or (index == bins - 1 and s == 1)]
        if not values:
            continue
        predicted = sum(s for s, _ in values) / len(values)
        observed = sum(y for _, y in values) / len(values)
        ece += len(values) / len(labels) * abs(predicted - observed)
        rows.append({"low": low, "high": high, "predicted": predicted, "observed": observed, "count": len(values)})
    brier = sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(labels)
    epsilon = 1e-12
    nll = -sum(y * math.log(max(epsilon, s)) + (1-y) * math.log(max(epsilon, 1-s)) for s, y in zip(scores, labels)) / len(labels)
    return ece, brier, nll, rows


def risk_coverage(labels: list[int], risks: list[float]) -> list[dict[str, float]]:
    if not labels or len(labels) != len(risks):
        return []
    ordered = sorted(enumerate(zip(risks, labels)), key=lambda item: (item[1][0], item[0]))
    rows = []
    rows.append({
        "coverage": 0.0, "risk_threshold": 0.0, "selective_accuracy": 1.0,
        "error_rate": 0.0, "review_rate": 1.0, "remaining_errors": 0.0,
    })
    for accepted in range(1, len(ordered) + 1):
        subset = ordered[:accepted]
        errors = sum(label for _, (_, label) in subset)
        coverage = accepted / len(ordered)
        rows.append({
            "coverage": coverage,
            "risk_threshold": subset[-1][1][0],
            "selective_accuracy": 1 - errors / accepted,
            "error_rate": errors / accepted,
            "review_rate": 1 - coverage,
            "remaining_errors": float(errors),
        })
    return rows


class CachedDemoProvider:
    info = ModelInfo(
        provider="cached_demo",
        model="mirage/cached-research-samples",
        version="1.0",
        mode="cached_demo",
        capabilities=ProviderCapabilities(),
    )

    def generate(self, example: DatasetExample, config: SamplingConfig) -> GenerationRecord:
        cached = example.metadata.get("cached_demo")
        if not cached:
            return GenerationRecord(
                response="", sampled_responses=[], latency_ms=0,
                error="No cached provider output exists for this example.",
                provider_metadata={"execution_mode": "cached_demo"},
            )
        samples = list(cached.get("samples", []))[: config.semantic_samples]
        return GenerationRecord(
            response=cached["response"],
            sampled_responses=samples or [cached["response"]],
            token_logprobs=None,
            token_entropies=None,
            token_usage={
                "input": len(example.question.split()),
                "output": len(cached["response"].split()),
            },
            latency_ms=float(cached.get("latency_ms", 0)),
            estimated_cost=0,
            provider_metadata={
                "execution_mode": "cached_demo",
                "cache_version": "1.0",
                "logprobs_unavailable": True,
                "self_verification": cached.get("self_verification"),
            },
        )


def load_manifest(name: str = "mirage-starter") -> DatasetManifest:
    path = DATASET_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset {name!r} does not exist.")
    return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))


def list_manifests() -> list[DatasetManifest]:
    return [DatasetManifest.model_validate_json(path.read_text(encoding="utf-8")) for path in sorted(DATASET_DIR.glob("*.json"))]


def verification_from_cache(record: GenerationRecord) -> VerificationResult | None:
    cached = record.provider_metadata.get("self_verification")
    if not cached:
        return None
    source = (
        "cached_model_self_verification"
        if record.provider_metadata.get("execution_mode") == "cached_demo"
        else "live_model_self_verification"
    )
    try:
        return VerificationResult.model_validate({**cached, "source": source})
    except ValidationError as exc:
        record.provider_metadata["self_verification_parsing_error"] = str(exc)
        return None


def failure_types(result: CorrectnessResult, risk: float | None, signals: SignalValues) -> list[str]:
    failures: list[str] = []
    label = result.human_label if result.human_label is not None else result.correct
    if label is False:
        failures.append(result.error_type)
        if risk is not None and risk < 0.35:
            failures.append("overconfident_error")
    if label is True and risk is not None and risk > 0.65:
        failures.append("underconfident_correct_answer")
    if signals.semantic_entropy is not None and signals.semantic_entropy > 0.55:
        failures.append("semantic_disagreement")
    return sorted(set(failures))


def evaluate_example(example: DatasetExample, config: ExperimentConfig, provider: Any) -> ExampleResult:
    raw = provider.generate(example, config.sampling)
    if raw.error:
        empty_correctness = CorrectnessResult(
            correct=None, score=None, exact_match=None, token_f1=None, method="not_run",
            reason=raw.error, error_type="provider_failure", automated_label=None,
        )
        return ExampleResult(
            example_id=example.id, question=example.question, reference_answer=example.reference_answer,
            acceptable_answers=example.acceptable_answers, domain=example.domain, difficulty=example.difficulty,
            source=example.source, tags=example.tags, raw_generation=raw, semantic_clusters=[],
            verification=None, signals=SignalValues(token_uncertainty_reason="Provider did not return log probabilities."),
            predicted_risk=None, risk_contributions={}, correctness=empty_correctness,
            failure_types=["tool_or_api_failure"], error_state=raw.error,
        )
    semantic_metadata: dict[str, Any]
    if config.signals.clustering_method == "lexical_fallback":
        clusters = semantic_clusters(raw.sampled_responses)
        semantic_metadata = {
            "method": "lexical_fallback_jaccard",
            "threshold": config.signals.similarity_threshold,
        }
    else:
        from api.semantics import cluster_responses
        clusters, semantic_metadata = cluster_responses(
            raw.sampled_responses, method=config.signals.clustering_method,
            threshold=config.signals.similarity_threshold,
            embedding_model=config.signals.embedding_model,
        )
    entropy = entropy_from_clusters(clusters, len(raw.sampled_responses))
    consistency = consistency_signals(raw.sampled_responses, clusters)
    verification = verification_from_cache(raw)
    signals = SignalValues(
        token_uncertainty=None,
        token_uncertainty_reason="Unavailable: the cached demonstration provider does not expose token log probabilities.",
        semantic_entropy=entropy if config.signals.semantic_entropy else None,
        response_consistency=consistency["response_consistency"] if config.signals.response_consistency else None,
        self_verification_uncertainty=(1 - verification.confidence) if verification and config.signals.self_verification else None,
        exact_response_consistency=consistency["exact_response_consistency"],
        answer_switch_rate=consistency["answer_switch_rate"],
        numerical_variance=consistency["numerical_variance"],
        entity_disagreement_rate=consistency["entity_disagreement_rate"],
    )
    predicted_risk, contributions, risk_trace = combine_risk(signals, config.signals.weights)
    judged = correctness(example, raw.response, config.evaluator)
    return ExampleResult(
        example_id=example.id, question=example.question, reference_answer=example.reference_answer,
        acceptable_answers=example.acceptable_answers, domain=example.domain, difficulty=example.difficulty,
        source=example.source, tags=example.tags, raw_generation=raw, semantic_clusters=clusters,
        verification=verification, signals=signals, predicted_risk=predicted_risk,
        risk_contributions=contributions, risk_trace=risk_trace, correctness=judged,
        failure_types=failure_types(judged, predicted_risk, signals),
        trace={
            "schema_version": SCHEMA_VERSION,
            "raw_output_sha256": hashlib.sha256(raw.response.encode()).hexdigest(),
            "clustering": semantic_metadata,
            "correctness_method": judged.method,
            "metric_versions": METRIC_VERSIONS,
        },
    )


def aggregate(results: list[ExampleResult]) -> AggregateMetrics:
    valid = [r for r in results if r.correctness.correct is not None and r.predicted_risk is not None]
    labels = [int(not bool(r.correctness.human_label if r.correctness.human_label is not None else r.correctness.correct)) for r in valid]
    risks = [float(r.predicted_risk) for r in valid]
    ece, brier, nll, bins = calibration(labels, risks)
    comparisons = []
    for field, invert in [
        ("semantic_entropy", False),
        ("response_consistency", True),
        ("self_verification_uncertainty", False),
    ]:
        rows = [
            (
                getattr(r.signals, field),
                int(not bool(
                    r.correctness.human_label
                    if r.correctness.human_label is not None
                    else r.correctness.correct
                )),
            )
            for r in results
            if r.correctness.correct is not None
        ]
        rows = [(1-v if invert else v, y) for v, y in rows if v is not None]
        signal_scores, signal_labels = [x[0] for x in rows], [x[1] for x in rows]
        signal_ece, signal_brier, _, _ = calibration(signal_labels, signal_scores)
        comparisons.append({
            "signal": field,
            "coverage": len(rows) / max(1, len(results)),
            "auroc": roc_auc(signal_labels, signal_scores),
            "auprc": average_precision(signal_labels, signal_scores),
            "ece": signal_ece,
            "brier": signal_brier,
        })
    latencies = [r.raw_generation.latency_ms for r in results if not r.error_state]
    exact = [r.correctness.exact_match for r in results if r.correctness.exact_match is not None]
    f1s = [r.correctness.token_f1 for r in results if r.correctness.token_f1 is not None]
    warnings = []
    if len(valid) < 30:
        warnings.append("Calibration metrics are unstable with fewer than 30 labelled examples.")
    status = {
        "accuracy": {"available": bool(labels), "reason": None if labels else "No labelled, risk-eligible examples."},
        "auroc": {"available": len(set(labels)) == 2, "reason": None if len(set(labels)) == 2 else "Requires both correct and incorrect classes."},
        "auprc": {"available": bool(labels) and any(labels), "reason": None if labels and any(labels) else "Requires at least one incorrect example."},
        "calibration": {"available": bool(labels), "reason": None if labels else "Requires bounded risks and correctness labels."},
    }
    return AggregateMetrics(
        total_examples=len(results), labelled_examples=len(valid),
        failed_examples=sum(bool(r.error_state) for r in results),
        accuracy=(1 - sum(labels) / len(labels)) if labels else None,
        exact_match=sum(exact) / len(exact) if exact else None,
        token_f1=sum(f1s) / len(f1s) if f1s else None,
        auroc=roc_auc(labels, risks), auprc=average_precision(labels, risks),
        ece=ece, brier=brier, negative_log_likelihood=nll,
        mean_latency_ms=statistics.mean(latencies) if latencies else None,
        p50_latency_ms=statistics.median(latencies) if latencies else None,
        p95_latency_ms=sorted(latencies)[min(len(latencies)-1, math.ceil(len(latencies)*.95)-1)] if latencies else None,
        average_input_tokens=statistics.mean([r.raw_generation.token_usage.get("input", 0) for r in results]) if results else None,
        average_output_tokens=statistics.mean([r.raw_generation.token_usage.get("output", 0) for r in results]) if results else None,
        total_estimated_cost=sum(r.raw_generation.estimated_cost or 0 for r in results),
        reliability_bins=bins, risk_coverage=risk_coverage(labels, risks),
        signal_comparison=comparisons, warnings=warnings, metric_status=status,
    )


def dataset_fingerprint(manifest: DatasetManifest) -> str:
    payload = manifest.model_dump(exclude={"build_timestamp"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def configuration_fingerprint(config: ExperimentConfig, dataset_sha256: str) -> str:
    payload = {
        "fingerprint_version": "1",
        "schema_version": SCHEMA_VERSION,
        "metric_versions": METRIC_VERSIONS,
        "dataset_sha256": dataset_sha256,
        "config": config.model_dump(mode="json"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def run_experiment(config: ExperimentConfig, git_commit: str | None = None, provider: Any | None = None) -> ExperimentRecord:
    manifest = load_manifest(config.dataset_name)
    if provider is None:
        from api.providers import provider_for
        provider = provider_for(config.provider, config.model)
    identity = {
        "schema": SCHEMA_VERSION,
        "dataset": [manifest.name, manifest.version, dataset_fingerprint(manifest)],
        "metric_versions": METRIC_VERSIONS,
        "config": config.model_dump(),
    }
    experiment_id = stable_id("exp", identity)
    created = utc_now()
    logger.info("experiment.started id=%s dataset=%s examples=%d", experiment_id, manifest.name, len(manifest.examples))
    examples = [
        example for example in manifest.examples
        if not config.verified_only or manifest.demonstration or example.verification_status == "verified"
    ]
    if config.max_examples:
        examples = examples[:config.max_examples]
    results = [evaluate_example(example, config, provider) for example in examples]
    failed = sum(bool(result.error_state) for result in results)
    state = "failed" if failed == len(results) else ("partially_complete" if failed else "completed")
    record = ExperimentRecord(
        experiment_id=experiment_id, experiment_name=config.experiment_name,
        creation_time=created, completed_time=utc_now(), state=state,
        dataset={
            "name": manifest.name, "version": manifest.version, "size": len(manifest.examples),
            "demonstration": manifest.demonstration, "description": manifest.description,
            "fingerprint": dataset_fingerprint(manifest),
        },
        model=provider.info, config=config, git_commit=git_commit,
        results=results, aggregates=aggregate(results),
        configuration_fingerprint=configuration_fingerprint(config, dataset_fingerprint(manifest)),
        provenance=git_provenance(),
    )
    logger.info("experiment.completed id=%s state=%s failed=%d", experiment_id, state, failed)
    return record


class ExperimentStore:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def migrate(self) -> None:
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)")
            db.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS human_overrides (
                    override_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    example_id TEXT NOT NULL,
                    original_label INTEGER,
                    human_label INTEGER NOT NULL,
                    note TEXT,
                    reviewer TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(human_overrides)").fetchall()}
            if "override_id" not in columns:
                db.execute("ALTER TABLE human_overrides RENAME TO human_overrides_legacy")
                db.execute("""CREATE TABLE human_overrides (
                    override_id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT NOT NULL,
                    example_id TEXT NOT NULL, original_label INTEGER, human_label INTEGER NOT NULL,
                    note TEXT, reviewer TEXT, created_at TEXT NOT NULL)""")
                db.execute("""INSERT INTO human_overrides
                    (experiment_id, example_id, original_label, human_label, note, reviewer, created_at)
                    SELECT experiment_id, example_id, original_label, human_label, note, NULL, created_at
                    FROM human_overrides_legacy""")
                db.execute("DROP TABLE human_overrides_legacy")
            db.execute("CREATE INDEX IF NOT EXISTS idx_experiments_state_created ON experiments(state, created_at)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_overrides_experiment_example ON human_overrides(experiment_id, example_id)")
            db.execute("INSERT OR IGNORE INTO schema_migrations VALUES (?, ?)", (SCHEMA_VERSION, utc_now()))

    def save(self, record: ExperimentRecord) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO experiments VALUES (?, ?, ?, ?, ?, ?)",
                (record.experiment_id, record.schema_version, record.experiment_name, record.state,
                 record.creation_time, record.model_dump_json()),
            )

    def list(self) -> list[ExperimentRecord]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM experiments ORDER BY created_at DESC").fetchall()
        return [ExperimentRecord.model_validate_json(row["payload"]) for row in rows]

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
        return ExperimentRecord.model_validate_json(row["payload"]) if row else None

    def delete(self, experiment_id: str) -> bool:
        with self.connect() as db:
            cursor = db.execute("DELETE FROM experiments WHERE experiment_id=?", (experiment_id,))
        return bool(cursor.rowcount)

    def override(self, experiment_id: str, example_id: str, human_label: bool, note: str | None, reviewer: str | None = None) -> ExperimentRecord:
        record = self.get(experiment_id)
        if not record:
            raise KeyError(experiment_id)
        result = next((item for item in record.results if item.example_id == example_id), None)
        if not result:
            raise KeyError(example_id)
        timestamp = utc_now()
        history = {
            "human_label": human_label, "note": note, "reviewer": reviewer,
            "timestamp": timestamp,
        }
        result.correctness.human_override_history.append(history)
        result.correctness.human_label = human_label
        result.correctness.human_override_at = timestamp
        result.correctness.human_note = note
        result.failure_types = failure_types(result.correctness, result.predicted_risk, result.signals)
        record.aggregates = aggregate(record.results)
        self.save(record)
        with self.connect() as db:
            db.execute(
                """INSERT INTO human_overrides
                   (experiment_id, example_id, original_label, human_label, note, reviewer, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (experiment_id, example_id,
                 None if result.correctness.automated_label is None else int(result.correctness.automated_label),
                 int(human_label), note, reviewer, timestamp),
            )
        return record


def export_experiment(record: ExperimentRecord, format_name: str) -> tuple[str, str]:
    if format_name == "json":
        return record.model_dump_json(indent=2), "application/json"
    if format_name == "csv":
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["example_id", "question", "response", "automated_correct", "human_correct", "effective_correct", "risk", "domain", "failure_types", "provider_error", "experiment_id", "configuration_fingerprint", "dataset_fingerprint", "state"])
        for row in record.results:
            effective_label = (
                row.correctness.human_label
                if row.correctness.human_label is not None
                else row.correctness.correct
            )
            writer.writerow([
                row.example_id, row.question, row.raw_generation.response, row.correctness.automated_label,
                row.correctness.human_label, effective_label, row.predicted_risk, row.domain,
                "|".join(row.failure_types), row.error_state, record.experiment_id,
                record.configuration_fingerprint, record.dataset.get("fingerprint"), record.state,
            ])
        return stream.getvalue(), "text/csv"
    markdown = [
        f"# {record.experiment_name}",
        "", f"- Experiment ID: `{record.experiment_id}`",
        f"- Configuration fingerprint: `{record.configuration_fingerprint or 'unavailable'}`",
        f"- Dataset fingerprint: `{record.dataset.get('fingerprint', 'unavailable')}`",
        f"- Dataset: {record.dataset['name']} {record.dataset['version']}",
        f"- State: {record.state}", f"- Schema: {record.schema_version}", "",
        "## Aggregate metrics", "",
        "```json", json.dumps(record.aggregates.model_dump() if record.aggregates else {}, indent=2), "```", "",
        "## Limitations", "",
        "This report evaluates correlations on a labelled demonstration dataset. It does not independently establish truth.",
    ]
    body = "\n".join(markdown)
    if format_name == "html":
        escaped = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<!doctype html><meta charset='utf-8'><title>{record.experiment_name}</title><pre>{escaped}</pre>", "text/html"
    return body, "text/markdown"


store = ExperimentStore()
