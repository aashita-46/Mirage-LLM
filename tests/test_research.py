import json
from pathlib import Path

import httpx
import pytest

from api.core import DatasetExample, ExperimentConfig, ExperimentStore, GenerationRecord, ModelInfo, ProviderCapabilities, average_precision, roc_auc
from api.core import DatasetManifest, load_manifest
from api.dataset_tools import dataset_report
from api.providers import OllamaProvider, OpenAICompatibleProvider, ProviderError, redact
from api.semantics import cluster_responses
from api.statistics import (
    apply_isotonic,
    apply_platt,
    bootstrap_ci,
    fit_isotonic,
    fit_platt,
    paired_bootstrap_difference,
    stratified_split,
)
from api.research import experiment_identity, run_resumable
from scripts.generate_research_report import generated


class FakeClient:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.requests = []

    def get(self, url):
        if self.error:
            raise self.error
        return httpx.Response(200, json={"models": [{"name": "test:latest"}]}, request=httpx.Request("GET", url))

    def post(self, url, json=None, headers=None):
        self.requests.append({"url": url, "json": json, "headers": headers})
        if self.error:
            raise self.error
        payload = self.responses.pop(0)
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))


def example():
    return DatasetExample(id="x", question="Who wrote the novel?", reference_answer="Hemingway", source="test")


def test_ollama_unavailable_and_missing_model():
    unavailable = OllamaProvider("x", client=FakeClient(error=httpx.ConnectError("offline")))
    with pytest.raises(ProviderError, match="unavailable"):
        unavailable.available_models()
    missing = OllamaProvider("missing", client=FakeClient())
    with pytest.raises(ProviderError, match="not installed"):
        missing.validate()


def test_successful_ollama_response_usage_and_capabilities():
    replies = [
        {"message": {"content": "Hemingway"}, "prompt_eval_count": 4, "eval_count": 2},
        {"message": {"content": "Ernest Hemingway"}, "prompt_eval_count": 4, "eval_count": 3},
        {"message": {"content": json.dumps({"verdict": "supported", "confidence": .9, "reason": "known", "claims": []})}},
    ]
    provider = OllamaProvider("test:latest", client=FakeClient(replies))
    config = ExperimentConfig().sampling.model_copy(update={"semantic_samples": 2})
    record = provider.generate(example(), config)
    assert record.response == "Hemingway"
    assert record.token_usage == {"input": 8, "output": 5}
    assert record.token_logprobs is None
    assert provider.info.capabilities.supports_structured_output


def test_openai_compatible_parsing_and_key_not_persisted(monkeypatch):
    raw = {
        "choices": [{"message": {"content": "Hemingway"}, "logprobs": {"content": [{"logprob": -0.1}]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    provider = OpenAICompatibleProvider(
        model="test", base_url="https://example.test/v1", api_key="secret-key",
        supports_logprobs=True, client=FakeClient([raw]), retries=0,
    )
    record = provider.generate(example(), ExperimentConfig().sampling)
    assert record.token_logprobs == [-0.1]
    assert "secret-key" not in json.dumps(record.model_dump())
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "secret-key")
    assert "secret-key" not in redact("failed secret-key")


def test_embedding_clustering_equivalence_and_separation():
    mapping = {
        "Ernest Hemingway wrote the novel.": [1.0, 0.0, 0.0],
        "The author was Hemingway.": [.99, .01, 0.0],
        "The answer is 43.": [0.0, 1.0, 0.0],
    }
    texts = list(mapping)
    clusters, metadata = cluster_responses(
        texts, method="embedding", threshold=.9,
        embedder=lambda values: [mapping[value] for value in values],
    )
    assert [cluster.size for cluster in clusters] == [2, 1]
    assert metadata["method"].startswith("embedding_cosine")


def test_embedding_contradiction_and_numeric_guard():
    texts = ["The treatment improves survival.", "The treatment does not improve survival.", "The answer is 42.", "The answer is 43."]
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    clusters, _ = cluster_responses(
        texts, method="embedding", threshold=.8,
        embedder=lambda values: vectors,
    )
    assert len(clusters) == 4


def test_embedding_threshold_is_deterministic():
    texts = ["a response", "another response"]
    embed = lambda values: [[1, 0], [.8, .6]]
    low, _ = cluster_responses(texts, "embedding", .79, embedder=embed)
    high, _ = cluster_responses(texts, "embedding", .81, embedder=embed)
    assert len(low) == 1 and len(high) == 2


def test_bootstrap_and_paired_comparison_are_reproducible():
    rows = [(0, .1), (0, .2), (1, .8), (1, .9)]
    first = bootstrap_ci(rows, roc_auc, resamples=200, seed=7)
    second = bootstrap_ci(rows, roc_auc, resamples=200, seed=7)
    assert first == second and first["estimate"] == 1
    comparison = paired_bootstrap_difference(
        [0, 0, 1, 1], [.1, .2, .8, .9], [.4, .3, .6, .5],
        roc_auc, resamples=200, seed=7,
    )
    assert comparison["difference"] is not None


def test_bootstrap_handles_one_class_samples():
    result = bootstrap_ci([(0, .1), (1, .9)], roc_auc, resamples=100, seed=1)
    assert 0 < result["valid_resamples"] < 100


def test_calibration_split_and_models_do_not_leak():
    labels = [0, 0, 0, 1, 1, 1]
    train, test = stratified_split(labels, test_fraction=.33, seed=3)
    assert set(train).isdisjoint(test)
    assert set(labels[i] for i in train) == {0, 1}
    platt = fit_platt([.1, .2, .8, .9], [0, 0, 1, 1])
    assert all(0 <= value <= 1 for value in apply_platt([.3, .7], platt))
    isotonic = fit_isotonic([.1, .2, .8, .9], [0, 0, 1, 1])
    calibrated = apply_isotonic([.15, .85], isotonic)
    assert calibrated[0] <= calibrated[1]


def test_average_precision_ties_are_order_independent():
    assert average_precision([1, 0], [.5, .5]) == average_precision([0, 1], [.5, .5]) == .5


def test_reliability_set_has_verified_provenance_and_statistics():
    manifest = load_manifest("mirage-reliability-set-v1")
    report = dataset_report(manifest)
    assert report["total_examples"] == report["verified_examples"] == 200
    assert report["missing_source_count"] == 0
    assert report["duplicate_ids"] == []
    assert len(report["domain_distribution"]) >= 6


def test_dataset_report_detects_duplicates_conflicts_and_pending():
    manifest = DatasetManifest(
        name="test", version="1", description="test", demonstration=False,
        examples=[
            DatasetExample(id="same", question="What is X?", reference_answer="A", source="one"),
            DatasetExample(id="same", question="What is X?", reference_answer="B", source="two"),
        ],
    )
    report = dataset_report(manifest)
    assert report["duplicate_ids"] == ["same"]
    assert report["duplicate_questions"]
    assert report["conflicting_reference_answers"]
    assert report["verified_examples"] == 0


class CountingProvider:
    def __init__(self):
        self.calls = 0
        self.info = ModelInfo(
            provider="test", model="deterministic", mode="test_mock",
            capabilities=ProviderCapabilities(),
        )

    def generate(self, item, config):
        self.calls += 1
        return GenerationRecord(
            response=item.reference_answer or "cannot be determined",
            sampled_responses=[item.reference_answer or "cannot be determined"] * config.semantic_samples,
            latency_ms=1,
        )


def test_resumable_experiment_skips_completed_examples(tmp_path: Path):
    store = ExperimentStore(tmp_path / "research.db")
    config = ExperimentConfig(max_examples=2)
    provider = CountingProvider()
    first = run_resumable(config, store, provider=provider)
    assert provider.calls == 2 and first.state == "completed"
    second = run_resumable(config, store, provider=provider, resume=True)
    assert provider.calls == 2 and len(second.results) == 2


def test_configuration_change_produces_distinct_identity():
    assert experiment_identity(ExperimentConfig(model="a"), "1") != experiment_identity(ExperimentConfig(model="b"), "1")


def test_report_generator_does_not_invent_missing_findings():
    results, models = generated("group-that-does-not-exist")
    assert "No completed live" in results
    assert "No completed live" in models
    assert "0.0" not in results
