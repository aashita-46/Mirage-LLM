from pathlib import Path

from fastapi.testclient import TestClient

from api.core import (
    CachedDemoProvider,
    DatasetExample,
    ExperimentConfig,
    ExperimentStore,
    average_precision,
    aggregate,
    calibration,
    correctness,
    numeric_match,
    risk_coverage,
    roc_auc,
    run_experiment,
    semantic_clusters,
    token_f1,
    export_experiment,
)
from api.index import app

client = TestClient(app)


def test_health_exposes_schema_version():
    payload = client.get("/api/v1/health").json()
    assert payload["status"] == "ok"
    assert payload["schema_version"] == "2.0"


def test_normalisation_and_alias_correctness():
    example = DatasetExample(
        id="x", question="Capital?", reference_answer="Canberra",
        acceptable_answers=["Canberra, ACT"], source="test",
    )
    result = correctness(example, "Canberra.", ExperimentConfig().evaluator)
    assert result.correct is True
    assert result.method == "deterministic_composite"


def test_token_f1():
    assert token_f1("Ernest Hemingway wrote it", "Ernest Hemingway") > 0.6
    assert token_f1("London", "Canberra") == 0


def test_numerical_tolerance():
    assert numeric_match("The result is 100.5", "100", 0.01) is True
    assert numeric_match("The result is 105", "100", 0.01) is False


def test_semantic_cluster_schema_and_method():
    clusters = semantic_clusters(["Canberra is the capital.", "The capital is Canberra.", "Sydney."])
    assert sum(cluster.size for cluster in clusters) == 3
    assert all(cluster.evaluator_method == "lexical_fallback_jaccard" for cluster in clusters)


def test_auroc_and_auprc_handling():
    assert roc_auc([0, 0], [.1, .2]) is None
    assert roc_auc([0, 1], [.1, .9]) == 1
    assert average_precision([0, 1], [.1, .9]) == 1


def test_calibration_metrics():
    ece, brier, nll, bins = calibration([0, 1], [.1, .9], bins=2)
    assert round(ece, 3) == .1
    assert round(brier, 3) == .01
    assert nll is not None and bins


def test_risk_coverage_is_monotonic():
    rows = risk_coverage([0, 1, 0], [.1, .9, .2])
    assert [row["coverage"] for row in rows] == sorted(row["coverage"] for row in rows)
    assert rows[0]["selective_accuracy"] == 1


def test_provider_does_not_fabricate_logprobs():
    record = run_experiment(ExperimentConfig())
    assert record.results
    assert all(result.raw_generation.token_logprobs is None for result in record.results)
    assert all(result.signals.token_uncertainty is None for result in record.results)


def test_experiment_is_deterministic_and_traceable():
    first = run_experiment(ExperimentConfig())
    second = run_experiment(ExperimentConfig())
    assert first.experiment_id == second.experiment_id
    assert first.schema_version == "2.0"
    assert all(result.trace["raw_output_sha256"] for result in first.results)


def test_experiment_persistence(tmp_path: Path):
    store = ExperimentStore(tmp_path / "test.db")
    record = run_experiment(ExperimentConfig())
    store.save(record)
    loaded = store.get(record.experiment_id)
    assert loaded is not None and loaded.experiment_id == record.experiment_id


def test_human_override_preserves_original(tmp_path: Path):
    store = ExperimentStore(tmp_path / "test.db")
    record = run_experiment(ExperimentConfig())
    store.save(record)
    target = record.results[0]
    original = target.correctness.automated_label
    updated = store.override(record.experiment_id, target.example_id, not bool(original), "manual review")
    changed = next(x for x in updated.results if x.example_id == target.example_id)
    assert changed.correctness.automated_label == original
    assert changed.correctness.human_label == (not bool(original))
    semantic = next(x for x in updated.aggregates.signal_comparison if x["signal"] == "semantic_entropy")
    rows = [x for x in updated.results if x.signals.semantic_entropy is not None]
    labels = [
        int(not bool(x.correctness.human_label if x.correctness.human_label is not None else x.correctness.correct))
        for x in rows
    ]
    scores = [float(x.signals.semantic_entropy) for x in rows]
    assert semantic["auroc"] == roc_auc(labels, scores)


def test_csv_export_uses_effective_human_label():
    record = run_experiment(ExperimentConfig())
    target = record.results[0]
    target.correctness.human_label = not bool(target.correctness.correct)
    record.aggregates = aggregate(record.results)
    body, media_type = export_experiment(record, "csv")
    target_row = next(line for line in body.splitlines() if line.startswith(f"{target.example_id},"))
    assert str(target.correctness.human_label) in target_row
    assert media_type == "text/csv"


def test_dataset_validation_reports_bad_rows():
    body = {"filename": "bad.jsonl", "content": '{"id":"x","question":"no","source":"test"}\nnot-json'}
    response = client.post("/api/v1/datasets/validate", json=body)
    assert response.status_code == 422


def test_provider_failure_does_not_become_valid_result():
    example = DatasetExample(id="missing", question="What is missing?", reference_answer="x", source="test")
    generated = CachedDemoProvider().generate(example, ExperimentConfig().sampling)
    assert generated.error
    assert generated.response == ""


def test_experiment_api_and_export():
    response = client.post("/api/v1/experiments", json=ExperimentConfig().model_dump())
    assert response.status_code == 200
    record = response.json()
    assert record["aggregates"]["auprc"] is not None
    exported = client.get(f"/api/v1/experiments/{record['experiment_id']}/export?format=csv")
    assert exported.status_code == 200
    assert "example_id,question" in exported.text


def test_live_model_discovery_and_dataset_report():
    models = client.get("/api/v1/models")
    assert models.status_code == 200
    assert any(item["provider"] == "ollama" for item in models.json()["models"])
    report = client.get("/api/v1/datasets/mirage-reliability-set-v1/report")
    assert report.status_code == 200
    assert report.json()["verified_examples"] == 200


def test_findings_are_absent_without_official_live_group():
    payload = client.get("/api/v1/findings/research-v1").json()
    assert payload["available"] is False
    assert payload["experiments"] == []
