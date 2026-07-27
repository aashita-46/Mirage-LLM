"""Resumable experiment execution and measured research analysis."""
from __future__ import annotations

from typing import Any

from api.core import (
    METRIC_VERSIONS,
    SCHEMA_VERSION,
    ExperimentConfig,
    ExperimentRecord,
    ExperimentStore,
    aggregate,
    calibration,
    evaluate_example,
    load_manifest,
    roc_auc,
    average_precision,
    configuration_fingerprint,
    dataset_fingerprint,
    git_provenance,
    stable_id,
    utc_now,
)
from api.providers import BaseProvider, provider_for
from api.statistics import (
    apply_isotonic,
    apply_platt,
    bootstrap_ci,
    bootstrap_statistic,
    fit_isotonic,
    fit_platt,
    paired_bootstrap_difference,
    stratified_split,
)


def experiment_identity(config: ExperimentConfig, dataset_version: str) -> str:
    manifest = load_manifest(config.dataset_name)
    fingerprint = dataset_fingerprint(manifest)
    return stable_id("exp", {"fingerprint": configuration_fingerprint(config, fingerprint)})


def configuration_diff(stored: ExperimentConfig, requested: ExperimentConfig) -> list[dict[str, Any]]:
    def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
        if isinstance(value, dict):
            return {key: nested for name, item in value.items() for key, nested in flatten(item, f"{prefix}.{name}" if prefix else name).items()}
        return {prefix: value}
    left, right = flatten(stored.model_dump(mode="json")), flatten(requested.model_dump(mode="json"))
    return [{"field": key, "stored": left.get(key), "requested": right.get(key)} for key in sorted(left.keys() | right.keys()) if left.get(key) != right.get(key)]


def run_resumable(
    config: ExperimentConfig,
    store: ExperimentStore,
    provider: BaseProvider | None = None,
    resume: bool = False,
    rerun_failed: bool = False,
    git_commit: str | None = None,
) -> ExperimentRecord:
    manifest = load_manifest(config.dataset_name)
    experiment_id = experiment_identity(config, manifest.version)
    existing = store.get(experiment_id) if resume or rerun_failed else None
    if existing and existing.config.model_dump() != config.model_dump():
        differences = configuration_diff(existing.config, config)
        details = "; ".join(f"{item['field']}: {item['stored']!r} -> {item['requested']!r}" for item in differences)
        raise ValueError(f"Stored experiment configuration is stale: {details}")
    active_provider = provider or provider_for(config.provider, config.model)
    examples = [
        item for item in manifest.examples
        if not config.verified_only or manifest.demonstration or item.verification_status == "verified"
    ]
    if config.max_examples:
        examples = examples[:config.max_examples]
    completed = {item.example_id: item for item in existing.results} if existing else {}
    if rerun_failed:
        completed = {key: value for key, value in completed.items() if not value.error_state}
    record = existing or ExperimentRecord(
        experiment_id=experiment_id, experiment_name=config.experiment_name,
        creation_time=utc_now(), state="running",
        dataset={
            "name": manifest.name, "version": manifest.version, "size": len(examples),
            "demonstration": manifest.demonstration, "description": manifest.description,
            "fingerprint": dataset_fingerprint(manifest),
        },
        model=active_provider.info, config=config, git_commit=git_commit, results=[],
        metric_versions=dict(METRIC_VERSIONS),
        configuration_fingerprint=configuration_fingerprint(config, dataset_fingerprint(manifest)),
        provenance=git_provenance(),
    )
    record.state = "running"
    record.results = list(completed.values())
    store.save(record)
    for example in examples:
        if example.id in completed:
            continue
        result = evaluate_example(example, config, active_provider)
        record.results.append(result)
        record.aggregates = aggregate(record.results)
        record.state = "partially_complete"
        store.save(record)
    failures = sum(bool(item.error_state) for item in record.results)
    record.state = "failed" if failures == len(record.results) else ("partially_complete" if failures else "completed")
    record.completed_time = utc_now()
    record.aggregates = aggregate(record.results)
    record.research_analysis = analyse_research(record, config.bootstrap_resamples, config.bootstrap_seed)
    store.save(record)
    return record


def _effective_label(result: Any) -> int:
    correct = result.correctness.human_label if result.correctness.human_label is not None else result.correctness.correct
    return int(not bool(correct))


def _coverage_at_target(record: ExperimentRecord, target: float) -> float | None:
    rows = record.aggregates.risk_coverage if record.aggregates else []
    valid = [row["coverage"] for row in rows if row["selective_accuracy"] >= target]
    return max(valid) if valid else None


def analyse_research(record: ExperimentRecord, resamples: int = 1000, seed: int = 42) -> dict[str, Any]:
    valid = [item for item in record.results if item.correctness.correct is not None and item.predicted_risk is not None]
    labels = [_effective_label(item) for item in valid]
    risks = [float(item.predicted_risk) for item in valid]
    output: dict[str, Any] = {
        "bootstrap": {"resamples": resamples, "seed": seed, "confidence": .95},
        "confidence_intervals": {},
        "paired_signal_comparisons": [],
        "coverage_at_90_selective_accuracy": _coverage_at_target(record, .90),
        "coverage_at_95_selective_accuracy": _coverage_at_target(record, .95),
        "calibration": {},
    }
    if len(valid) < 30 or resamples <= 0:
        output["warning"] = "Confidence intervals require at least 30 labelled results and positive resample count."
        return output
    output["confidence_intervals"]["auroc"] = bootstrap_ci(list(zip(labels, risks)), roc_auc, resamples, seed)
    output["confidence_intervals"]["auprc"] = bootstrap_ci(list(zip(labels, risks)), average_precision, resamples, seed)
    paired = list(zip(labels, risks))
    output["confidence_intervals"]["accuracy"] = bootstrap_statistic(
        paired, lambda rows: 1 - sum(label for label, _ in rows) / len(rows), resamples, seed,
    )
    output["confidence_intervals"]["ece"] = bootstrap_statistic(
        paired, lambda rows: calibration([x[0] for x in rows], [x[1] for x in rows])[0], resamples, seed,
    )
    output["confidence_intervals"]["brier"] = bootstrap_statistic(
        paired, lambda rows: calibration([x[0] for x in rows], [x[1] for x in rows])[1], resamples, seed,
    )
    for coverage in (.5, .7, .8, .9):
        output["confidence_intervals"][f"selective_accuracy_at_{int(coverage*100)}_coverage"] = bootstrap_statistic(
            paired,
            lambda rows, target=coverage: _selective_accuracy(rows, target),
            resamples, seed,
        )
    score_fields = {
        "semantic_entropy": lambda item: item.signals.semantic_entropy,
        "response_inconsistency": lambda item: None if item.signals.response_consistency is None else 1 - item.signals.response_consistency,
        "self_verification_uncertainty": lambda item: item.signals.self_verification_uncertainty,
    }
    aligned: dict[str, list[float]] = {}
    for name, getter in score_fields.items():
        values = [getter(item) for item in valid]
        if all(value is not None for value in values):
            aligned[name] = [float(value) for value in values]
    names = sorted(aligned)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            output["paired_signal_comparisons"].append({
                "first": first, "second": second, "metric": "auroc",
                **paired_bootstrap_difference(labels, aligned[first], aligned[second], roc_auc, resamples, seed),
            })
    train, test = stratified_split(labels, .3, seed)
    if train and test and len({labels[i] for i in train}) == 2:
        raw_test = [risks[i] for i in test]
        test_labels = [labels[i] for i in test]
        pre_ece, pre_brier, pre_nll, _ = calibration(test_labels, raw_test)
        platt = fit_platt([risks[i] for i in train], [labels[i] for i in train])
        iso = fit_isotonic([risks[i] for i in train], [labels[i] for i in train])
        output["calibration"] = {
            "split": {"seed": seed, "train_indices": train, "evaluation_indices": test},
            "raw": {"ece": pre_ece, "brier": pre_brier, "nll": pre_nll},
            "platt": _calibration_result(test_labels, apply_platt(raw_test, platt), platt),
            "isotonic": _calibration_result(test_labels, apply_isotonic(raw_test, iso), iso),
        }
    return output


def _selective_accuracy(rows: list[tuple[int, float]], coverage: float) -> float | None:
    accepted = max(1, round(len(rows) * coverage))
    subset = sorted(rows, key=lambda item: item[1])[:accepted]
    return 1 - sum(label for label, _ in subset) / len(subset)


def _calibration_result(labels: list[int], scores: list[float], parameters: dict[str, Any]) -> dict[str, Any]:
    ece, brier, nll, bins = calibration(labels, scores)
    return {"parameters": parameters, "ece": ece, "brier": brier, "nll": nll, "reliability_bins": bins}


def findings_for_group(records: list[ExperimentRecord], group: str) -> dict[str, Any]:
    eligible = [
        record for record in records
        if record.config.experiment_group == group and record.model.mode == "live"
        and record.aggregates and record.state == "completed" and not record.dataset.get("demonstration")
    ]
    if not eligible:
        return {
            "experiment_group": group, "available": False,
            "reason": "No completed, non-demonstration live experiment records exist for this group.",
            "requirements": ["state=completed", "model.mode=live", "dataset.demonstration=false", "aggregate metrics present"],
            "experiments": [],
        }
    strongest = max(eligible, key=lambda item: item.aggregates.accuracy or -1)
    calibrated = min(eligible, key=lambda item: item.aggregates.brier if item.aggregates.brier is not None else float("inf"))
    signal_rows = []
    for record in eligible:
        for signal in record.aggregates.signal_comparison:
            signal_rows.append({**signal, "model": record.model.model, "experiment_id": record.experiment_id})
    best_signal = max(
        (item for item in signal_rows if item.get("auroc") is not None),
        key=lambda item: item["auroc"], default=None,
    )
    return {
        "experiment_group": group, "available": True,
        "result_status": "official" if group == "research-v1" else "preliminary",
        "dataset": eligible[0].dataset, "schema_version": eligible[0].schema_version,
        "experiment_ids": [item.experiment_id for item in eligible],
        "executive_findings": {
            "strongest_model": {"model": strongest.model.model, "accuracy": strongest.aggregates.accuracy},
            "best_calibrated_model": {"model": calibrated.model.model, "brier": calibrated.aggregates.brier},
            "best_signal": best_signal,
            "major_limitation": "Results are conditional on source-gold labels, local models, prompts, sampling, and evaluator behavior.",
        },
        "experiments": [
            {
                "experiment_id": item.experiment_id, "model": item.model.model,
                "state": item.state, "metrics": item.aggregates.model_dump(),
                "research_analysis": item.research_analysis,
            }
            for item in eligible
        ],
        "signal_leaderboard": sorted(
            signal_rows, key=lambda item: item.get("auroc") if item.get("auroc") is not None else -1, reverse=True,
        ),
    }
