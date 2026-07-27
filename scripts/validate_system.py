"""Internal, non-generative validation checks for MirageEval."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.core import (
    METRIC_REGISTRY, average_precision, calibration, dataset_fingerprint,
    list_manifests, risk_coverage, roc_auc, store,
)
from api.dataset_tools import dataset_report

ROOT = Path(__file__).resolve().parents[1]


def metrics() -> list[dict[str, str]]:
    checks = {
        "auroc_one_class_unavailable": roc_auc([1, 1], [.2, .8]) is None,
        "auroc_ties_half_credit": roc_auc([0, 1], [.5, .5]) == .5,
        "auprc_no_positive_unavailable": average_precision([0, 0], [.2, .8]) is None,
        "calibration_missing_unavailable": calibration([], [])[0] is None,
        "risk_coverage_includes_zero_and_full": [row["coverage"] for row in risk_coverage([0, 1], [.1, .9])] == [0, .5, 1],
        "registry_complete": all("version" in item and "range" in item for item in METRIC_REGISTRY.values()),
    }
    return [{"check": name, "status": "pass" if passed else "fail"} for name, passed in checks.items()]


def datasets() -> list[dict[str, str]]:
    rows = []
    for manifest in list_manifests():
        report = dataset_report(manifest)
        passed = not report["duplicate_ids"] and not report["duplicate_questions"] and not report["conflicting_reference_answers"]
        rows.append({"check": f"dataset:{manifest.name}", "status": "pass" if passed else "fail", "fingerprint": dataset_fingerprint(manifest)})
    return rows


def experiments() -> list[dict[str, str]]:
    rows = []
    for record in store.list():
        required = bool(record.configuration_fingerprint and record.dataset.get("fingerprint"))
        raw = all(item.raw_generation.response or item.error_state for item in record.results)
        rows.append({"check": f"experiment:{record.experiment_id}", "status": "pass" if required and raw else "warning", "detail": "fingerprints and raw outputs present" if required and raw else "legacy or incomplete provenance"})
    return rows


def secrets() -> list[dict[str, str]]:
    patterns = [
        re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
        re.compile(r"(?i)(?:api[_-]?key|authorization)[ \t]*[:=][ \t]*['\"]?(?!\[REDACTED\]|$)[A-Za-z0-9_-]{16,}"),
    ]
    findings = []
    excluded = {".git", "node_modules", "dist", ".venv", "__pycache__"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in excluded for part in path.parts) or path.suffix.lower() in {".jpg", ".png", ".db", ".pyc"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in patterns):
            findings.append(str(path.relative_to(ROOT)))
    return [{"check": "secret_scan", "status": "pass" if not findings else "fail", "detail": json.dumps(findings)}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scope", choices=["all", "metrics", "datasets", "experiments", "secrets"], default="all", nargs="?")
    args = parser.parse_args()
    selected = {
        "metrics": metrics, "datasets": datasets, "experiments": experiments, "secrets": secrets,
    }
    rows = []
    for name, function in selected.items():
        if args.scope in {"all", name}:
            rows.extend(function())
    print(json.dumps(rows, indent=2))
    raise SystemExit(1 if any(row["status"] == "fail" for row in rows) else 0)


if __name__ == "__main__":
    main()
