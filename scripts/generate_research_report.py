from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.core import store

ROOT = Path(__file__).resolve().parents[1]


def fmt(value):
    return "N/A" if value is None else f"{value:.3f}"


def generated(group: str) -> tuple[str, str]:
    records = [
        record for record in store.list()
        if record.config.experiment_group == group and record.model.mode == "live"
        and record.state == "completed" and not record.dataset.get("demonstration")
    ]
    if not records:
        return f"No completed live results are eligible for `{group}`. This placeholder is intentionally retained.", f"No completed live model runs are eligible for `{group}`."
    if group == "research-v1":
        models = {record.model.model for record in records}
        sufficiently_sized = [record for record in records if len(record.results) >= 200]
        if len(models) < 3 or len(sufficiently_sized) < 3:
            return (
                "Official conclusion withheld: research-v1 requires three completed live model runs "
                "with at least 200 eligible examples each.",
                "Official model comparison unavailable because the stored experiment group does not meet eligibility requirements.",
            )
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip() or "unavailable"
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    if dirty:
        commit += "-dirty"
    reproduce_config = "config/live-smoke-v1.json" if group == "live-smoke-v1" else "config/research-experiment-v1.json"
    lines = [
        f"Experiment group: `{group}`. These results must not be interpreted beyond the recorded sample size.",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Git commit: `{commit}`",
        "",
        "| Experiment | Model | N | Accuracy | AUROC | AUPRC | ECE | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        metrics = record.aggregates
        lines.append(
            f"| `{record.experiment_id}` | {record.model.model} | {len(record.results)} | "
            f"{fmt(metrics.accuracy if metrics else None)} | {fmt(metrics.auroc if metrics else None)} | "
            f"{fmt(metrics.auprc if metrics else None)} | {fmt(metrics.ece if metrics else None)} | "
            f"{fmt(metrics.brier if metrics else None)} |"
        )
    lines.extend([
        "", f"Dataset: `{records[0].dataset['name']}` v{records[0].dataset['version']}",
        f"Schema: `{records[0].schema_version}`",
        f"Metric versions: `{json.dumps(records[0].metric_versions, sort_keys=True)}`",
        "", "Reproduce:", "",
        "```bash", f"python scripts/evaluate.py --config {reproduce_config} --resume", "```",
    ])
    models = "\n".join(
        f"- `{record.model.provider}/{record.model.model}` — experiment `{record.experiment_id}`, "
        f"temperature {record.config.sampling.temperature}, samples {record.config.sampling.semantic_samples}"
        for record in records
    )
    return "\n".join(lines), models


def replace_block(text: str, marker: str, content: str) -> str:
    start = f"<!-- {marker} -->"
    end = f"<!-- END_{marker} -->"
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return f"{before}{start}\n\n{content}\n\n{end}{after}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-group", default="research-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "mirage-reliability-study-v1.md")
    args = parser.parse_args()
    text = args.output.read_text(encoding="utf-8")
    results, models = generated(args.experiment_group)
    text = replace_block(text, "GENERATED_FROM_EXPERIMENT", results)
    text = replace_block(text, "GENERATED_MODEL_CONFIGURATIONS", models)
    args.output.write_text(text, encoding="utf-8", newline="\n")
    print(args.output)


if __name__ == "__main__":
    main()
