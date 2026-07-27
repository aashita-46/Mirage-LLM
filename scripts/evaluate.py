"""Provider-independent Mirage evaluation CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.core import ExperimentConfig, export_experiment, run_experiment, store


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible Mirage evaluation.")
    parser.add_argument("--config", type=Path, help="Optional ExperimentConfig JSON file.")
    parser.add_argument("--name", default="CLI starter reliability study")
    parser.add_argument("--samples", type=int, choices=range(2, 11), default=6)
    parser.add_argument("--export", choices=["json", "csv", "markdown", "html"], default="json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.config:
        config = ExperimentConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    else:
        config = ExperimentConfig(experiment_name=args.name)
        config.sampling.semantic_samples = args.samples

    record = run_experiment(config)
    store.save(record)
    body, _ = export_experiment(record, args.export)
    destination = args.output or Path("data") / "exports" / f"{record.experiment_id}.{args.export if args.export != 'markdown' else 'md'}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body, encoding="utf-8", newline="\n")
    print(json.dumps({
        "experiment_id": record.experiment_id,
        "state": record.state,
        "examples": len(record.results),
        "output": str(destination),
        "metrics": record.aggregates.model_dump() if record.aggregates else None,
    }, indent=2))


if __name__ == "__main__":
    main()
