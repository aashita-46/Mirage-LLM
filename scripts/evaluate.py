"""Provider-independent Mirage evaluation CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.core import ExperimentConfig, export_experiment, run_experiment, store
from api.research import run_resumable


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible Mirage evaluation.")
    parser.add_argument("--config", type=Path, help="Optional ExperimentConfig JSON file.")
    parser.add_argument("--name", default="CLI starter reliability study")
    parser.add_argument("--samples", type=int, choices=range(2, 11), default=6)
    parser.add_argument("--export", choices=["json", "csv", "markdown", "html"], default="json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--experiment-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.experiment_id:
        existing = store.get(args.experiment_id)
        if not existing:
            raise SystemExit(f"Experiment {args.experiment_id!r} does not exist.")
        config = existing.config
    elif args.config:
        raw_config = json.loads(args.config.read_text(encoding="utf-8"))
        if "models" in raw_config:
            base = {key: value for key, value in raw_config.items() if key != "models"}
            summaries = []
            for model_entry in raw_config["models"]:
                config = ExperimentConfig.model_validate({**base, **model_entry})
                if args.dry_run:
                    summaries.append({"provider": config.provider, "model": config.model, "config": config.model_dump()})
                    continue
                record = run_resumable(config, store, resume=args.resume, rerun_failed=args.rerun_failed)
                summaries.append({"experiment_id": record.experiment_id, "model": config.model, "state": record.state})
            print(json.dumps({"experiment_group": raw_config.get("experiment_group"), "runs": summaries}, indent=2))
            return
        config = ExperimentConfig.model_validate(raw_config)
    else:
        config = ExperimentConfig(experiment_name=args.name)
        config.sampling.semantic_samples = args.samples

    if args.dry_run:
        print(config.model_dump_json(indent=2))
        return
    record = run_resumable(config, store, resume=args.resume, rerun_failed=args.rerun_failed)
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
