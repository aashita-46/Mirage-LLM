from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.core import store
from api.semantics import threshold_sensitivity

parser = argparse.ArgumentParser()
parser.add_argument("--experiment-group", default="research-v1")
parser.add_argument("--thresholds", default="0.70,0.75,0.78,0.80,0.85")
args = parser.parse_args()
records = [record for record in store.list() if record.config.experiment_group == args.experiment_group]
examples = [
    (result.raw_generation.sampled_responses, int(not (result.correctness.human_label if result.correctness.human_label is not None else result.correctness.correct)))
    for record in records for result in record.results
    if result.correctness.correct is not None and len(result.raw_generation.sampled_responses) >= 2
]
if not examples:
    print(json.dumps({"available": False, "reason": "No eligible stored samples for this group."}, indent=2))
else:
    print(json.dumps(threshold_sensitivity(examples, [float(value) for value in args.thresholds.split(",")]), indent=2))
