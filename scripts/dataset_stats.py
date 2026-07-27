from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.core import load_manifest
from api.dataset_tools import dataset_report

parser = argparse.ArgumentParser()
parser.add_argument("dataset", nargs="?", default="mirage-reliability-set-v1")
args = parser.parse_args()
print(json.dumps(dataset_report(load_manifest(args.dataset)), indent=2))
