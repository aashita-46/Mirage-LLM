"""Assemble Mirage Reliability Set v1 from pinned, human-authored benchmarks."""
from __future__ import annotations

import csv
import io
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.core import DatasetManifest, utc_now
from api.dataset_tools import dataset_report

ROOT = Path(__file__).resolve().parents[1]
TRUTHFUL_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/013686a06be7a7bde5bf8223943e106c7250123c/TruthfulQA.csv"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def domain_for(category: str) -> str:
    value = category.casefold()
    mapping = {
        "health": "medicine", "nutrition": "medicine", "psychology": "medicine", "medicine": "medicine", "law": "law",
        "finance": "finance", "economics": "finance", "history": "history",
        "science": "science", "weather": "science", "statistics": "science",
        "technology": "technology", "computer": "technology",
        "places": "geography", "location": "geography", "geography": "geography",
        "fiction": "literature", "language": "literature", "misquotation": "literature",
        "proverbs": "literature",
    }
    return next((domain for key, domain in mapping.items() if key in value), "general")


def build() -> DatasetManifest:
    truthful_rows = list(csv.DictReader(io.StringIO(fetch(TRUTHFUL_URL).decode("utf-8"))))
    squad = json.loads(fetch(SQUAD_URL))
    examples = []
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in truthful_rows:
        by_category[row.get("Category", "general")].append(row)
    selected_truthful = []
    depth = 0
    while len(selected_truthful) < 160:
        added = False
        for category in sorted(by_category):
            if depth < len(by_category[category]) and len(selected_truthful) < 160:
                selected_truthful.append(by_category[category][depth])
                added = True
        if not added:
            break
        depth += 1
    for index, row in enumerate(selected_truthful, 1):
        aliases = [value.strip() for value in row.get("Correct Answers", "").split(";") if value.strip()]
        best = row["Best Answer"].strip()
        examples.append({
            "id": f"mrs_truthful_{index:03d}", "question": row["Question"].strip(),
            "reference_answer": best,
            "acceptable_answers": [value for value in aliases if value.casefold() != best.casefold()],
            "answerability": "answerable", "unanswerable": False,
            "domain": domain_for(row.get("Category", "")), "difficulty": "medium",
            "question_type": "truthfulness_stress", "source": "TruthfulQA v1 pinned source",
            "source_name": "TruthfulQA", "source_reference": TRUTHFUL_URL,
            "source_date": "2021", "verification_status": "verified",
            "verification_notes": "Gold answer imported from the human-authored TruthfulQA source; not independently reverified by Mirage.",
            "tags": ["historically_fixed", "truthfulqa", row.get("Category", "general")],
            "metadata": {"original_category": row.get("Category"), "original_source": row.get("Source")},
        })
    impossible = []
    for article in squad["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if qa.get("is_impossible"):
                    impossible.append((article["title"], paragraph["context"], qa))
    for offset, (title, context, qa) in enumerate(impossible[:40], 1):
        examples.append({
            "id": f"mrs_squad_unanswerable_{offset:03d}",
            "question": f"Using only this passage, answer the question. Passage: {context}\nQuestion: {qa['question']}",
            "reference_answer": None, "acceptable_answers": [],
            "answerability": "unanswerable", "unanswerable": True,
            "domain": "general", "difficulty": "hard", "question_type": "context_unanswerable",
            "source": "SQuAD 2.0 development gold annotation",
            "source_name": "SQuAD 2.0", "source_reference": SQUAD_URL,
            "source_date": "2018", "verification_status": "verified",
            "verification_notes": "Unanswerability label imported from the SQuAD 2.0 development gold annotation; not independently reverified by Mirage.",
            "tags": ["unanswerable", "source_grounded", "historically_fixed"],
            "metadata": {"article_title": title, "original_id": qa["id"]},
        })
    return DatasetManifest(
        name="mirage-reliability-set-v1", version="1.0",
        description="A 200-item provenance-aware research set assembled from pinned human-authored TruthfulQA and SQuAD 2.0 gold annotations. Verification means source-gold-labelled, not independently reverified by Mirage.",
        license="Composite: TruthfulQA Apache-2.0; SQuAD 2.0 CC BY-SA 4.0. Preserve source attribution.",
        demonstration=False,
        source_revisions={
            "TruthfulQA": "013686a06be7a7bde5bf8223943e106c7250123c",
            "SQuAD2": "dev-v2.0.json",
        },
        build_script_version="1.1",
        build_timestamp=utc_now(),
        examples=examples,
    )


def main() -> None:
    manifest = build()
    dataset_path = ROOT / "data" / "datasets" / "mirage-reliability-set-v1.json"
    report_path = ROOT / "data" / "validation" / "mirage-reliability-set-v1-report.json"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    report = dataset_report(manifest)
    report["source_urls"] = [TRUTHFUL_URL, SQUAD_URL]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
