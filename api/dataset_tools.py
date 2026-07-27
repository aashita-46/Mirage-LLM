"""Dataset statistics and integrity validation."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from api.core import DatasetManifest, dataset_fingerprint, normalise_text
from api.semantics import cluster_responses


def dataset_report(manifest: DatasetManifest, similarity_threshold: float = .92) -> dict[str, Any]:
    examples = manifest.examples
    ids = Counter(item.id for item in examples)
    normalised_questions = defaultdict(list)
    answers = Counter()
    for item in examples:
        normalised_questions[normalise_text(item.question)].append(item.id)
        if item.reference_answer:
            answers[normalise_text(item.reference_answer)] += 1
    duplicate_questions = [
        {"question": question, "ids": item_ids}
        for question, item_ids in normalised_questions.items() if len(item_ids) > 1
    ]
    conflicting = []
    for duplicate in duplicate_questions:
        refs = {
            normalise_text(item.reference_answer or "")
            for item in examples
            if normalise_text(item.question) == duplicate["question"]
        }
        if len(refs) > 1:
            conflicting.append({**duplicate, "references": sorted(refs)})
    leaked = [
        item.id for item in examples
        if item.reference_answer
        and len(normalise_text(item.reference_answer)) > 3
        and normalise_text(item.reference_answer) in normalise_text(item.question)
    ]
    highly_similar: list[dict[str, Any]] = []
    if len(examples) <= 500:
        # Lexical prefilter only; this report does not claim semantic deduplication.
        _, metadata = cluster_responses(
            [item.question for item in examples], "lexical_fallback", similarity_threshold,
        )
        matrix = metadata["similarity_matrix"]
        highly_similar = [
            {"first": examples[i].id, "second": examples[j].id, "similarity": matrix[i][j]}
            for i in range(len(examples)) for j in range(i + 1, len(examples))
            if matrix[i][j] >= similarity_threshold
        ]
    verified = [item for item in examples if item.verification_status == "verified"]
    return {
        "dataset": manifest.name, "version": manifest.version,
        "dataset_fingerprint": dataset_fingerprint(manifest),
        "source_revisions": manifest.source_revisions,
        "build_script_version": manifest.build_script_version,
        "build_timestamp": manifest.build_timestamp,
        "total_examples": len(examples), "verified_examples": len(verified),
        "pending_examples": len(examples) - len(verified),
        "domain_distribution": dict(Counter(item.domain for item in examples)),
        "difficulty_distribution": dict(Counter(item.difficulty for item in examples)),
        "answerability_distribution": dict(Counter(item.answerability for item in examples)),
        "question_type_distribution": dict(Counter(item.question_type for item in examples)),
        "missing_source_count": sum(not (item.source_name and item.source_reference) for item in examples),
        "duplicate_ids": sorted(item_id for item_id, count in ids.items() if count > 1),
        "duplicate_questions": duplicate_questions,
        "highly_similar_questions": highly_similar,
        "conflicting_reference_answers": conflicting,
        "duplicate_reference_answer_count": sum(count - 1 for count in answers.values() if count > 1),
        "answer_leakage_ids": leaked,
        "official_metric_eligible": len(verified),
        "warnings": _balance_warnings(examples),
    }


def _balance_warnings(examples: list[Any]) -> list[str]:
    if not examples:
        return ["Dataset is empty."]
    distribution = Counter(item.answerability for item in examples)
    ratios = {key: value / len(examples) for key, value in distribution.items()}
    warnings = []
    if not .60 <= ratios.get("answerable", 0) <= .70:
        warnings.append("Answerable share is outside the 60–70% design target.")
    if not .10 <= ratios.get("unanswerable", 0) <= .15:
        warnings.append("Unanswerable share is outside the 10–15% design target.")
    if not .10 <= ratios.get("false_premise", 0) <= .15:
        warnings.append("False-premise share is outside the 10–15% design target.")
    if not .05 <= ratios.get("ambiguous", 0) <= .10:
        warnings.append("Ambiguous share is outside the 5–10% design target.")
    return warnings
