"""Semantic-equivalence clustering with explicit method metadata."""
from __future__ import annotations

import math
import re
import hashlib
from collections.abc import Callable
from typing import Any

import httpx

from api.core import SemanticCluster, lexical_similarity, normalise_text

Embedder = Callable[[list[str]], list[list[float]]]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    left = math.sqrt(sum(x * x for x in a))
    right = math.sqrt(sum(y * y for y in b))
    return dot / (left * right) if left and right else 0.0


def ollama_embeddings(
    texts: list[str],
    model: str = "all-minilm",
    base_url: str = "http://127.0.0.1:11434",
    timeout: float = 120,
) -> list[list[float]]:
    response = httpx.post(
        f"{base_url.rstrip('/')}/api/embed",
        json={"model": model, "input": texts}, timeout=timeout,
    )
    response.raise_for_status()
    vectors = response.json().get("embeddings", [])
    if len(vectors) != len(texts):
        raise RuntimeError("Embedding provider returned an unexpected vector count.")
    return vectors


def _numeric_tokens(text: str) -> set[tuple[float, str]]:
    pattern = r"(-?(?:\d+(?:,\d{3})*|\d*\.\d+)(?:[eE][+-]?\d+)?)\s*(%|km|m|cm|mm|kg|g|mg|usd|eur|°c|°f)?"
    return {(float(number.replace(",", "")), unit.casefold()) for number, unit in re.findall(pattern, text.casefold())}


def _negated_propositions(text: str) -> set[str]:
    normalised = normalise_text(text)
    matches = re.findall(r"\b(?:not|never|cannot|does not|did not|is not|was not)\s+(\w+)", normalised)
    return set(matches)


def _stem(token: str) -> str:
    value = token.casefold()
    if value.endswith("s") and len(value) > 3:
        return value[:-1]
    return re.sub(r"(?:ed|ing)$", "", value)


def hard_contradiction(a: str, b: str) -> bool:
    nums_a, nums_b = _numeric_tokens(a), _numeric_tokens(b)
    if nums_a and nums_b and nums_a != nums_b:
        return True
    neg_a, neg_b = _negated_propositions(a), _negated_propositions(b)
    if neg_a != neg_b:
        shared = {_stem(token) for token in normalise_text(a).split()} & {_stem(token) for token in normalise_text(b).split()}
        return bool(shared & {_stem(token) for token in (neg_a | neg_b)})
    return False


def connected_components(similar: list[list[bool]], responses: list[str] | None = None) -> list[list[int]]:
    groups: list[list[int]] = []
    seen: set[int] = set()
    ordering = sorted(range(len(similar)), key=lambda i: (normalise_text(responses[i]), responses[i], i)) if responses else list(range(len(similar)))
    for start in ordering:
        if start in seen:
            continue
        stack, group = [start], []
        while stack:
            index = stack.pop()
            if index in seen:
                continue
            seen.add(index)
            group.append(index)
            neighbors = [i for i, linked in enumerate(similar[index]) if linked and i not in seen]
            stack.extend(reversed(sorted(neighbors)))
        groups.append(sorted(group))
    return sorted(groups, key=lambda group: min((normalise_text(responses[i]), responses[i]) for i in group) if responses else group[0])


def cluster_responses(
    responses: list[str],
    method: str = "lexical_fallback",
    threshold: float = 0.78,
    embedding_model: str = "all-minilm",
    embedder: Embedder | None = None,
) -> tuple[list[SemanticCluster], dict[str, Any]]:
    if not responses:
        return [], {"method": method, "threshold": threshold, "similarity_matrix": [], "warnings": ["No responses were available for clustering."]}
    warnings: list[str] = []
    empty = [index for index, response in enumerate(responses) if not response.strip()]
    if empty:
        warnings.append(f"{len(empty)} empty or whitespace-only responses were isolated.")
    if method == "lexical_fallback":
        matrix = [[lexical_similarity(a, b) for b in responses] for a in responses]
        label = "lexical_fallback_jaccard"
    elif method == "embedding":
        vectors = (embedder or (lambda values: ollama_embeddings(values, embedding_model)))(responses)
        if len(vectors) != len(responses) or any(not vector for vector in vectors):
            raise RuntimeError("Embedding provider returned missing or empty vectors.")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise RuntimeError("Embedding vectors have inconsistent dimensions.")
        matrix = [[cosine(a, b) for b in vectors] for a in vectors]
        label = f"embedding_cosine:{embedding_model}"
    else:
        raise ValueError(f"Semantic method {method!r} is not configured.")
    linked = [
        [
            i == j or (
                bool(responses[i].strip()) and bool(responses[j].strip())
                and matrix[i][j] >= threshold and not hard_contradiction(responses[i], responses[j])
            )
            for j in range(len(responses))
        ]
        for i in range(len(responses))
    ]
    groups = connected_components(linked, responses)
    clusters = [
        SemanticCluster(
            cluster_id=f"cluster_{index + 1}", response_indices=group,
            responses=[responses[i] for i in group], size=len(group),
            representative_answer=responses[min(
                group,
                key=lambda i: (-sum(matrix[i][j] for j in group), normalise_text(responses[i]), responses[i], i),
            )],
            probability=len(group) / len(responses), evaluator_method=label,
        )
        for index, group in enumerate(groups)
    ]
    return clusters, {
        "method": label, "embedding_model": embedding_model if method == "embedding" else None,
        "embedding_model_digest": hashlib.sha256(embedding_model.encode()).hexdigest()[:16] if method == "embedding" else None,
        "algorithm": "threshold_graph_connected_components_v1",
        "threshold": threshold, "similarity_matrix": matrix,
        "sample_ordering": [{"index": i, "text_sha256": hashlib.sha256(text.encode()).hexdigest()} for i, text in enumerate(responses)],
        "cluster_assignments": {str(i): cluster.cluster_id for cluster in clusters for i in cluster.response_indices},
        "warnings": warnings,
    }


def threshold_sensitivity(
    examples: list[tuple[list[str], int | None]],
    thresholds: list[float],
    embedding_model: str = "all-minilm",
    embedder: Embedder | None = None,
) -> list[dict[str, Any]]:
    from api.core import average_precision, entropy_from_clusters, roc_auc
    baseline: list[tuple[tuple[int, ...], ...]] | None = None
    output = []
    for threshold in thresholds:
        structures, entropies, labels = [], [], []
        collapsed = separated = 0
        for responses, label in examples:
            clusters, _ = cluster_responses(responses, "embedding", threshold, embedding_model, embedder)
            structure = tuple(sorted(tuple(sorted(cluster.response_indices)) for cluster in clusters))
            structures.append(structure)
            entropy = entropy_from_clusters(clusters, len(responses))
            if entropy is not None and label is not None:
                entropies.append(entropy)
                labels.append(label)
            collapsed += int(len(clusters) == 1)
            separated += int(bool(responses) and len(clusters) == len(responses))
        if baseline is None:
            baseline = structures
        output.append({
            "threshold": threshold,
            "mean_clusters": sum(len(item) for item in structures) / len(structures) if structures else None,
            "mean_semantic_entropy": sum(entropies) / len(entropies) if entropies else None,
            "auroc": roc_auc(labels, entropies), "auprc": average_precision(labels, entropies),
            "structure_change_fraction": sum(a != b for a, b in zip(baseline, structures)) / len(structures) if structures else None,
            "single_cluster_fraction": collapsed / len(structures) if structures else None,
            "all_singletons_fraction": separated / len(structures) if structures else None,
            "warning": "Extreme threshold may collapse most examples." if structures and collapsed / len(structures) > .8 else ("Extreme threshold may separate most responses." if structures and separated / len(structures) > .8 else None),
        })
    return output
