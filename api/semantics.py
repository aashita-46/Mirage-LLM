"""Semantic-equivalence clustering with explicit method metadata."""
from __future__ import annotations

import math
import re
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


def _hard_contradiction(a: str, b: str) -> bool:
    nums_a = set(re.findall(r"-?\d+(?:\.\d+)?", a))
    nums_b = set(re.findall(r"-?\d+(?:\.\d+)?", b))
    if nums_a and nums_b and nums_a != nums_b:
        return True
    neg_a = bool(re.search(r"\b(?:not|no|never|cannot|doesn't|does not)\b", a.casefold()))
    neg_b = bool(re.search(r"\b(?:not|no|never|cannot|doesn't|does not)\b", b.casefold()))
    return neg_a != neg_b and lexical_similarity(a, b) >= 0.35


def _connected_components(similar: list[list[bool]]) -> list[list[int]]:
    groups: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(similar)):
        if start in seen:
            continue
        stack, group = [start], []
        while stack:
            index = stack.pop()
            if index in seen:
                continue
            seen.add(index)
            group.append(index)
            stack.extend(i for i, linked in enumerate(similar[index]) if linked and i not in seen)
        groups.append(sorted(group))
    return groups


def cluster_responses(
    responses: list[str],
    method: str = "lexical_fallback",
    threshold: float = 0.78,
    embedding_model: str = "all-minilm",
    embedder: Embedder | None = None,
) -> tuple[list[SemanticCluster], dict[str, Any]]:
    if not responses:
        return [], {"method": method, "threshold": threshold, "similarity_matrix": []}
    if method == "lexical_fallback":
        matrix = [[lexical_similarity(a, b) for b in responses] for a in responses]
        label = "lexical_fallback_jaccard"
    elif method == "embedding":
        vectors = (embedder or (lambda values: ollama_embeddings(values, embedding_model)))(responses)
        matrix = [[cosine(a, b) for b in vectors] for a in vectors]
        label = f"embedding_cosine:{embedding_model}"
    else:
        raise ValueError(f"Semantic method {method!r} is not configured.")
    linked = [
        [
            i == j or (matrix[i][j] >= threshold and not _hard_contradiction(responses[i], responses[j]))
            for j in range(len(responses))
        ]
        for i in range(len(responses))
    ]
    groups = _connected_components(linked)
    clusters = [
        SemanticCluster(
            cluster_id=f"cluster_{index + 1}", response_indices=group,
            responses=[responses[i] for i in group], size=len(group),
            representative_answer=max(
                (responses[i] for i in group),
                key=lambda item: sum(matrix[responses.index(item)][j] for j in group),
            ),
            probability=len(group) / len(responses), evaluator_method=label,
        )
        for index, group in enumerate(groups)
    ]
    return clusters, {
        "method": label, "embedding_model": embedding_model if method == "embedding" else None,
        "threshold": threshold, "similarity_matrix": matrix,
    }
