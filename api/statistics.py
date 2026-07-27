"""Versioned statistical analysis for research experiments."""
from __future__ import annotations

import math
import random
from typing import Callable
from typing import Any

METRIC_VERSION = "2.1"


def _interval_payload(
    estimate: float | None, measured: list[float], requested: int, seed: int,
    confidence: float, sample_count: int,
) -> dict[str, Any]:
    measured.sort()
    alpha = (1 - confidence) / 2
    low = measured[max(0, int(alpha * len(measured)))] if measured else None
    high = measured[min(len(measured) - 1, max(0, math.ceil((1 - alpha) * len(measured)) - 1))] if measured else None
    warnings = []
    if sample_count < 30:
        warnings.append("Bootstrap interval is unstable with fewer than 30 eligible examples.")
    if len(measured) < requested:
        warnings.append("Some resamples were rejected because the statistic was undefined.")
    return {
        "estimate": estimate, "low": low, "high": high, "confidence_level": confidence,
        "requested_resamples": requested, "valid_resamples": len(measured),
        "rejected_resamples": requested - len(measured), "seed": seed,
        "method": "percentile_bootstrap", "eligible_samples": sample_count, "warnings": warnings,
    }


def bootstrap_ci(
    values: list[tuple[int, float]],
    metric: Callable[[list[int], list[float]], float | None],
    resamples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if not values:
        return _interval_payload(None, [], resamples, seed, confidence, 0)
    labels, scores = zip(*values)
    estimate = metric(list(labels), list(scores))
    rng, measured = random.Random(seed), []
    for _ in range(resamples):
        indices = [rng.randrange(len(values)) for _ in values]
        sample = metric([labels[i] for i in indices], [scores[i] for i in indices])
        if sample is not None and math.isfinite(sample):
            measured.append(sample)
    return _interval_payload(estimate, measured, resamples, seed, confidence, len(values))


def bootstrap_statistic(
    rows: list[Any],
    statistic: Callable[[list[Any]], float | None],
    resamples: int = 1000,
    seed: int = 42,
    confidence: float = .95,
) -> dict[str, Any]:
    if not rows:
        return _interval_payload(None, [], resamples, seed, confidence, 0)
    estimate = statistic(rows)
    rng, values = random.Random(seed), []
    for _ in range(resamples):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        value = statistic(sample)
        if value is not None and math.isfinite(value):
            values.append(value)
    return _interval_payload(estimate, values, resamples, seed, confidence, len(rows))


def paired_bootstrap_difference(
    labels: list[int],
    first: list[float],
    second: list[float],
    metric: Callable[[list[int], list[float]], float | None],
    resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float | int | str | None]:
    if not (len(labels) == len(first) == len(second)) or not labels:
        raise ValueError("Paired comparison requires aligned non-empty inputs.")
    first_value, second_value = metric(labels, first), metric(labels, second)
    estimate = None if first_value is None or second_value is None else first_value - second_value
    rng, differences = random.Random(seed), []
    for _ in range(resamples):
        indices = [rng.randrange(len(labels)) for _ in labels]
        ys = [labels[i] for i in indices]
        a, b = metric(ys, [first[i] for i in indices]), metric(ys, [second[i] for i in indices])
        if a is not None and b is not None:
            differences.append(a - b)
    differences.sort()
    low = differences[int(.025 * len(differences))] if differences else None
    high = differences[min(len(differences) - 1, int(.975 * len(differences)))] if differences else None
    interpretation = "inconclusive"
    if low is not None and low > 0:
        interpretation = "evidence_of_improvement"
    elif high is not None and high < 0:
        interpretation = "evidence_of_decrease"
    return {
        "difference": estimate, "low": low, "high": high,
        "confidence_level": .95, "requested_resamples": resamples,
        "valid_resamples": len(differences), "rejected_resamples": resamples - len(differences),
        "seed": seed, "method": "paired_percentile_bootstrap",
        "eligible_samples": len(labels), "missing_value_exclusions": 0,
        "interpretation": "insufficient_data" if estimate is None else interpretation,
    }


def stratified_split(labels: list[int], test_fraction: float = .3, seed: int = 42) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    train, test = [], []
    for label in sorted(set(labels)):
        indices = [i for i, value in enumerate(labels) if value == label]
        rng.shuffle(indices)
        cut = max(1, round(len(indices) * test_fraction)) if len(indices) > 1 else 0
        test.extend(indices[:cut])
        train.extend(indices[cut:])
    return sorted(train), sorted(test)


def fit_platt(scores: list[float], labels: list[int], iterations: int = 1000, rate: float = .05) -> dict[str, float]:
    if len(set(labels)) < 2:
        raise ValueError("Platt scaling requires both classes in the training split.")
    a, b = 1.0, 0.0
    for _ in range(iterations):
        probs = [1 / (1 + math.exp(-max(-30, min(30, a * score + b)))) for score in scores]
        da = sum((p - y) * score for p, y, score in zip(probs, labels, scores)) / len(scores)
        db = sum(p - y for p, y in zip(probs, labels)) / len(scores)
        a -= rate * da
        b -= rate * db
    return {"method": "platt", "a": a, "b": b}


def apply_platt(scores: list[float], parameters: dict[str, float]) -> list[float]:
    return [1 / (1 + math.exp(-max(-30, min(30, parameters["a"] * score + parameters["b"])))) for score in scores]


def fit_isotonic(scores: list[float], labels: list[int]) -> dict[str, list[float] | str]:
    ordered = sorted(zip(scores, labels))
    blocks = [{"x": [x], "sum": float(y), "count": 1} for x, y in ordered]
    index = 0
    while index < len(blocks) - 1:
        left, right = blocks[index], blocks[index + 1]
        if left["sum"] / left["count"] > right["sum"] / right["count"]:
            left["x"].extend(right["x"])
            left["sum"] += right["sum"]
            left["count"] += right["count"]
            blocks.pop(index + 1)
            index = max(0, index - 1)
        else:
            index += 1
    return {
        "method": "isotonic",
        "thresholds": [max(block["x"]) for block in blocks],
        "values": [block["sum"] / block["count"] for block in blocks],
    }


def apply_isotonic(scores: list[float], parameters: dict[str, list[float] | str]) -> list[float]:
    thresholds = parameters["thresholds"]
    values = parameters["values"]
    assert isinstance(thresholds, list) and isinstance(values, list)
    return [next((values[i] for i, threshold in enumerate(thresholds) if score <= threshold), values[-1]) for score in scores]
