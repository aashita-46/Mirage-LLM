"""Mirage API: deterministic demo engine plus reusable uncertainty mathematics."""
from __future__ import annotations

import hashlib
import math
import os
import random
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Mirage API", version="1.0.0", docs_url="/api/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DEMO_KNOWLEDGE = {
    "capital of australia": (
        "Australia's capital is Canberra. It was selected as a compromise between Sydney and Melbourne.",
        ["Canberra is the capital of Australia.", "The capital is Canberra, in the Australian Capital Territory.",
         "Canberra—not Sydney—is Australia's capital.", "Australia's seat of government is Canberra."],
        0.94,
    ),
    "first world war": (
        "The Treaty of Versailles, signed in 1919, formally ended the state of war between Germany and the Allied powers.",
        ["The Treaty of Versailles ended the war with Germany.", "Versailles is the commonly expected answer.",
         "The 1919 Treaty of Versailles ended hostilities with Germany.", "Several treaties ended the war; for Germany, Versailles."],
        0.82,
    ),
    "penicillin": (
        "Alexander Fleming discovered penicillin in 1928 after observing mould inhibiting bacterial growth.",
        ["Alexander Fleming discovered it in 1928.", "Fleming's 1928 observation led to penicillin.",
         "Penicillin was discovered by Alexander Fleming in 1928.", "Fleming noticed the antibacterial mould in 1928."],
        0.91,
    ),
    "paracetamol": (
        "For many healthy adults, common labels set a maximum of 4,000 mg in 24 hours, but lower limits may apply. Follow the product label and a clinician's advice; liver disease, alcohol use, low body weight, or interacting medicines can change what is safe.",
        ["Many labels specify no more than 4,000 mg per 24 hours.", "A typical adult ceiling is 4 g daily, with important exceptions.",
         "Some guidance uses a lower 3,000 mg limit depending on product and patient.", "Ask a clinician; the limit depends on health and formulation."],
        0.63,
    ),
    "default": (
        "This question needs careful verification. Mirage can show where generated answers agree, disagree, or rely on uncertain details, but it cannot independently establish factual truth.",
        ["The available context is insufficient for a definitive answer.", "This claim should be checked against a primary source.",
         "There may be multiple reasonable interpretations.", "A reliable answer requires additional evidence."],
        0.45,
    ),
}

class AnalyseRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    domain: str = Field(default="general", max_length=40)
    sample_count: int = Field(default=6, ge=2, le=10)
    temperature: float = Field(default=0.7, ge=0, le=1.5)
    top_p: float = Field(default=0.9, ge=0.1, le=1)

class StressRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    types: list[str] = Field(default_factory=lambda: ["neutral", "formal", "distractor", "leading"])

class BenchRequest(BaseModel):
    count: int = Field(default=12, ge=4, le=24)
    seed: int = 42

def importance(token: str) -> tuple[float, str]:
    clean = token.strip(".,;:!?()[]")
    if re.fullmatch(r"\d+(?:[.,]\d+)?%?", clean):
        return 1.65, "number"
    if re.fullmatch(r"(?:mg|kg|g|ml|km|usd|inr|€|£|\$)", clean.lower()):
        return 1.55, "unit"
    if clean[:1].isupper() and len(clean) > 2:
        return 1.35, "entity"
    if re.fullmatch(r"\W+", token):
        return 0.35, "punctuation"
    if clean.lower() in {"the", "a", "an", "is", "of", "and", "to", "in", "as", "it"}:
        return 0.55, "function"
    return 1.0, "content"

def tokenise(answer: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    parts = re.findall(r"\S+\s*", answer)
    rows = []
    for i, text in enumerate(parts):
        weight, category = importance(text)
        base = min(0.96, max(0.05, rng.betavariate(2.4, 5.5)))
        if category in {"number", "entity", "unit"}:
            base = min(1, base + rng.uniform(0.04, 0.22))
        rows.append({
            "id": i, "text": text, "logprob": round(-0.1 - base * 2.3, 3),
            "entropy": round(base * 5.2, 3), "normalisedUncertainty": round(base, 3),
            "importanceWeight": weight, "weightedRisk": round(min(1, base * weight), 3),
            "category": category,
        })
    return rows

def semantic_entropy(assignments: list[int]) -> tuple[float, float, dict[int, float]]:
    counts = {x: assignments.count(x) for x in set(assignments)}
    probabilities = {k: v / len(assignments) for k, v in counts.items()}
    raw = -sum(p * math.log(p) for p in probabilities.values())
    max_h = math.log(len(assignments)) if len(assignments) > 1 else 0
    return raw, (raw / max_h if max_h else 0), probabilities

def aggregate_token_risk(tokens: list[dict[str, Any]]) -> float:
    risks = sorted((t["weightedRisk"] for t in tokens), reverse=True)
    if not risks:
        return 0
    weights = [t["importanceWeight"] for t in tokens]
    mean = sum(t["normalisedUncertainty"] * t["importanceWeight"] for t in tokens) / sum(weights)
    idx = min(len(risks) - 1, max(0, math.ceil(len(risks) * 0.1) - 1))
    return min(1, 0.7 * mean + 0.3 * risks[idx])

def score(signals: dict[str, float | None]) -> tuple[float, dict[str, float]]:
    base = {"semantic": .5, "token": .2, "ptrue": .2, "stability": .1}
    active = {k: w for k, w in base.items() if signals.get(k) is not None}
    total = sum(active.values())
    weights = {k: w / total for k, w in active.items()}
    return 100 * sum(float(signals[k]) * weights[k] for k in active), weights

def analyse(req: AnalyseRequest) -> dict[str, Any]:
    started = time.perf_counter()
    key = next((k for k in DEMO_KNOWLEDGE if k != "default" and k in req.question.lower()), "default")
    primary, choices, p_true = DEMO_KNOWLEDGE[key]
    digest = int(hashlib.sha256(req.question.encode()).hexdigest()[:8], 16)
    tokens = tokenise(primary, digest)
    samples = []
    assignments = []
    for i in range(req.sample_count):
        answer = choices[i % len(choices)]
        cluster = 1 if key == "paracetamol" and i in {2, 3} else (1 if key == "default" and i % 3 == 2 else 0)
        assignments.append(cluster)
        samples.append({"id": f"s{i+1}", "answer": answer, "cluster": cluster,
                        "latency": round(.34 + (i * .11), 2), "agreement": round(.94 if cluster == 0 else .52, 2)})
    raw_h, norm_h, probs = semantic_entropy(assignments)
    token_risk = aggregate_token_risk(tokens)
    final, weights = score({"semantic": norm_h, "token": token_risk, "ptrue": 1-p_true, "stability": None})
    clusters = [{"id": str(k), "label": "Dominant meaning" if k == 0 else "Alternative meaning",
                 "sampleIds": [s["id"] for s in samples if s["cluster"] == k],
                 "probability": p, "representativeAnswer": next(s["answer"] for s in samples if s["cluster"] == k)}
                for k, p in probs.items()]
    return {
        "id": str(uuid.uuid4()), "question": req.question, "answer": primary, "tokens": tokens,
        "samples": samples, "clusters": clusters, "semanticEntropy": round(raw_h, 4),
        "normalisedSemanticEntropy": round(norm_h, 4), "pTrue": p_true,
        "verification": "The central claim is internally consistent; important details should still be checked against a primary source.",
        "score": round(final, 1), "breakdown": {"semantic": norm_h, "token": token_risk, "inversePTrue": 1-p_true},
        "weights": weights, "mode": "cached_demo", "calibrationStatus": "uncalibrated",
        "model": "Mirage curated demonstration", "device": "serverless CPU",
        "latency": round(time.perf_counter()-started+.42, 2),
        "metadata": {"temperature": req.temperature, "topP": req.top_p, "sampleCount": req.sample_count,
                     "nliThreshold": .7, "timestamp": datetime.now(timezone.utc).isoformat(), "seed": digest},
    }

@app.get("/api/v1/health")
def health():
    return {"status": "ok", "mode": "cached_demo", "version": "1.0.0"}

@app.get("/api/v1/system")
def system():
    return {"backend": True, "mode": "cached_demo", "provider": "Curated demo provider",
            "modelLoaded": True, "model": "Mirage curated demonstration", "nli": "Lexical/NLI fallback",
            "device": "CPU", "gpu": False, "groq": bool(os.getenv("GROQ_API_KEY")),
            "demoCache": True, "database": "Ephemeral deployment store"}

@app.get("/api/v1/demo/examples")
def examples():
    return {"examples": [
        "What is the capital of Australia?",
        "Who discovered penicillin, and in which year?",
        "Which treaty formally ended the First World War?",
        "What is the maximum recommended daily dose of paracetamol for a healthy adult?",
        "If every raven is black, does seeing a black bird prove it is a raven?",
    ]}

@app.post("/api/v1/analyse")
def analyse_route(req: AnalyseRequest):
    return analyse(req)

@app.post("/api/v1/stress")
def stress(req: StressRequest):
    variants = []
    transforms = {
        "neutral": lambda q: f"In other words, {q[:1].lower()}{q[1:]}",
        "formal": lambda q: f"Please provide a precise, formally worded response to: {q}",
        "conversational": lambda q: f"Quick question: {q}",
        "distractor": lambda q: f"Ignore unrelated background details and answer this: {q}",
        "leading": lambda q: f"Some people claim an obvious answer exists. {q}",
        "ambiguity": lambda q: f"Depending on interpretation, {q[:1].lower()}{q[1:]}",
        "negation": lambda q: f"Adversarial negation test: Is it false that — {q}",
    }
    base = analyse(AnalyseRequest(question=req.question, sample_count=4))
    for i, kind in enumerate(req.types[:8]):
        text = transforms.get(kind, transforms["neutral"])(req.question)
        delta = ((i % 3) - 1) * 4.7
        variants.append({"id": i, "type": kind, "question": text, "answer": base["answer"],
                         "score": round(max(0, min(100, base["score"] + delta)), 1),
                         "agreement": round(max(.4, .94 - i*.06), 2),
                         "relation": "equivalent" if kind not in {"negation"} else "unclear",
                         "adversarial": kind in {"negation", "leading"}})
    stability = sum(v["agreement"] for v in variants) / len(variants)
    return {"original": base, "variants": variants, "stability": round(stability, 3),
            "instability": round(1-stability, 3),
            "summary": f"{sum(v['relation']=='equivalent' for v in variants)} of {len(variants)} variants preserved the same answer meaning."}

BENCH = [
    ("Capital of Australia?", "Canberra", True, .12), ("2 + 2?", "4", True, .08),
    ("Who discovered penicillin?", "Alexander Fleming", True, .18), ("Capital of Turkey?", "Ankara", True, .19),
    ("Is correlation proof of causation?", "No", True, .21), ("Treaty with Germany after WWI?", "Versailles", True, .25),
    ("Can all revenue be recognised before delivery?", "No", True, .31), ("Boiling point of water at sea level?", "100°C", True, .22),
    ("Does every black bird have to be a raven?", "No", True, .38), ("Largest planet?", "Jupiter", True, .16),
    ("Who was the first person on Mars?", "Nobody", True, .44), ("Square root of -1 over reals?", "Undefined", True, .35),
    ("Ambiguous bank question", "Context needed", False, .67), ("False premise example", "Premise rejected", False, .78),
    ("Unverifiable future claim", "Unknown", False, .84), ("Context-free dosage", "Needs context", False, .73),
]

def roc_auc(targets: list[int], preds: list[float]) -> float | None:
    pos = [p for p, y in zip(preds, targets) if y == 1]
    neg = [p for p, y in zip(preds, targets) if y == 0]
    if not pos or not neg:
        return None
    return sum((a > b) + .5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))

@app.post("/api/v1/bench/runs")
def bench(req: BenchRequest):
    rng = random.Random(req.seed)
    records = list(BENCH)
    rng.shuffle(records)
    records = records[:req.count]
    targets = [0 if r[2] else 1 for r in records]
    preds = [r[3] for r in records]
    auc = roc_auc(targets, preds)
    brier = sum((p-y)**2 for p, y in zip(preds, targets))/len(records)
    bins = []
    for low in [0, .2, .4, .6, .8]:
        vals = [(p,y) for p,y in zip(preds,targets) if low <= p < low+.2]
        if vals:
            bins.append({"range": f"{low:.1f}–{low+.2:.1f}", "predicted": sum(p for p,_ in vals)/len(vals),
                         "observed": sum(y for _,y in vals)/len(vals), "count": len(vals)})
    ece = sum(b["count"]/len(records)*abs(b["predicted"]-b["observed"]) for b in bins)
    rows = [{"question": r[0], "reference": r[1], "correct": r[2], "risk": r[3],
             "method": "curated reference match"} for r in records]
    return {"id": str(uuid.uuid4()), "dataset": "Mirage curated demo", "count": len(records),
            "incorrect": sum(targets), "auroc": round(auc, 3) if auc is not None else None,
            "ece": round(ece, 3), "brier": round(brier, 3), "bins": bins, "records": rows,
            "mode": "computed_demo", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/api/v1/models")
def models():
    return {"models": [{"id": "demo", "name": "Mirage curated demonstration", "available": True},
                       {"id": "qwen-0.5b", "name": "Qwen2.5 0.5B Instruct", "available": False}]}
