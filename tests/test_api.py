from fastapi.testclient import TestClient
from api.index import app, aggregate_token_risk, importance, roc_auc, score, semantic_entropy

client = TestClient(app)

def test_health():
    assert client.get("/api/v1/health").json()["status"] == "ok"

def test_one_cluster_entropy():
    raw, norm, probs = semantic_entropy([0, 0, 0])
    assert raw == 0 and norm == 0 and probs == {0: 1.0}

def test_multi_cluster_entropy():
    raw, norm, probs = semantic_entropy([0, 0, 1, 1])
    assert raw > 0 and 0 < norm <= 1 and probs[0] == .5

def test_weight_renormalisation():
    value, weights = score({"semantic": .5, "token": .5, "ptrue": .5, "stability": None})
    assert round(sum(weights.values()), 5) == 1 and value == 50

def test_importance():
    assert importance("1928")[0] > importance("the")[0]
    assert importance("Canberra")[1] == "entity"

def test_token_aggregate():
    value = aggregate_token_risk([{"weightedRisk": .8, "importanceWeight": 1, "normalisedUncertainty": .8}])
    assert 0 <= value <= 1

def test_auc_single_class():
    assert roc_auc([0, 0], [.1, .2]) is None

def test_analysis_contract():
    data = client.post("/api/v1/analyse", json={"question": "What is the capital of Australia?", "sample_count": 6}).json()
    assert data["answer"].startswith("Australia") and len(data["tokens"]) > 3
    assert data["mode"] == "cached_demo"

def test_benchmark_computes_metrics():
    data = client.post("/api/v1/bench/runs", json={"count": 12, "seed": 42}).json()
    assert data["count"] == 12 and data["auroc"] is not None
