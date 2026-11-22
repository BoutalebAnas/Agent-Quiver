"""Tests of the API with the MOCK recommender (always, whatever AQ_RECOMMENDER says)."""
import importlib
import os

from fastapi.testclient import TestClient

os.environ["AQ_RECOMMENDER"] = "mock"
import src.api.main as main  # noqa: E402

importlib.reload(main)
app = main.app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_recommend_returns_k_repos_grouped_by_interest():
    r = client.get("/recommend", params={"username": "torvalds", "k": 10})
    assert r.status_code == 200
    body = r.json()
    repos = [rec for it in body["interests"] for rec in it["recommendations"]]
    assert len(repos) == 10
    assert sorted(x["rank"] for x in repos) == list(range(1, 11))
    assert len(body["interests"]) == 4
    assert all(it["explanation"].startswith("Because you work on") for it in body["interests"])


def test_scores_follow_rank():
    body = client.get("/recommend", params={"username": "gvanrossum"}).json()
    repos = sorted((rec for it in body["interests"] for rec in it["recommendations"]),
                   key=lambda x: x["rank"])
    scores = [x["score"] for x in repos]
    assert scores == sorted(scores, reverse=True)


def test_same_user_same_answer_different_user_different_answer():
    ids = lambda u: [rec["repo"]["repo_id"] for it in
                     client.get("/recommend", params={"username": u}).json()["interests"]
                     for rec in it["recommendations"]]
    assert ids("alice") == ids("alice")
    assert ids("alice") != ids("bob")


def test_invalid_username_and_k():
    assert client.get("/recommend", params={"username": "has space"}).status_code == 422
    assert client.get("/recommend", params={"username": "ok", "k": 0}).status_code == 422


def test_log_interaction():
    payload = {"user_id": "alice", "repo_id": "vllm-project/vllm",
               "interest_id": 0, "rank": 1, "action": "useful"}
    r = client.post("/log_interaction", json=payload)
    assert r.status_code == 201
    assert r.json()["interaction_id"] >= 1
    assert any(e["repo_id"] == "vllm-project/vllm"
               for e in client.get("/interactions", params={"user_id": "alice"}).json())
    bad = dict(payload, action="maybe")
    assert client.post("/log_interaction", json=bad).status_code == 422


def test_users_endpoint():
    r = client.get("/users", params={"limit": 2})
    assert r.status_code == 200 and len(r.json()) == 2
