"""'Your stack' (GET /history) and recommendations for a pasted stack (POST /recommend_stack)."""
import importlib
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.api.stack import normalize, parse_packages

CKPT = Path("models/ckpt_loo_c4.pt")
DATA = Path("data/processed/interactions.csv")


def test_parse_requirements_txt():
    text = ("# comment\n-r base.txt\n-e .\nhttps://x.org/pkg.whl\nrequests[security]==2.31 ; python_version>'3.8'\n"
            "httpx>=0.27\nGoogle_Cloud.Storage~=2.0  # inline comment\nnumpy\nnumpy\n")
    assert parse_packages(text) == ["requests", "httpx", "google-cloud-storage", "numpy"]


def test_parse_package_json_and_lists():
    pj = '{"name": "app", "dependencies": {"react": "^18", "@supabase/supabase-js": "^2"}, "devDependencies": {"vite": "5"}}'
    assert parse_packages(pj) == ["react", "@supabase/supabase-js", "vite"]
    assert parse_packages("openai, anthropic,langchain") == ["openai", "anthropic", "langchain"]
    assert normalize("Typing_Extensions") == "typing-extensions"


needs_model = pytest.mark.skipif(not (CKPT.exists() and DATA.exists()),
                                 reason="needs models/ckpt_loo_c4.pt and data/processed")


@pytest.fixture(scope="module")
def client():
    os.environ["AQ_RECOMMENDER"] = "model"
    os.environ["AQ_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
    import src.api.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    yield TestClient(main.app)
    os.environ["AQ_RECOMMENDER"] = "mock"
    importlib.reload(main)


@needs_model
def test_history_matches_the_dataset(client):
    user = "pypi:langchain-google-vertexai"
    r = client.get("/history", params={"username": user})
    assert r.status_code == 200
    inter = pd.read_csv(DATA, usecols=["developer", "item"])
    assert {h["package"] for h in r.json()} == set(inter.loc[inter.developer == user, "item"])
    assert [h["date"] for h in r.json()] == sorted(h["date"] for h in r.json())
    assert client.get("/history", params={"username": "pypi:does-not-exist-xyz"}).status_code == 404


@needs_model
def test_recommend_stack(client):
    text = "langchain-core\ngoogle-cloud-storage\nhttpx>=0.27\nnot-a-real-package-xyz"
    r = client.post("/recommend_stack", json={"text": text, "k": 10})
    assert r.status_code == 200
    d = r.json()
    assert "pypi:httpx" in d["matched"] and d["unknown"] == ["not-a-real-package-xyz"]
    recs = [x["repo"]["repo_id"] for it in d["interests"] for x in it["recommendations"]]
    assert len(recs) == 10 and len(set(recs)) == 10
    assert not set(recs) & set(d["matched"])                        # never the stack itself
    assert all(x.split(":")[0] in {"pypi", "tool"} for x in recs)  # stays in the stack's ecosystems
    assert sorted(x["rank"] for it in d["interests"] for x in it["recommendations"]) == list(range(1, 11))
    assert ("pypi:httpx", "pypi:httpx2") in {(t["replace"], t["with"]) for t in d["better_tools"]}


@needs_model
def test_recommend_stack_npm_and_unknown(client):
    d = client.post("/recommend_stack", json={"text": '{"dependencies": {"react": "^18", "axios": "^1"}}', "k": 5}).json()
    recs = [x["repo"]["repo_id"] for it in d["interests"] for x in it["recommendations"]]
    assert all(x.split(":")[0] in {"npm", "tool"} for x in recs)
    assert client.post("/recommend_stack", json={"text": "zzz-unknown-1\nzzz-unknown-2"}).status_code == 404
