"""'Better tools' route and the same-ecosystem filter of the trained model."""
import importlib
import os
import tempfile
from pathlib import Path

import pytest

DATA = Path("data/processed/migrations.csv")
CKPT = Path("models/ckpt_loo_c4.pt")
pytestmark = pytest.mark.skipif(not (DATA.exists() and CKPT.exists()),
                                reason="needs data/processed/migrations.csv and models/ckpt_loo_c4.pt")


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


def test_better_tools_for_a_project_using_toml(client):
    r = client.get("/better_tools", params={"username": "pypi:prefect"})
    assert r.status_code == 200
    pairs = {(t["replace"], t["with"]) for t in r.json()}
    assert ("pypi:toml", "pypi:tomli") in pairs
    assert all(t["projects"] >= 3 for t in r.json())
    olds = [t["replace"] for t in r.json()]
    assert len(olds) == len(set(olds))          # one suggestion per replaced package


def test_better_tools_never_suggest_an_adopted_package(client):
    import pandas as pd
    inter = pd.read_csv("data/processed/interactions.csv", usecols=["developer", "item"])
    for user in ["pypi:prefect", "pypi:langchain-google-vertexai", "gh:DhruvTambekar24"]:
        history = set(inter.loc[inter["developer"] == user, "item"])
        for t in client.get("/better_tools", params={"username": user}).json():
            assert t["with"] not in history


def test_better_tools_unknown_user_is_404(client):
    assert client.get("/better_tools", params={"username": "nobody-xyz-123"}).status_code == 404


def test_recommendations_stay_in_the_user_ecosystems(client):
    # a PyPI-only project must not be offered npm packages
    body = client.get("/recommend", params={"username": "pypi:pyobjc", "k": 20}).json()
    ids = [x["repo"]["repo_id"] for it in body["interests"] for x in it["recommendations"]]
    assert len(ids) == 20
    assert all(i.split(":")[0] in ("pypi", "tool") for i in ids)
