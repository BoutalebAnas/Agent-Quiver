"""End-to-end tests with the TRAINED model. Skipped if the checkpoint or data are missing."""
import importlib
import os
import tempfile
from pathlib import Path

import pytest

CKPT = Path("models/ckpt_loo_c4.pt")
DATA = Path("data/processed/interactions.csv")
pytestmark = pytest.mark.skipif(not (CKPT.exists() and DATA.exists()),
                                reason="needs models/ckpt_loo_c4.pt and data/processed/interactions.csv")


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


def test_health_is_not_mock(client):
    body = client.get("/health").json()
    assert body["is_mock"] is False
    assert body["model_version"] == "ckpt_loo_c4.pt"


def test_recommend_real_user(client):
    user = client.get("/users", params={"limit": 1}).json()[0]
    r = client.get("/recommend", params={"username": user, "k": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["is_mock"] is False
    recs = [x for it in body["interests"] for x in it["recommendations"]]
    assert sorted(x["rank"] for x in recs) == list(range(1, 11))
    assert all(x["repo"]["kind"] == "package" for x in recs)
    assert all(it["explanation"].startswith("Because you used") for it in body["interests"])


def test_unknown_user_is_404(client):
    assert client.get("/recommend", params={"username": "nobody-xyz-123"}).status_code == 404
