"""Tests for the curve parser and the predictor. No checkpoints are overwritten by these tests."""
from pathlib import Path

import pytest

from scripts.training_curves import parse_log
from scripts.predict import ModelPredictor


def test_curve_parser_only_reads_observed_lines():
    rows = parse_log("[retr] ep00 loss 9.12 valid R@20 0.123 NDCG@10 0.045\n"
                     "[rank] ep00 valid R@20 0.145 NDCG@10 0.067\n"
                     "TEST {'some': 2}\n")
    assert len(rows) == 2
    assert rows[0]["train_loss"] == 9.12
    assert rows[1]["stage"] == "ranker"
    assert rows[1]["train_loss"] == ""


def test_curve_parser_rejects_missing_history():
    assert parse_log("TEST {'recall@20': 0.2}") == []


@pytest.fixture(scope="module")
def predictor():
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "models" / "ckpt_loo_c4.pt"
    if not checkpoint.exists():
        pytest.skip("No local checkpoint")
    return ModelPredictor(checkpoint)


def test_prediction_returns_real_unseen_items(predictor):
    username = next(iter(predictor.user_ids))
    pred = predictor.predict(username, k=3)
    assert len(pred) == 3
    assert all(r["item"].startswith(("npm:", "pypi:", "tool:")) for r in pred)
    uid = predictor.user_ids[username]
    seen = set(predictor.interactions.loc[predictor.interactions.user_id == uid, "item_id"])
    assert not ({r["item_id"] for r in pred} & seen)
    assert [r["rank"] for r in pred] == [1, 2, 3]
    assert all(r["score_type"] in ("ranker_logit", "retriever_similarity") for r in pred)


def test_unknown_user_is_rejected(predictor):
    with pytest.raises(ValueError, match="Unknown user"):
        predictor.predict("unknown_user_not_in_dataset_xyz", 3)
