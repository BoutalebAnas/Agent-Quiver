"""AgentQuiver: real Top-K package recommendations from a saved checkpoint.

Usage (from repository root):
 python -m scripts.predict --username gh:SOME_LOGIN --k 5 --checkpoint models/ckpt_loo_c4.pt

Returns *software packages* (npm:/pypi: ids), not invented GitHub repository cards.
Scores are uncalibrated model logits/similarity, NOT probabilities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.core_model import AgentQuiver, Ranker
from src.data import MAX_LEN, load


class ModelPredictor:
    """Load weights once and predict for any user (project or developer) present in interactions.csv."""

    def __init__(self, checkpoint: str | Path, data_dir: str | Path | None = None):
        self.checkpoint = Path(checkpoint)
        # Only load checkpoints from trusted sources; use safe tensor-only loading.
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        state = saved["model"]
        cfg = saved.get("cfg", {})
        n_items = int(state["enc.item.weight"].shape[0]) - 1
        d = int(state["enc.item.weight"].shape[1])
        c = int(state["mi.q"].shape[0])
        stored_text = state.get("enc.text_vecs")
        if stored_text is None and not cfg.get("no_text", False) and "enc.text.weight" in state:
            raise ValueError("Text-enabled checkpoint lacks its stored text vectors")
        self.model = AgentQuiver(n_items=n_items, d=d, c=c, text_vecs=stored_text,
                                 diffusion=bool(cfg.get("diffusion", False)))
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.ranker = None
        if saved.get("ranker") is not None:
            self.ranker = Ranker(self.model, d=d, c=c)
            self.ranker.load_state_dict(saved["ranker"], strict=True)
            self.ranker.eval()
        # NOTE: older checkpoint cfg does not store no_age; full-model checkpoints
        # from the user's run use no_age=False. Do not use this to claim a No Age model.
        self.interactions, self.items = load(data_dir) if data_dir is not None else load()
        # users are read from interactions.csv: ids look like "gh:<login>" or "pypi:<project>"
        users = self.interactions[["user_id", "developer"]].drop_duplicates("user_id")
        self.user_ids = dict(zip(users["developer"].astype(str).str.lower(), users["user_id"]))
        self.item_by_id = self.items.set_index("item_id")
        self.version = self.checkpoint.name
        self.n_items = n_items

    @torch.no_grad()
    def predict(self, username: str, k: int = 10, as_of: str | None = None) -> list[dict]:
        """Recommend k previously unseen *items* using current available history.

        ``as_of`` can bound the history (ISO timestamp) to avoid look-ahead for a demo.
        This is serving-style inference, not an offline test-set measurement.
        """
        if not 1 <= k <= 50:
            raise ValueError("k must be between 1 and 50")
        uid = self.user_ids.get(str(username).lower())
        if uid is None:
            raise ValueError(f"Unknown user {username!r}: not present in the dataset")
        instant = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
        if instant.tzinfo is None:
            instant = instant.tz_localize("UTC")
        else:
            instant = instant.tz_convert("UTC")
        history = self.interactions[(self.interactions["user_id"] == uid) &
                                    (self.interactions["date"] <= instant)].sort_values(["ts", "item_id"])
        if history.empty:
            raise ValueError(f"No interaction history for user {username!r} by {instant}")
        seen = set((history["item_id"].to_numpy(dtype=np.int64) + 1).tolist())
        recent = history.tail(MAX_LEN)
        ids = recent["item_id"].to_numpy(dtype=np.int64) + 1
        ages = ((instant - recent["date"]).dt.total_seconds().to_numpy() / 86400).clip(min=0)
        item_tensor = torch.from_numpy(ids[None, :])
        gap_tensor = torch.from_numpy(ages[None, :].astype(np.float32))
        retriever_scores = self.model.scores(item_tensor, gap_tensor)
        retriever_scores[:, 0] = float("-inf")
        if seen:
            retriever_scores[:, sorted(seen)] = float("-inf")
        candidates = min(100, self.n_items - len(seen))
        if candidates <= 0:
            return []
        candidate_ids = retriever_scores.topk(candidates, dim=1).indices
        if self.ranker is not None:
            candidate_scores = self.ranker(item_tensor, gap_tensor, candidate_ids)[0]
        else:
            candidate_scores = retriever_scores[0, candidate_ids[0]]
        order = torch.argsort(candidate_scores, descending=True)[:k]
        records = []
        for rank, pos in enumerate(order.tolist(), start=1):
            shifted_id = int(candidate_ids[0, pos])
            original_id = shifted_id - 1
            row = self.item_by_id.loc[original_id]
            score = float(candidate_scores[pos].item())
            repo_url = row.get("repo_url", "")
            records.append({
                "rank": rank,
                "item_id": original_id,
                "item": str(row["item"]),
                "score": round(score, 6),
                "score_type": "ranker_logit" if self.ranker else "retriever_similarity",
                "repo_url": "" if pd.isna(repo_url) else str(repo_url),
                "summary": "" if pd.isna(row.get("summary")) else str(row["summary"])[:240],
                "ecosystems": "" if pd.isna(row.get("ecosystems")) else str(row["ecosystems"]),
            })
        return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--username", required=True, help="User id as in interactions.csv, e.g. gh:<login> or pypi:<project>")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--checkpoint", required=True, help="Path to your saved full-model .pt")
    p.add_argument("--as-of", help="Optional ISO timestamp for serving history cutoff")
    args = p.parse_args()
    predictor = ModelPredictor(args.checkpoint)
    result = predictor.predict(args.username, k=args.k, as_of=args.as_of)
    print(json.dumps({"username": args.username, "model_version": predictor.version,
                      "is_mock": False, "recommendations": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
