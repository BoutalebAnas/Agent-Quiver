"""Bridge between the API and the trained AgentQuiver model (week 3).

Uses the team's `ModelPredictor` (scripts/predict.py) for the top-k list, so the
API returns exactly what the model predicts. On top of it, this file adds what
the API contract needs:
  * each recommendation is assigned to the user interest (OMIE channel) it
    matches best, so the app can show "top items per interest";
  * each interest is explained by the history items it attends to most
    ("Because you used X and Y"), taken from the model's attention weights.

Items are npm / PyPI packages (e.g. "pypi:requests"), not GitHub repositories.
Scores are cosine similarity x learned scale, not probabilities.
"""
from __future__ import annotations

import re

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch

from scripts.predict import ModelPredictor
from src.data import MAX_LEN

from .recommender import UnknownUserError
from .schemas import Interest, RecommendedRepo, RepoCard


def _short(item: str) -> str:
    """'pypi:requests' -> 'requests'."""
    return item.split(":", 1)[1] if ":" in item else item


def _ecos(value) -> set:
    """'npm|pypi' -> {'npm', 'pypi'}."""
    if not isinstance(value, str):
        return set()
    return {e.strip() for e in re.split(r"[|,]", value) if e.strip()}


def _url(item: str) -> str | None:
    eco, _, name = item.partition(":")
    if eco == "pypi":
        return f"https://pypi.org/project/{name}/"
    if eco == "npm":
        return f"https://www.npmjs.com/package/{name}"
    return None


class ModelRecommender:
    is_mock = False

    def __init__(self, checkpoint: str | Path, data_dir: str | Path | None = None):
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint} (put ckpt_loo_c4.pt in models/)")
        self.p = ModelPredictor(checkpoint, data_dir)   # loads weights ONCE, at API start-up
        self.model_version = self.p.version
        self.score_type = "ranker_logit" if self.p.ranker is not None else "retriever_similarity"
        self.n_channels = int(self.p.model.mi.q.shape[0])

    # ------------------------------------------------------------------ helpers
    def _history(self, username: str, instant: pd.Timestamp) -> pd.DataFrame:
        """Same history selection as ModelPredictor.predict (last MAX_LEN events before `instant`)."""
        uid = self.p.user_ids.get(username.lower())
        inter = self.p.interactions
        h = inter[(inter["user_id"] == uid) & (inter["date"] <= instant)].sort_values(["ts", "item_id"])
        return h.tail(MAX_LEN)

    def known_users(self, limit: int = 20) -> List[str]:
        counts = self.p.interactions.groupby("developer").size().sort_values(ascending=False)
        return [str(u) for u in counts.index[:limit]]

    # --------------------------------------------------------------------- main
    @torch.no_grad()
    def recommend(self, username: str, k: int = 10, exclude: Iterable[str] = ()) -> List[Interest]:
        """Top-k items for `username`. Items in `exclude` (already rated by the user in the app)
        are skipped; extra candidates are fetched so that k items are still returned."""
        instant = pd.Timestamp.now(tz="UTC")
        exclude = set(exclude)
        try:
            candidates = self.p.predict(username, k=min(50, k + len(exclude)), as_of=instant.isoformat(),
                                        same_ecosystem=True)
        except ValueError as e:                       # unknown user / empty history
            raise UnknownUserError(str(e)) from e

        hist = self._history(username, instant)
        records = [r for r in candidates if r["item"] not in exclude][:k]
        ids = hist["item_id"].to_numpy(dtype=np.int64) + 1
        ages = ((instant - hist["date"]).dt.total_seconds().to_numpy() / 86400).clip(min=0)
        return self._group(records, ids, ages)

    @torch.no_grad()
    def recommend_items(self, items: List[str], k: int = 10) -> List[Interest]:
        """Top-k items for an ad-hoc stack (e.g. a pasted requirements.txt): `items` are catalogue ids
        such as "pypi:requests", oldest first. They are treated as just adopted (age 0)."""
        id_of = dict(zip(self.p.items["item"], self.p.items["item_id"]))
        ids = np.array([id_of[i] + 1 for i in items if i in id_of], dtype=np.int64)[-MAX_LEN:]
        if len(ids) == 0:
            raise UnknownUserError("None of these packages is in the catalogue")
        items_t = torch.from_numpy(ids[None, :])
        gaps_t = torch.zeros((1, len(ids)), dtype=torch.float32)
        scores = self.p.model.scores(items_t, gaps_t)[0]
        scores[0] = float("-inf")
        scores[torch.from_numpy(ids)] = float("-inf")                       # already in the stack
        allowed = set(self.p.item_eco[ids]) | {"tool"}                       # same ecosystems as the stack
        scores[torch.from_numpy(~np.isin(self.p.item_eco, list(allowed)))] = float("-inf")
        top = scores.topk(min(k, int(torch.isfinite(scores).sum()))).indices.tolist()
        records = []
        for rank, shifted in enumerate(top, start=1):
            row = self.p.item_by_id.loc[shifted - 1]
            url = row.get("repo_url", "")
            records.append({"rank": rank, "item_id": shifted - 1, "item": str(row["item"]),
                            "score": round(float(scores[shifted]), 6),
                            "repo_url": "" if pd.isna(url) else str(url),
                            "summary": "" if pd.isna(row.get("summary")) else str(row["summary"])[:240],
                            "ecosystems": "" if pd.isna(row.get("ecosystems")) else str(row["ecosystems"])})
        return self._group(records, ids, np.zeros(len(ids)))

    @torch.no_grad()
    def _group(self, records: list, ids: np.ndarray, ages: np.ndarray) -> List[Interest]:
        """Assign each recommendation to the interest it matches best and explain each interest by the
        two history items its attention channel looks at most."""
        for new_rank, r in enumerate(records, start=1):   # ranks 1..k after filtering
            r["rank"] = new_rank
        items_t = torch.from_numpy(ids[None, :])
        gaps_t = torch.from_numpy(np.asarray(ages, dtype=np.float32)[None, :])

        z, att = self.p.model.interests(items_t, gaps_t)      # (1,c,d), (1,c,L)
        sim = self.p.model.sim(z)[0]                         # (c, n_items+1)
        att = att[0].numpy()

        # explanation of each interest = the 2 history items it attends to most
        evidence = {}
        for c in range(self.n_channels):
            top = np.argsort(-att[c])[:2]
            evidence[c] = [str(self.p.item_by_id.loc[int(ids[j]) - 1, "item"]) for j in top]

        groups: dict[int, list[RecommendedRepo]] = {c: [] for c in range(self.n_channels)}
        for r in records:
            c = int(sim[:, r["item_id"] + 1].argmax())        # interest this item matches best
            item = r["item"]
            groups[c].append(RecommendedRepo(
                rank=r["rank"], score=r["score"], is_new=False,
                repo=RepoCard(
                    repo_id=item, kind="package",
                    description=r["summary"] or None,
                    topics=sorted(_ecos(r["ecosystems"])),
                    url=r["repo_url"] or _url(item),
                ),
            ))

        out = []
        for c in range(self.n_channels):
            if not groups[c]:
                continue
            names = " & ".join(dict.fromkeys(_short(x) for x in evidence[c]))
            out.append(Interest(
                interest_id=c,
                label=f"Around {names}",
                explanation=f"Because you used {names}",
                evidence=evidence[c],
                recommendations=groups[c],
            ))
        return out
