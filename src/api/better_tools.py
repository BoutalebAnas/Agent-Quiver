"""'Better tools': replacements that other projects actually made (data/processed/migrations.csv).

A rule A -> B comes from PyPI release histories: a release drops A and adopts B (see src/pypi_pipeline/).
Only rules checked by hand as real replacements are served. A user gets the suggestion when A is still
in use in its history and B has never been adopted; for each A, only the replacement most projects chose.
"""
from pathlib import Path
from typing import List

import pandas as pd

from .recommender import UnknownUserError

DATA = Path(__file__).resolve().parents[2] / "data" / "processed"


class BetterTools:
    def __init__(self, data_dir: Path = DATA):
        rules = pd.read_csv(data_dir / "migrations.csv")
        self.rules = rules[rules["verdict"] == "replacement"].sort_values(["support", "lift"], ascending=False)
        inter = pd.read_csv(data_dir / "interactions.csv", usecols=["developer", "item", "still_in_use"])
        inter["developer"] = inter["developer"].astype(str).str.lower()
        self.history = inter.groupby("developer")["item"].apply(set).to_dict()
        self.in_use = inter[inter["still_in_use"].astype(bool)].groupby("developer")["item"].apply(set).to_dict()

    def suggest(self, username: str) -> List[dict]:
        user = username.lower()
        if user not in self.history:
            raise UnknownUserError(f"Unknown user {username!r}: not present in the dataset")
        history, in_use = self.history[user], self.in_use.get(user, set())
        out, done = [], set()
        for r in self.rules.itertuples(index=False):           # columns: from, to, support, lift, ...
            # one suggestion per package: the replacement most projects chose
            if r[0] in in_use and r[0] not in done and r[1] not in history:
                done.add(r[0])
                out.append({"replace": r[0], "with_": r[1], "projects": int(r.support),
                            "lift": float(r.lift), "note": "" if pd.isna(r.note) else str(r.note)})
        return out
