"""A user's stack: the packages in their history, and reading a pasted requirements.txt / package.json list.

Used by GET /history (the "Your stack" panel) and POST /recommend_stack (recommendations for any list of
packages, not only users present in the dataset).
"""
import json
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .recommender import UnknownUserError

DATA = Path(__file__).resolve().parents[2] / "data" / "processed"
_SPEC = re.compile(r"[\s\[;<>=!~@(,]")          # where a requirement's name ends (version, extras, markers, url)


def normalize(name: str) -> str:
    """PEP 503 style: lower case, runs of '-', '_', '.' become '-'. npm scopes (@org/pkg) are kept."""
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def parse_packages(text: str) -> List[str]:
    """Package names from a requirements.txt, a package.json, or one name per line / comma."""
    text = text.strip()
    if text.startswith("{"):                                       # package.json
        try:
            data = json.loads(text)
            names = []
            for key in ("dependencies", "devDependencies", "peerDependencies"):
                names += list((data.get(key) or {}).keys())
            return [n.strip().lower() for n in names if n.strip()]
        except (ValueError, AttributeError):
            pass
    names = []
    for line in re.split(r"[\n,]", text):
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-") or "://" in line:
            continue                                               # options (-r, -e, --index-url) and urls
        if line.startswith("@"):                                   # npm scope: keep "@org/pkg"
            name = re.split(r"[\s@=<>^~]", line[1:], maxsplit=1)[0]
            names.append("@" + name.lower())
        else:
            names.append(normalize(_SPEC.split(line, maxsplit=1)[0]))
    return [n for n in dict.fromkeys(names) if n and n != "@"]


class Stacks:
    def __init__(self, data_dir: Path = DATA):
        inter = pd.read_csv(data_dir / "interactions.csv",
                            usecols=["developer", "item", "date", "is_dated", "still_in_use"])
        inter["developer"] = inter["developer"].astype(str).str.lower()
        inter = inter.sort_values(["developer", "date"])
        self.by_user = {u: g for u, g in inter.groupby("developer", sort=False)}
        items = pd.read_csv(data_dir / "items.csv", usecols=["item", "members", "summary"])
        self.summary = dict(zip(items["item"], items["summary"].fillna("")))
        # every name a package can be written with -> its catalogue id (a cross-ecosystem tool wins)
        self.lookup = {}
        for item, members in zip(items["item"], items["members"].fillna("")):
            for full in [item] + [m for m in str(members).split("|") if m]:
                eco, _, name = full.partition(":")
                key = name.lower() if name.startswith("@") else normalize(name)
                if item.startswith("tool:") or key not in self.lookup:
                    self.lookup[key] = item

    def history(self, username: str) -> List[dict]:
        g = self.by_user.get(username.lower())
        if g is None:
            raise UnknownUserError(f"Unknown user {username!r}: not present in the dataset")
        return [{"package": r.item, "date": str(r.date)[:10], "dated": bool(r.is_dated),
                 "still_in_use": bool(r.still_in_use), "description": self.summary.get(r.item, "") or None}
                for r in g.itertuples(index=False)]

    def match(self, names: List[str]) -> Tuple[List[str], List[str]]:
        """Catalogue ids for the names we know, and the names we don't."""
        known, unknown = [], []
        for n in names:
            item = self.lookup.get(n)
            (known if item else unknown).append(item or n)
        return list(dict.fromkeys(known)), unknown
