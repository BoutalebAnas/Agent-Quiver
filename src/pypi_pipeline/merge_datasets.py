"""Merge the two team datasets into one final dataset (data/combined/).

Source A, "pypi_projects" (src/pypi_pipeline/): user = a PyPI project, interaction = the project
declares a dependency (cleaned: metadata gaps, dated adoptions, persistent drops).
Source B (optional: without --github-dir the dataset is PyPI projects only), "github_devs" (team dataset, data/external/github_devs/): user = a GitHub
developer, interaction = the developer starts using an npm/PyPI package in their repos
(the team's GitHub dataset: items.csv, interactions.csv, stars.csv, forks.csv).

One schema for both:
- item ids: "pypi:<name>", "npm:<name>", or "tool:<name>" when the same tool exists in several
  ecosystems (source B's canonical mapping, e.g. tool:openai = pypi:openai + npm:openai).
  The mapping is applied to source A too, so pypi:openai is tool:openai everywhere.
- user ids: "pypi:<project>" or "gh:<developer>".
- one global temporal split for every interaction (the protocol of our reported results):
  train < 2025-01-01 <= valid < 2025-07-01 <= test < 2026-11-01 <= after_test.
- dated = the date is the real adoption date. Source A's initial dependency sets and the
  adoptions right after a metadata gap are not dated: use them as history, never as targets.

    python -m src.pypi_pipeline.merge_datasets --github-dir <folder with the GitHub dataset CSVs>
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from src.pypi_pipeline import pypi_events as D  # noqa: E402

PIPE = Path(__file__).resolve().parent
ROOT = D.ROOT                                        # repository root

OUT = ROOT / "data" / "interim" / "combined"         # git-ignored; data/processed is built from it
CUTS = {"valid": "2025-01-01", "test": "2025-07-01", "after_test": "2026-11-01"}


def split_of(date):
    if date < CUTS["valid"]:
        return "train"
    if date < CUTS["test"]:
        return "valid"
    if date < CUTS["after_test"]:
        return "test"
    return "after_test"


def iso(t):
    """Both sources -> 'YYYY-MM-DDTHH:MM:SS' in UTC."""
    return pd.Timestamp(t).tz_localize("UTC").isoformat()[:19] if pd.Timestamp(t).tzinfo is None \
        else pd.Timestamp(t).tz_convert("UTC").isoformat()[:19]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--github-dir", default=None,
                    help="optional: items.csv, interactions.csv, stars.csv, forks.csv of the GitHub developers; "
                         "without it the dataset is PyPI projects only")
    args = ap.parse_args()
    gh = Path(args.github_dir) if args.github_dir else None
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- source B: GitHub developers ----------------
    if gh is not None:
        b_items = pd.read_csv(gh / "items.csv")
        b_inter = pd.read_csv(gh / "interactions.csv")
    else:
        b_items = pd.DataFrame(columns=["item", "members", "ecosystems", "summary", "is_ai"])
        b_inter = pd.DataFrame(columns=["developer", "item", "date", "still_in_use", "n_repos"])
    canon = {}                                   # member -> canonical item id
    for _, r in b_items.iterrows():
        for m in str(r["members"]).split("|"):
            canon[m] = r["item"]
    ai_items = set(b_items.loc[b_items["is_ai"].astype(bool), "item"]) if len(b_items) else set()

    rows = []
    for _, r in b_inter.iterrows():
        rows.append({"user": f"gh:{r['developer']}", "user_type": "developer", "source": "github_devs",
                     "item": r["item"], "date": iso(r["date"]), "dated": True,
                     "still_in_use": bool(r["still_in_use"]), "n_repos": int(r["n_repos"])})

    # ---------------- source A: PyPI projects (cleaned) ----------------
    projects = D.load_projects()
    for p in projects.values():
        adoptions, drops, _ = D.events_of(p)
        valid = [d for _, _, d in p.releases if d is not None]
        last = valid[-1] if valid else frozenset()
        for t, dep, not_dated in adoptions:
            item = canon.get(f"pypi:{dep}", f"pypi:{dep}")
            rows.append({"user": f"pypi:{p.name}", "user_type": "project", "source": "pypi_projects",
                         "item": item, "date": iso(t), "dated": not not_dated,
                         "still_in_use": dep in last, "n_repos": 1})

    inter = pd.DataFrame(rows)
    # one row per (user, item): the earliest adoption (two pypi names can map to one tool)
    inter = inter.sort_values(["user", "item", "date", "dated"], ascending=[True, True, True, False])
    n_before = len(inter)
    inter = inter.drop_duplicates(["user", "item"], keep="first")
    merged_dupes = n_before - len(inter)
    inter["is_ai"] = inter["item"].isin(ai_items)
    inter["split"] = inter["date"].map(split_of)

    # ---------------- users and items ----------------
    users = (inter.groupby("user").agg(user_type=("user_type", "first"), source=("source", "first"),
                                       n_interactions=("item", "size"), first_date=("date", "min"),
                                       last_date=("date", "max")).reset_index())
    users.insert(0, "user_id", range(len(users)))
    uid = dict(zip(users["user"], users["user_id"]))

    b_meta = b_items.set_index("item")
    summaries = {f"pypi:{p.name}": p.summary for p in projects.values()}
    item_rows = []
    for item, g in inter.groupby("item"):
        eco = item.split(":", 1)[0]
        if item in b_meta.index:
            m = b_meta.loc[item]
            members, ecos, summary = m["members"], m["ecosystems"], m["summary"]
        else:
            members, ecos, summary = item, eco, summaries.get(item, "")
        item_rows.append({"item": item, "members": members, "ecosystems": ecos,
                          "summary": "" if pd.isna(summary) else str(summary)[:300],
                          "is_ai": item in ai_items,
                          "n_users": g["user"].nunique(),
                          "n_projects": int((g["source"] == "pypi_projects").sum()),
                          "n_developers": int((g["source"] == "github_devs").sum()),
                          "in_both_sources": g["source"].nunique() == 2})
    items = pd.DataFrame(item_rows).sort_values("item").reset_index(drop=True)
    items.insert(0, "item_id", range(len(items)))
    iid = dict(zip(items["item"], items["item_id"]))

    inter.insert(0, "user_id", inter["user"].map(uid))
    inter.insert(1, "item_id", inter["item"].map(iid))
    inter = inter.sort_values(["user_id", "date", "item_id"]).reset_index(drop=True)
    cols = ["user_id", "item_id", "user", "item", "date", "dated", "split", "source", "user_type",
            "is_ai", "still_in_use", "n_repos"]
    inter = inter[cols]

    # ---------------- migrations (source A, cleaned + manual check) ----------------
    table, audit = D.migration_table(projects, "2100-01-01", with_audit=True)
    manual = pd.read_csv(PIPE / "migration_manual_check.csv")
    kept = [r for r in audit if r["kept"]]
    mig = pd.DataFrame(kept, columns=["from", "to", "support", "lift", "coexist", "inlined", "kept"])[
        ["from", "to", "support", "lift", "coexist"]]
    mig = mig.merge(manual[["from", "to", "verdict", "note"]], on=["from", "to"], how="left")
    mig["verdict"] = mig["verdict"].fillna("not_checked")      # new rules from fresh data have no manual verdict yet
    mig["from"] = mig["from"].map(lambda x: canon.get(f"pypi:{x}", f"pypi:{x}"))
    mig["to"] = mig["to"].map(lambda x: canon.get(f"pypi:{x}", f"pypi:{x}"))
    mig = mig.sort_values(["support", "lift"], ascending=False)

    # ---------------- side information from source B ----------------
    for name in (("stars", "forks") if gh is not None else ()):
        side = pd.read_csv(gh / f"{name}.csv").drop(columns=["user_id"], errors="ignore")
        side.insert(0, "user_id", ("gh:" + side["developer"].astype(str)).map(uid))
        side = side[side["user_id"].notna()].astype({"user_id": int})
        side.to_csv(OUT / f"github_{name}.csv", index=False)

    # ---------------- checks (fail loudly rather than ship a wrong file) ----------------
    assert inter["user_id"].notna().all() and inter["item_id"].notna().all()
    assert not inter.duplicated(["user_id", "item_id"]).any()
    assert set(inter["item"]) == set(items["item"]) and set(inter["user"]) == set(users["user"])
    assert items["item"].is_unique and users["user"].is_unique
    assert (users.set_index("user")["n_interactions"] == inter.groupby("user").size()).all()
    assert inter["date"].str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$").all()
    assert set(inter["split"]) <= {"train", "valid", "test", "after_test"}
    assert set(mig["from"]) | set(mig["to"]) <= set(iid), "migration package missing from items"
    b_pairs = len(b_inter.drop_duplicates(["developer", "item"]))
    assert (inter["source"] == "github_devs").sum() == b_pairs, "lost GitHub interactions"
    assert len(inter) > 0, "no interactions: is data/raw/pypi/pypi_histories_repaired.jsonl.gz there?"

    inter.to_csv(OUT / "interactions.csv", index=False)
    users.to_csv(OUT / "users.csv", index=False)
    items.to_csv(OUT / "items.csv", index=False)
    mig.to_csv(OUT / "migrations.csv", index=False)

    by_src = inter.groupby("source")
    info = {
        "description": "AgentQuiver final dataset: PyPI projects x dependencies + GitHub developers x npm/PyPI packages",
        "split_cuts": CUTS,
        "counts": {
            "users": len(users), "items": len(items), "interactions": len(inter),
            "users_by_type": users["user_type"].value_counts().to_dict(),
            "interactions_by_source": by_src.size().to_dict(),
            "dated_interactions_by_source": by_src["dated"].sum().astype(int).to_dict(),
            "items_by_ecosystem": Counter(i.split(":", 1)[0] for i in items["item"]),
            "items_in_both_sources": int(items["in_both_sources"].sum()),
            "ai_items": int(items["is_ai"].sum()), "ai_interactions": int(inter["is_ai"].sum()),
            "split": inter["split"].value_counts().to_dict(),
            "dated_split": inter[inter["dated"]]["split"].value_counts().to_dict(),
            "duplicate_user_item_pairs_merged": int(merged_dupes),
            "migration_rules": len(mig), "migration_rules_verified_replacement":
                int(mig["verdict"].fillna("").str.startswith("replacement").sum()),
            "density_%": round(100 * len(inter) / (len(users) * len(items)), 4),
        },
        "pypi_projects_cleaning": D.summary_stats(projects),
    }
    (OUT / "dataset_info.json").write_text(json.dumps(info, indent=1, default=int))
    shutil.copy(PIPE / "DATASET.md", OUT / "README.md")
    print(json.dumps(info["counts"], indent=1, default=int))


if __name__ == "__main__":
    main()
