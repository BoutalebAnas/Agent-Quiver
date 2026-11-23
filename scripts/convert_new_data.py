"""Convert the merged dataset (PyPI projects + GitHub developers) to the format expected by src/data.py.
Usage:  python -m scripts.convert_new_data data/interim/combined data/processed keep_undated
        (input = output of python -m src.pypi_pipeline.merge_datasets)
  default        : keep only rows with dated==True (reliable adoption date)
  keep_undated   : also keep undated rows (e.g. initial dependencies) as TRAIN-ONLY history;
                   validation/test targets are always dated rows.
It rebuilds split_loo and split_temporal itself (it does NOT use the old `split` column) and prints diagnostics."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
KEEP_UNDATED = "keep_undated" in sys.argv[3:]
dst.mkdir(parents=True, exist_ok=True)
inter = pd.read_csv(src / "interactions.csv")
items = pd.read_csv(src / "items.csv")
print("raw interactions:", len(inter), "| users:", inter.user_id.nunique(), "| items:", inter.item_id.nunique())
for c in ("split", "dated", "source", "user_type"):
    if c in inter.columns:
        print(f"\n{c} value counts:\n", inter[c].value_counts(dropna=False).head(10).to_string())

inter["date"] = pd.to_datetime(inter["date"], utc=True, errors="coerce")
n0 = len(inter)
inter = inter[inter["date"].notna()]
inter["is_dated"] = True
if "dated" in inter.columns:
    ok = inter["dated"].astype(str).str.lower().isin(["true", "1", "yes", "1.0"])
    if ok.sum() > 0:
        inter["is_dated"] = ok
        if not KEEP_UNDATED:
            inter = inter[ok]
print(f"\nkept {len(inter)}/{n0} rows (keep_undated={KEEP_UNDATED}; dated rows: {int(inter['is_dated'].sum())})")

inter = inter.sort_values(["user_id", "date", "item_id"]).drop_duplicates(["user_id", "item_id"], keep="first")
cnt = inter.groupby("user_id")["item_id"].transform("size")
ndat = inter.groupby("user_id")["is_dated"].transform("sum")
bad = (cnt < 3) | (ndat < 2)          # need >= 2 dated events (valid + test target) and >= 3 events overall
print("users dropped (<3 events or <2 dated events):", inter[bad].user_id.nunique())
inter = inter[~bad].copy()

# contiguous ids
uids = {u: i for i, u in enumerate(sorted(inter.user_id.unique()))}
iids = {t: i for i, t in enumerate(sorted(inter.item_id.unique()))}
inter["user_id"] = inter["user_id"].map(uids); inter["item_id"] = inter["item_id"].map(iids)
inter = inter.sort_values(["user_id", "date", "item_id"]).reset_index(drop=True)
inter["ts"] = ((inter["date"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds() // 3600).astype(int)   # hours since 1970, independent of the pandas datetime unit

# leave-last-out
inter["split_loo"] = "train"
sub = inter[inter["is_dated"]]                      # targets are always dated rows
rfe = sub.groupby("user_id").cumcount(ascending=False)
inter.loc[sub.index[rfe == 0], "split_loo"] = "test"
inter.loc[sub.index[rfe == 1], "split_loo"] = "valid"
# temporal: 70% / 15% / 15% of the date range (by quantile)
t_valid, t_test = inter["date"].quantile(0.70), inter["date"].quantile(0.85)
inter["split_temporal"] = np.where(inter["date"] < t_valid, "train", np.where(inter["date"] < t_test, "valid", "test"))
# Undated rows never become targets. Before T_VALID they are train history; from T_VALID on they are
# left out of the temporal protocol ("none", ignored by src/data.py), otherwise the model would train on
# dependencies only observed after the cut (leakage).
undated = ~inter["is_dated"]
inter.loc[undated & (inter["date"] < t_valid), "split_temporal"] = "train"
inter.loc[undated & (inter["date"] >= t_valid), "split_temporal"] = "none"
print("undated rows left out of the temporal protocol (date >= T_VALID):", int((undated & (inter["date"] >= t_valid)).sum()))
print("\nT_VALID =", t_valid, "| T_TEST =", t_test)
tr_items_loo = set(inter.loc[inter.split_loo == "train", "item_id"]); tr_items_t = set(inter.loc[inter.split_temporal == "train", "item_id"])
inter["item_in_train_loo"] = inter["item_id"].isin(tr_items_loo); inter["item_in_train_temporal"] = inter["item_id"].isin(tr_items_t)
inter["developer"] = inter["user"] if "user" in inter.columns else inter["user_id"]

items = items[items["item_id"].isin(iids)].copy()
items["item_id"] = items["item_id"].map(iids)
items = items.sort_values("item_id").reset_index(drop=True)
for c in ("summary", "keywords", "classifiers"):
    if c not in items.columns: items[c] = ""
n_int = inter.groupby("item_id").size()
items["n_interactions"] = items["item_id"].map(n_int).fillna(0).astype(int)
missing = set(range(len(iids))) - set(items["item_id"])
print("items missing from items.csv:", len(missing), "| empty summary:", int((items["summary"].fillna("") == "").sum()))

inter.to_csv(dst / "interactions.csv", index=False); items.to_csv(dst / "items.csv", index=False)
print("\nFINAL:", len(inter), "interactions |", inter.user_id.nunique(), "users |", len(items), "items |",
      f"sparsity {1 - len(inter) / (inter.user_id.nunique() * len(items)):.4f}")
print("loo:", inter.split_loo.value_counts().to_dict()); print("temporal:", inter.split_temporal.value_counts().to_dict())
print("test targets not in train (temporal):", int((~inter[inter.split_temporal == 'test'].item_in_train_temporal).sum()))
