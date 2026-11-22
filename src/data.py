"""Data loading + evaluation for AgentQuiver (MLE-Core). No torch needed here."""
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"
MAX_LEN = 50


def load(data_dir=DATA):
    inter = pd.read_csv(Path(data_dir) / "interactions.csv", parse_dates=["date"])
    items = pd.read_csv(Path(data_dir) / "items.csv")
    inter["day"] = (inter["date"] - inter["date"].min()).dt.total_seconds() / 86400.0
    inter = inter.sort_values(["user_id", "date", "item_id"]).reset_index(drop=True)   # chronological, does not rely on ts
    return inter, items


def make_samples(inter, protocol="loo"):
    """Return dict with train / valid / test lists of samples.
    sample = (user, hist_items, hist_days, query_day, target_item, seen_set)
    Item ids are shifted by +1 so that 0 = padding."""
    col = f"split_{'loo' if protocol == 'loo' else 'temporal'}"
    out = {"train": [], "valid": [], "test": []}
    for u, g in inter.groupby("user_id"):
        it = g["item_id"].to_numpy() + 1
        dy = g["day"].to_numpy()
        sp = g[col].to_numpy()
        tr = np.where(sp == "train")[0]
        va = np.where(sp == "valid")[0]
        te = np.where(sp == "test")[0]
        # training: predict each train item from the train items before it
        for k in range(1, len(tr)):
            h = tr[:k][-MAX_LEN:]
            out["train"].append((u, it[h], dy[h], dy[tr[k]], it[tr[k]], None))
        # valid: history = train; test: history = train + valid
        for name, tgt, base in (("valid", va, tr), ("test", te, np.concatenate([tr, va]))):
            if len(base) == 0:
                continue
            for t in tgt:
                b = base[dy[base] <= dy[t]]          # history = what the user had BEFORE the target (no look-ahead)
                if len(b) == 0:
                    continue
                h = b[-MAX_LEN:]
                out[name].append((u, it[h], dy[h], dy[t], it[t], set(it[b].tolist())))
    return out


def pad_batch(batch):
    B = len(batch)
    L = max(len(s[1]) for s in batch)
    items = np.zeros((B, L), dtype=np.int64)
    gaps = np.zeros((B, L), dtype=np.float32)
    for i, s in enumerate(batch):
        n = len(s[1])
        items[i, L - n:] = s[1]                      # left padding
        gaps[i, L - n:] = np.maximum(s[3] - s[2], 0)  # age in days of each event
    tgt = np.array([s[4] for s in batch], dtype=np.int64)
    return items, gaps, tgt


def metrics_from_scores(scores, targets, seen_list, k_r=20, k_n=10):
    """scores: (B, n_items+1) numpy. Column 0 = padding. Seen items are masked."""
    scores = scores.copy()
    scores[:, 0] = -np.inf
    for i, seen in enumerate(seen_list):
        if seen:
            idx = [j for j in seen if j != targets[i]]
            scores[i, idx] = -np.inf
    tscore = scores[np.arange(len(targets)), targets]
    rank = (scores > tscore[:, None]).sum(1)          # 0 = best
    rec = (rank < k_r).astype(float)
    ndcg = np.where(rank < k_n, 1.0 / np.log2(rank + 2), 0.0)
    return rec, ndcg


def evaluate(score_fn, samples, in_train_flags=None, bs=256):
    """score_fn(items, gaps) -> (B, n_items+1) scores. Returns Recall@20, NDCG@10 (+ per-sample arrays)."""
    R, N = [], []
    for i in range(0, len(samples), bs):
        b = samples[i:i + bs]
        items, gaps, tgt = pad_batch(b)
        r, n = metrics_from_scores(score_fn(items, gaps), tgt, [s[5] for s in b])
        R.append(r); N.append(n)
    R, N = np.concatenate(R), np.concatenate(N)
    return {"recall@20": R.mean(), "ndcg@10": N.mean(), "n": len(R)}, R, N


def bootstrap_ci(x, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def popularity_scorer(inter, n_items, protocol="loo"):
    pop = np.zeros(n_items + 1)
    col = "split_loo" if protocol == "loo" else "split_temporal"
    tr = inter[inter[col] == "train"]       # popularity must come from the train part of the SAME protocol
    np.add.at(pop, tr["item_id"].to_numpy() + 1, 1)
    return lambda items, gaps: np.tile(pop, (len(items), 1))


if __name__ == "__main__":
    import sys
    inter, items = load(sys.argv[1] if len(sys.argv) > 1 else DATA)
    n_items = int(inter["item_id"].max()) + 1
    for proto in ("loo", "temporal"):
        S = make_samples(inter, proto)
        print(proto, {k: len(v) for k, v in S.items()})
        if S["test"]:
            m, r, _ = evaluate(popularity_scorer(inter, n_items, proto), S["test"])
            print("  popularity test:", {k: round(float(v), 4) for k, v in m.items()},
                  "CI recall", bootstrap_ci(r).round(3))
