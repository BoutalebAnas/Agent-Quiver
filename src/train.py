"""Usage (from the project root):
  python -m src.train --protocol loo                 # full model (retriever + ranker)
  python -m src.train --protocol loo --diffusion     # + diffusion
  python -m src.ablate                               # all ablations, 3 seeds
Not run by the author (no torch in the sandbox) -> send me any traceback."""
import argparse, numpy as np, torch, torch.nn.functional as F
from .data import load, make_samples, pad_batch, metrics_from_scores, bootstrap_ci
from .core_model import AgentQuiver, Ranker


def text_matrix(items, n_items, dim=32):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    txt = items["summary"].fillna("") + " " + items["keywords"].fillna("") + " " + items["classifiers"].fillna("")
    X = TfidfVectorizer(min_df=2, stop_words="english").fit_transform(txt)
    Z = TruncatedSVD(min(dim, X.shape[1] - 1), random_state=0).fit_transform(X)
    M = np.zeros((n_items + 1, Z.shape[1]), dtype=np.float32)
    M[items["item_id"].to_numpy() + 1] = Z
    return torch.tensor(M)


def batches(samples, bs, shuffle):
    idx = np.random.permutation(len(samples)) if shuffle else np.arange(len(samples))
    for i in range(0, len(idx), bs):
        yield [samples[j] for j in idx[i:i + bs]]


def tens(b, no_age=False):
    it, gp, tg = pad_batch(b)
    if no_age: gp = gp * 0
    return torch.tensor(it), torch.tensor(gp), torch.tensor(tg)


def seen_mask(batch, n):
    m = torch.zeros(len(batch), n, dtype=torch.bool)
    for i, s in enumerate(batch):
        if s[5]:
            idx = [j for j in s[5] if j != s[4]]
            m[i, idx] = True
    return m


@torch.no_grad()
def full_eval(model, ranker, samples, K=100, no_age=False):
    """Retrieval top-K (seen items removed) -> ranker re-order. Returns per-sample recall/ndcg arrays."""
    model.eval(); ranker and ranker.eval()
    R, N = [], []
    for b in batches(samples, 128, False):
        it, gp, tg = tens(b, no_age)
        s = model.scores(it, gp); s[:, 0] = -1e9
        s[seen_mask(b, s.shape[1])] = -1e9
        if ranker is None:
            final = s
        else:
            top = s.topk(K, 1).indices                        # (B,K)
            rs = ranker(it, gp, top)
            final = torch.full_like(s, -1e9)
            final.scatter_(1, top, rs)
        r, n = metrics_from_scores(final.numpy(), tg.numpy(), [x[5] for x in b])
        R.append(r); N.append(n)
    return np.concatenate(R), np.concatenate(N)


def ranker_auc(model, ranker, samples, n_neg=99):
    """AUC of the ranker: positive vs random negatives."""
    model.eval(); ranker.eval(); a = []
    n_items = model.enc.item.num_embeddings
    with torch.no_grad():
        for b in batches(samples, 128, False):
            it, gp, tg = tens(b)
            neg = torch.randint(1, n_items, (len(b), n_neg))
            sc = ranker(it, gp, torch.cat([tg[:, None], neg], 1))
            a.append((sc[:, :1] > sc[:, 1:]).float().mean(1))
    return torch.cat(a).mean().item()


def run(protocol="loo", c=4, d=64, epochs=60, no_text=False, no_age=False, diffusion=False,
        seed=0, use_ranker=True, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    inter, items = load()
    n_items = int(max(inter["item_id"].max(), items["item_id"].max())) + 1
    S = make_samples(inter, protocol)
    train_items = {s[4] for s in S["train"]} | {i for s in S["train"] for i in s[1]}
    tv = None if no_text else text_matrix(items, n_items)
    model = AgentQuiver(n_items, d, c, tv, diffusion=diffusion)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-2)
    best, bad, state = -1, 0, None
    for ep in range(epochs):                                   # ---- stage 1: retriever
        model.train(); tot = 0
        for b in batches(S["train"], 128, True):
            loss = model.loss(*tens(b, no_age)); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        R, N = full_eval(model, None, S["valid"], no_age=no_age)
        if verbose: print(f"[retr] ep{ep:02d} loss {tot:.2f} valid R@20 {R.mean():.3f} NDCG@10 {N.mean():.3f}")
        if N.mean() > best: best, bad, state = N.mean(), 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 8: break
    model.load_state_dict(state)
    ranker = None
    if use_ranker:                                             # ---- stage 2: ranker (retriever frozen)
        ranker = Ranker(model, d, c); ropt = torch.optim.AdamW(
            [p for n, p in ranker.named_parameters() if not n.startswith("r.")], 1e-3, weight_decay=1e-2)
        rbest, rbad, rstate = -1, 0, None
        for ep in range(30):
            ranker.train(); model.eval()
            for b in batches(S["train"], 64, True):
                it, gp, tg = tens(b, no_age)
                with torch.no_grad():
                    s = model.scores(it, gp); s[:, 0] = -1e9; s[torch.arange(len(b)), tg] = -1e9
                    hard = s.topk(15, 1).indices
                cands = torch.cat([tg[:, None], hard, torch.randint(1, n_items, (len(b), 15))], 1)
                loss = F.cross_entropy(ranker(it, gp, cands), torch.zeros(len(b), dtype=torch.long))
                ropt.zero_grad(); loss.backward(); ropt.step()
            R, N = full_eval(model, ranker, S["valid"], no_age=no_age)
            if verbose: print(f"[rank] ep{ep:02d} valid R@20 {R.mean():.3f} NDCG@10 {N.mean():.3f}")
            if N.mean() > rbest: rbest, rbad, rstate = N.mean(), 0, {k: v.clone() for k, v in ranker.state_dict().items()}
            else:
                rbad += 1
                if rbad >= 6: break
        ranker.load_state_dict(rstate)
    R, N = full_eval(model, ranker, S["test"], no_age=no_age)
    cold = np.array([s[4] not in train_items for s in S["test"]])
    out = {"recall@20": R.mean(), "ndcg@10": N.mean(), "ci_recall": bootstrap_ci(R), "ci_ndcg": bootstrap_ci(N),
           "n_test": len(R), "n_cold": int(cold.sum()), "cold_recall@20": R[cold].mean() if cold.any() else float("nan"),
           "coverage": None}
    if ranker is not None: out["auc"] = ranker_auc(model, ranker, S["test"])
    import os
    os.makedirs("models", exist_ok=True)
    torch.save({"model": model.state_dict(), "ranker": ranker.state_dict() if ranker else None,
                "cfg": dict(protocol=protocol, c=c, d=d, diffusion=diffusion, no_text=no_text)},
               f"models/ckpt_{protocol}_c{c}{'_diff' if diffusion else ''}.pt")
    return out, model, ranker, S


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="loo"); ap.add_argument("--c", type=int, default=4)
    ap.add_argument("--no_text", action="store_true"); ap.add_argument("--no_age", action="store_true")
    ap.add_argument("--diffusion", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_ranker", action="store_true")
    a = ap.parse_args()
    o, *_ = run(a.protocol, a.c, no_text=a.no_text, no_age=a.no_age, diffusion=a.diffusion,
                seed=a.seed, use_ranker=not a.no_ranker)
    print("TEST", {k: (np.round(v, 4) if v is not None else None) for k, v in o.items()})
