"""Runs the plan's ablations on both protocols, 3 seeds, writes results.csv. Diffusion enters the product only if it beats 'full' on NDCG@10 (validation rule from the plan; here we report test)."""
import numpy as np, pandas as pd
from .train import run

CONFIGS = {"full": {}, "one_interest": {"c": 1}, "no_age(no action/time)": {"no_age": True},
           "no_text": {"no_text": True}, "diffusion": {"diffusion": True},
           "retriever_only": {"use_ranker": False}}
rows = []
import sys
RETR = "retr" in sys.argv                    # python -m src.ablate retr  -> ablations WITHOUT the ranker (final system)
PROTOS = [x for x in sys.argv[1:] if x != "retr"] or ["loo"]   # python -m src.ablate loo temporal
if RETR:
    CONFIGS = {"retr_full": {}, "retr_one_interest": {"c": 1}, "retr_no_age": {"no_age": True},
               "retr_no_text": {"no_text": True}, "retr_diffusion": {"diffusion": True}}
    CONFIGS = {k: {**v, "use_ranker": False} for k, v in CONFIGS.items()}
OUTCSV = "results_retr.csv" if RETR else "results.csv"
for proto in PROTOS:
    for name, cfg in CONFIGS.items():
        res = [run(proto, seed=s, verbose=False, **cfg)[0] for s in range(2)]
        rows.append({"protocol": proto, "config": name,
                     **{k: float(np.nanmean([r[k] for r in res])) for k in ("recall@20", "ndcg@10", "cold_recall@20")},
                     "ndcg_std": float(np.std([r["ndcg@10"] for r in res]))})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUTCSV, index=False)   # saved after every config
pd.DataFrame(rows).to_csv(OUTCSV, index=False)
