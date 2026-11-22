"""Reproducible baselines and basic sample-split integrity checks.

Run from the repository root:
    python -m scripts.evaluate_baselines --protocol both

This module does not train or overwrite model checkpoints. It writes only
results under reports/evaluation/. These checks are not a complete leakage audit.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.data import bootstrap_ci, evaluate, load, make_samples, popularity_scorer

DEFAULT_OUTPUT = Path("reports") / "evaluation"


def sample_key(sample):
    """Identify a prediction example by user, query timestamp, and target item."""
    return (int(sample[0]), float(sample[3]), int(sample[4]))


def integrity_checks(samples):
    """Limited structural checks for duplicate targets and future-dated histories."""
    train_keys = {sample_key(s) for s in samples["train"]}
    valid_keys = {sample_key(s) for s in samples["valid"]}
    test_keys = {sample_key(s) for s in samples["test"]}
    future = {}
    for name in ("train", "valid", "test"):
        future[name] = int(sum(np.any(np.asarray(s[2]) > s[3]) for s in samples[name]))
    return {
        "train_valid_duplicate_keys": len(train_keys & valid_keys),
        "train_test_duplicate_keys": len(train_keys & test_keys),
        "valid_test_duplicate_keys": len(valid_keys & test_keys),
        "future_history_samples": future,
        "note": "Only structural sample checks; does not audit feature fitting, text preprocessing, or model selection.",
    }


def evaluate_baselines(interactions, items, samples, protocol, seed=0):
    """Use the project's own evaluator and the same holdout for both baselines."""
    n_items = int(max(interactions["item_id"].max(), items["item_id"].max())) + 1
    result = {}
    # A single RNG across batches matches the project's standard 256-example
    # evaluation batching. Random baseline results are specific to this seed.
    rng = np.random.default_rng(seed)
    scorers = {
        "most_popular": popularity_scorer(interactions, n_items, protocol),
        "random": lambda x, gaps: rng.random((len(x), n_items + 1)),
    }
    for baseline, scorer in scorers.items():
        m, recall, ndcg = evaluate(scorer, samples["test"])
        result[baseline] = {
            "protocol": protocol,
            "baseline": baseline,
            "seed": seed if baseline == "random" else None,
            "recall@20": float(m["recall@20"]),
            "ndcg@10": float(m["ndcg@10"]),
            "n_test": int(m["n"]),
            "ci_recall_95": bootstrap_ci(recall, seed=seed).tolist(),
            "ci_ndcg_95": bootstrap_ci(ndcg, seed=seed).tolist(),
        }
    return result


def generate(protocols, output_dir, seed=0):
    interactions, items = load()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_checks = {}
    for protocol in protocols:
        samples = make_samples(interactions, protocol)
        checks = integrity_checks(samples)
        checks["sample_counts"] = {name: len(rows) for name, rows in samples.items()}
        all_checks[protocol] = checks
        baselines = evaluate_baselines(interactions, items, samples, protocol, seed)
        all_rows.extend(baselines.values())
        for name, metric in baselines.items():
            print(f"{protocol:<8} {name:<13} Recall@20={metric['recall@20']:.4f} "
                  f"NDCG@10={metric['ndcg@10']:.4f} n={metric['n_test']}")
        print(f"{protocol:<8} integrity={checks}")

    with (output_dir / "baseline_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["protocol", "baseline", "seed", "recall@20", "ndcg@10", "n_test", "ci_recall_95", "ci_ndcg_95"])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    (output_dir / "data_checks.json").write_text(
        json.dumps(all_checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Files written to {output_dir}")
    return all_rows, all_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["loo", "temporal", "both"], default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    protocols = ["loo", "temporal"] if args.protocol == "both" else [args.protocol]
    generate(protocols, args.output, args.seed)


if __name__ == "__main__":
    main()
