"""Save real training logs and plots without overwriting tracked checkpoints.

python -m scripts.training_curves --run --protocol loo --c 4 --seed 0 --no_ranker

Training runs with cwd outside the Git repository, so src.train's checkpoint
is written under Downloads/AgentQuiver_training/<run>/models/, not in the Git repository.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "training"
_PATTERN_RETR = re.compile(r"^\[retr\] ep(\d+) loss ([\d.eE+-]+) valid R@20 ([\d.eE+-]+) NDCG@10 ([\d.eE+-]+)$")
_PATTERN_RANK = re.compile(r"^\[rank\] ep(\d+) valid R@20 ([\d.eE+-]+) NDCG@10 ([\d.eE+-]+)$")


def parse_log(content: str) -> list[dict]:
    """Parse only genuine text from src.train; do not interpolate missing values."""
    rows = []
    for line in content.splitlines():
        m = _PATTERN_RETR.fullmatch(line.strip())
        if m:
            ep, loss, recall, ndcg = m.groups()
            rows.append({"stage": "retriever", "epoch": int(ep), "train_loss": float(loss),
                         "valid_recall@20": float(recall), "valid_ndcg@10": float(ndcg)})
            continue
        m = _PATTERN_RANK.fullmatch(line.strip())
        if m:
            ep, recall, ndcg = m.groups()
            rows.append({"stage": "ranker", "epoch": int(ep), "train_loss": "",
                         "valid_recall@20": float(recall), "valid_ndcg@10": float(ndcg)})
    return rows


def record_training(protocol: str, c: int, seed: int, logfile: Path, no_ranker: bool = False):
    tag = "_noranker" if no_ranker else ""
    run_dir = Path.home() / "Downloads" / "AgentQuiver_training" / f"{protocol}_c{c}_seed{seed}{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-u", "-m", "src.train", "--protocol", protocol, "--c", str(c), "--seed", str(seed)]
    if no_ranker:
        cmd.append("--no_ranker")      # final system = retriever only
    print("Training in:", run_dir, flush=True)
    print("Git checkpoint remains unchanged; new checkpoint is saved in this external directory.", flush=True)
    with logfile.open("w", encoding="utf-8") as f:
        with subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace") as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
            exit_code = proc.wait()
    if exit_code:
        raise RuntimeError(f"Training failed (exit {exit_code}). See {logfile}")
    return run_dir


def export_curves(rows: list[dict], out: Path):
    if not any(row["stage"] == "retriever" for row in rows):
        raise ValueError("No retriever training lines found; curves cannot be claimed")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "training_curves.csv"
    keys = ["stage", "epoch", "train_loss", "valid_recall@20", "valid_ndcg@10"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    retr = [r for r in rows if r["stage"] == "retriever"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["epoch"] for r in retr], [r["train_loss"] for r in retr], marker="o")
    ax.set(xlabel="Epoch", ylabel="Sum of batch training losses", title="Retriever training loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "retriever_training_loss.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for stage in ("retriever", "ranker"):
        stage_rows = [r for r in rows if r["stage"] == stage]
        if stage_rows:
            ax.plot([r["epoch"] for r in stage_rows],
                    [r["valid_ndcg@10"] for r in stage_rows], marker="o", label=stage)
    ax.set(xlabel="Epoch (within each stage)", ylabel="Validation NDCG@10",
           title="Real validation NDCG@10 across training stages")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "validation_ndcg.png", dpi=160)
    plt.close(fig)
    return csv_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true", help="Train and record logs from scratch")
    p.add_argument("--log", help="Parse an existing *real* training log without retraining")
    p.add_argument("--protocol", choices=["loo", "temporal"], default="loo")
    p.add_argument("--c", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_ranker", action="store_true", help="Train the retriever only (final system)")
    p.add_argument("--output", default=str(OUT))
    a = p.parse_args()
    out = Path(a.output).resolve()
    logfile = Path(a.log).resolve() if a.log else out / f"training_{a.protocol}_c{a.c}_seed{a.seed}.log"
    if a.run:
        record_training(a.protocol, a.c, a.seed, logfile, a.no_ranker)
    elif not logfile.exists():
        p.error("No log to parse. Use --run to train and capture real results.")
    rows = parse_log(logfile.read_text(encoding="utf-8"))
    print("Parsed stages:", {stage: sum(r["stage"] == stage for r in rows) for stage in ("retriever", "ranker")})
    print("Saved:", export_curves(rows, out))
    print("Saved:", out / "retriever_training_loss.png")
    print("Saved:", out / "validation_ndcg.png")


if __name__ == "__main__":
    main()
