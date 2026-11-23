"""Export the cleaned PyPI data as CSV files (data/interim/pypi/), without the GitHub part.

    python -m src.pypi_pipeline.export_pypi

Outputs
- interactions.csv : project, package, date, dated, still_in_use (one row per project x package, first adoption)
- migrations.csv   : the "better tool" rules kept (A -> B, support, lift, coexistence) + manual verdict
- migration_audit.csv : every candidate pair with its scores and whether it was kept
- stats.json       : counts after cleaning
"""
import csv
import json
import sys
from pathlib import Path

from src.pypi_pipeline import pypi_events as D  # noqa: E402

PIPE = Path(__file__).resolve().parent
OUT = D.ROOT / "data" / "interim" / "pypi"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    projects = D.load_projects()
    with open(OUT / "interactions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["project", "package", "date", "dated", "still_in_use"])
        n = 0
        for p in sorted(projects.values(), key=lambda p: p.name):
            adoptions, _, _ = D.events_of(p)
            valid = [d for _, _, d in p.releases if d is not None]
            last = valid[-1] if valid else frozenset()
            for t, dep, not_dated in adoptions:
                w.writerow([p.name, dep, t, not not_dated, dep in last])
                n += 1

    table, audit = D.migration_table(projects, "2100-01-01", with_audit=True)
    manual = {}
    with open(PIPE / "migration_manual_check.csv") as f:
        for r in csv.DictReader(f):
            manual[(r["from"], r["to"])] = (r["verdict"], r["note"])
    audit.sort(key=lambda r: (-r["kept"], -r["support"], -r["lift"]))
    with open(OUT / "migration_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["from", "to", "support", "lift", "coexist", "inlined", "kept"])
        w.writeheader()
        w.writerows(audit)
    with open(OUT / "migrations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["from", "to", "support", "lift", "coexist", "verdict", "note"])
        for r in audit:
            if r["kept"]:
                w.writerow([r["from"], r["to"], r["support"], r["lift"], r["coexist"],
                            *manual.get((r["from"], r["to"]), ("not_checked", ""))])

    stats = D.summary_stats(projects)
    stats["interactions_exported"] = n
    stats["migration_rules_kept"] = sum(r["kept"] for r in audit)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
