"""Repair releases whose dependencies are missing from the PyPI JSON API.

The JSON API often reports `requires_dist = null` for a release (22% of our releases),
typically when the first uploaded file was an sdist. For many of these releases, a wheel
exists and PyPI serves its core metadata (PEP 658: <wheel url>.metadata). We read
`Requires-Dist` from there. Releases that still have no metadata stay marked
`has_metadata = false` and are handled in data.py (never compared against).

    python -m src.pypi_pipeline.repair_metadata --workers 32
Input : data/raw/pypi/pypi_histories.jsonl(.gz)   Output: data/raw/pypi/pypi_histories_repaired.jsonl.gz
"""
import argparse
import gzip
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from src.pypi_pipeline.collect_pypi import RAW, runtime_deps  # noqa: E402

_local = threading.local()


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = "agentquiver-student-project (ECE Paris)"
    return _local.s


def get(url, accept=None, tries=3):
    for _ in range(tries):
        try:
            r = session().get(url, timeout=30, headers={"Accept": accept} if accept else {})
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
    return None


def wheel_version(filename):
    # name-version(-build)?-pyver-abi-plat.whl
    parts = filename[:-4].split("-")
    return parts[1] if len(parts) >= 5 else None


def repair(rec):
    missing = [r for r in rec["releases"] if not r["has_metadata"]]
    if not missing:
        return rec, 0, 0
    r = get(f"https://pypi.org/simple/{rec['project']}/", accept="application/vnd.pypi.simple.v1+json")
    if r is None:
        return rec, len(missing), 0
    by_version = {}
    for f in r.json().get("files", []):
        if f["filename"].endswith(".whl") and f.get("core-metadata"):
            v = wheel_version(f["filename"])
            # prefer a pure-python wheel, it carries the generic dependency list
            if v and (v not in by_version or f["filename"].endswith("none-any.whl")):
                by_version[v] = f["url"]
    fixed = 0
    for rel in missing:
        url = by_version.get(rel["version"])
        if not url:
            continue
        m = get(url + ".metadata")
        if m is None:
            continue
        reqs = [l[len("Requires-Dist:"):].strip() for l in m.text.splitlines() if l.startswith("Requires-Dist:")]
        rel["deps"] = runtime_deps(reqs)
        rel["has_metadata"] = True
        rel["metadata_source"] = "wheel"
        fixed += 1
    return rec, len(missing), fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    src = RAW / "pypi_histories.jsonl"
    opener = open if src.exists() else gzip.open
    src = src if src.exists() else Path(str(src) + ".gz")
    recs = [json.loads(l) for l in opener(src, "rt")]
    total_missing = total_fixed = 0
    out = []
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(repair, r) for r in recs]
        for i, f in enumerate(as_completed(futs), 1):
            rec, miss, fixed = f.result()
            out.append(rec)
            total_missing += miss
            total_fixed += fixed
            if i % 1000 == 0:
                print(f"{i}/{len(recs)} projects, repaired {total_fixed}/{total_missing}", flush=True)
    out.sort(key=lambda r: r["project"])
    with gzip.open(RAW / "pypi_histories_repaired.jsonl.gz", "wt") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    n = sum(len(r["releases"]) for r in out)
    print(json.dumps({"releases": n, "missing_before": total_missing, "repaired_from_wheels": total_fixed,
                      "still_missing": total_missing - total_fixed}), flush=True)


if __name__ == "__main__":
    main()
