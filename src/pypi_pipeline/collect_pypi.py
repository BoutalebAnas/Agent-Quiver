"""Collect dated dependency histories of popular PyPI projects.

For each project: the list of releases with their upload date, and for each release
its declared runtime dependencies (requires_dist, extras excluded). Output is
append-only JSONL, so the script can be stopped and resumed.

    python -m src.pypi_pipeline.collect_pypi --top 15000 --workers 32
"""
import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[2]          # repository root
RAW = ROOT / "data" / "raw" / "pypi"                 # git-ignored
TOP_URL = "https://raw.githubusercontent.com/hugovk/top-pypi-packages/main/top-pypi-packages.min.json"
SINCE = "2016-01-01"
MAX_VERSIONS = 40

_local = threading.local()


def session():
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = "agentquiver-student-project (ECE Paris)"
        _local.s = s
    return _local.s


def get_json(url, tries=4):
    for attempt in range(tries):
        try:
            r = session().get(url, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def runtime_deps(requires_dist):
    deps = set()
    for line in requires_dist or []:
        try:
            req = Requirement(line)
        except InvalidRequirement:
            continue
        if req.marker is not None and "extra" in str(req.marker):
            continue  # optional extra, not a core dependency
        deps.add(normalize(req.name))
    return sorted(deps)


def stable_versions(releases):
    out = []
    for ver, files in releases.items():
        if not files:
            continue
        try:
            v = Version(ver)
        except InvalidVersion:
            continue
        if v.is_prerelease or v.is_devrelease:
            continue
        times = [f.get("upload_time_iso_8601") or f.get("upload_time") for f in files]
        times = [t for t in times if t]
        if not times:
            continue
        t = min(times)[:19]
        if t >= SINCE:
            out.append((t, ver))
    out.sort()
    return out[-MAX_VERSIONS:]


def collect_project(name):
    meta = get_json(f"https://pypi.org/pypi/{name}/json")
    if meta is None:
        return None
    info = meta.get("info") or {}
    versions = stable_versions(meta.get("releases") or {})
    rows = []
    for t, ver in versions:
        vmeta = get_json(f"https://pypi.org/pypi/{name}/{ver}/json")
        if vmeta is None:
            continue
        rd = (vmeta.get("info") or {}).get("requires_dist")
        rows.append({"version": ver, "time": t, "deps": runtime_deps(rd), "has_metadata": rd is not None})
    return {
        "project": normalize(name),
        "summary": (info.get("summary") or "")[:300],
        "keywords": (info.get("keywords") or "")[:200],
        "classifiers": [c for c in (info.get("classifiers") or []) if c.startswith("Topic ::")][:10],
        "home_page": info.get("home_page") or (info.get("project_urls") or {}).get("Homepage") or "",
        "releases": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    top_path = RAW / "top_pypi_packages.json"
    if not top_path.exists():
        top_path.write_text(requests.get(TOP_URL, timeout=60).text)
    top = json.loads(top_path.read_text())["rows"][: args.top]
    names = [normalize(r["project"]) for r in top]
    out_path = RAW / "pypi_histories.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done.add(json.loads(line)["project"])
            except (ValueError, KeyError):
                pass
    todo = [n for n in names if n not in done]
    print(f"{len(done)} already collected, {len(todo)} to go", flush=True)
    lock = threading.Lock()
    start = time.time()
    with out_path.open("a") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(collect_project, n): n for n in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as exc:  # keep going, report at the end
                print(f"error {futures[fut]}: {exc}", file=sys.stderr)
                rec = None
            if rec:
                with lock:
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
            if i % 100 == 0:
                rate = i / (time.time() - start)
                print(f"{i}/{len(todo)} projects, {rate:.1f}/s, ETA {(len(todo)-i)/rate/60:.0f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
