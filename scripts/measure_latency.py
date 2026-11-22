"""Measure /recommend response time on the running API (proof for Q5: < 2 s).

Usage (API already running in another terminal):
    uv run python -m scripts.measure_latency --users 20 --repeat 3

Writes reports/week3/latency.csv and prints median / p95 / max.
"""
import argparse
import csv
import json
import statistics
import time
import urllib.parse
import urllib.request
from pathlib import Path


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--users", type=int, default=20, help="number of real users to test")
    ap.add_argument("--repeat", type=int, default=3, help="calls per user")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="reports/week3/latency.csv")
    a = ap.parse_args()

    _, body = get(f"{a.api}/health")
    health = json.loads(body)
    print("health:", health)
    if health["is_mock"]:
        print("WARNING: the API is running the MOCK. Start it with AQ_RECOMMENDER=model.")

    _, body = get(f"{a.api}/users?limit={a.users}")
    users = json.loads(body)

    rows, times = [], []
    for u in users:
        for i in range(a.repeat):
            url = f"{a.api}/recommend?" + urllib.parse.urlencode({"username": u, "k": a.k})
            t0 = time.perf_counter()
            status, _ = get(url)
            dt = time.perf_counter() - t0
            times.append(dt)
            rows.append({"user": u, "call": i + 1, "status": status, "seconds": round(dt, 4)})

    times.sort()
    p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
    summary = {"calls": len(times), "median_s": round(statistics.median(times), 4),
               "p95_s": round(p95, 4), "max_s": round(times[-1], 4),
               "under_2s": all(t < 2 for t in times), "model_version": health["model_version"]}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"details saved to {a.out}")


if __name__ == "__main__":
    main()
