"""From raw PyPI release histories to adoption events, splits and item metadata.

Definitions
- user: a Python project (its maintainers), seen through its releases over time.
- item: a PyPI package that a project can declare as a runtime dependency.
- valid release: a release whose dependency list is known. Releases with no metadata
  (null requires_dist and no wheel metadata) are NOT treated as "no dependencies":
  they are skipped, and the next valid release is compared with the last valid one.
  A release that suddenly declares nothing between two releases that declare
  dependencies is also treated as missing metadata (a packaging glitch, not a drop).
- adoption: package B appears in a valid release and was in no earlier release.
  It is *dated* only if the previous release is valid (no metadata gap in between);
  otherwise we know it was adopted but not when, so it is kept in the project's state
  but never used as a training label or a test target.
- drop: package A leaves and never comes back in a later valid release (a temporary
  removal is not a drop).
- migration A -> B: one dated release transition drops A and adopts B, in a project that
  had at least MIN_DEPS dependencies, and the transition is small (<= 3 drops, <= 3 adds),
  so that A and B are plausibly paired and not part of a full rewrite.
"""
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repository root
RAW = ROOT / "data" / "raw" / "pypi"                 # git-ignored
PROCESSED = ROOT / "data" / "processed"

# Global temporal split (the same cut for every project, so the future never leaks).
VAL_START = "2025-01-01"
TEST_START = "2025-07-01"
TEST_END = "2026-11-01"
MIN_HISTORY = 3   # a test/val user needs at least this many dependencies before the cut
MIN_DEPS = 3      # a project must declare at least this many dependencies at some point to be used at all
MAX_MIGRATION_CHANGES = 3
OWN_DEPS = {}     # every collected package -> dependencies of its latest valid release (filled by load_projects)


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass
class Project:
    name: str
    summary: str = ""
    text: str = ""
    releases: list = field(default_factory=list)  # [(time, version, frozenset(deps) or None if unknown)]


def default_raw_path():
    for name in ("pypi_histories_repaired.jsonl.gz", "pypi_histories.jsonl", "pypi_histories.jsonl.gz"):
        if (RAW / name).exists():
            return RAW / name
    raise FileNotFoundError("no raw histories in data/raw")


def load_projects(path=None, min_deps=MIN_DEPS):
    """Read the raw histories (repaired file first). Projects that never declare
    `min_deps` dependencies are left out: they say nothing about adoption choices."""
    import gzip
    path = Path(path) if path else default_raw_path()
    opener = gzip.open if path.suffix == ".gz" else open
    projects = {}
    for line in opener(path, "rt"):
        rec = json.loads(line)
        rels = sorted((r["time"], r["version"], frozenset(r["deps"]) if r.get("has_metadata") else None)
                      for r in rec["releases"])
        rels = _mark_glitches(rels)
        last = [d for _, _, d in rels if d is not None]
        OWN_DEPS[rec["project"]] = set(last[-1]) if last else set()
        if max((len(d) for _, _, d in rels if d is not None), default=0) < min_deps:
            continue
        name = rec["project"]
        topics = " ".join(c.split("::")[-1].strip() for c in rec.get("classifiers", []))
        text = " ".join([name.replace("-", " "), rec.get("summary", ""), rec.get("keywords", ""), topics])
        projects[name] = Project(name, rec.get("summary", ""), text, rels)
    return projects


def _mark_glitches(rels):
    """An empty dependency list between two non-empty valid releases -> unknown."""
    valid = [i for i, r in enumerate(rels) if r[2] is not None]
    out = list(rels)
    for j in range(1, len(valid) - 1):
        a, b, c = valid[j - 1], valid[j], valid[j + 1]
        if not rels[b][2] and rels[a][2] and rels[c][2]:
            out[b] = (rels[b][0], rels[b][1], None)
    return out


def events_of(project):
    """Adoptions [(t, pkg, not_dated)], persistent drops [(t, pkg)], migrations [(t, a, b)].

    `not_dated` is True for the first valid release (initial dependency set) and for
    adoptions that follow a metadata gap: both are known but not dated precisely."""
    adoptions, drops, migrations = [], [], []
    valid = [(t, d) for t, _, d in project.releases if d is not None]
    later = [set() for _ in valid]                      # union of deps in later valid releases
    acc = set()
    for i in range(len(valid) - 1, -1, -1):
        later[i] = set(acc)
        acc |= valid[i][1]
    seen, prev, gap, vi = set(), None, False, 0
    for t, _, deps in project.releases:
        if deps is None:
            gap = True
            continue
        if prev is None:
            for d in sorted(deps):
                adoptions.append((t, d, True))
        else:
            added = sorted(d for d in deps - prev if d not in seen)
            dropped = sorted(d for d in prev - deps if d not in later[vi] and d not in deps)
            for d in added:
                adoptions.append((t, d, gap))
            for d in dropped:
                drops.append((t, d))
            if (not gap and len(prev) >= MIN_DEPS and dropped and added
                    and len(dropped) <= MAX_MIGRATION_CHANGES and len(added) <= MAX_MIGRATION_CHANGES):
                migrations += [(t, a, b) for a in dropped for b in added]
        seen |= deps
        prev, gap = deps, False
        vi += 1
    return adoptions, drops, migrations


def state_before(adoptions, drops, cut):
    """Dependencies a project has just before `cut`, and its adoption sequence up to then."""
    current, sequence = set(), []
    timeline = [(t, 0, d) for t, d, _ in adoptions] + [(t, 1, d) for t, d in drops]
    for t, kind, d in sorted(timeline):
        if t >= cut:
            break
        if kind == 0:
            current.add(d)
            sequence.append(d)
        else:
            current.discard(d)
    return current, sequence


def build_split(projects, cut, end):
    """Users with a history before `cut` and their new adoptions in [cut, end)."""
    users = []
    for p in projects.values():
        adoptions, drops, _ = events_of(p)
        current, sequence = state_before(adoptions, drops, cut)
        if len(current) < MIN_HISTORY:
            continue
        before = set(sequence)
        targets = sorted({d for t, d, initial in adoptions
                          if cut <= t < end and not initial and d not in before})
        if targets:
            users.append({"user": p.name, "history": sequence, "current": sorted(current), "targets": targets})
    return users


def training_examples(projects, cut):
    """(history, next adopted package) pairs from events strictly before `cut`."""
    examples = []
    for p in projects.values():
        adoptions, drops, _ = events_of(p)
        for i, (t, d, initial) in enumerate(adoptions):
            if initial or t >= cut:
                continue
            _, seq = state_before(adoptions, drops, t)
            if len(seq) >= 2:
                examples.append((seq, d))
    return examples


def interactions_before(projects, cut):
    """Every (project, package) adoption before `cut` (used for popularity, co-occurrence, vocab)."""
    rows = []
    for p in projects.values():
        adoptions, _, _ = events_of(p)
        rows += [(p.name, d, t) for t, d, _ in adoptions if t < cut]
    return rows


def migration_table(projects, cut, min_support=3, min_lift=100.0, max_coexist=0.5, with_audit=False):
    """A -> B replacement rules from clean release transitions before `cut`.

    support   = number of distinct projects that dropped A and adopted B in one clean transition.
    lift      = P(adopt B | drop A) / P(adopt B in any transition): removes "everyone adopts it" pairs.
    coexist   = share of projects using both A and B that declare them *together* in some release.
                A real replacement (gym -> gymnasium) is rarely kept side by side with what it replaces.
    inlined   = B is a dependency of A's latest release while A has 3+ dependencies: dropping a
                wrapper and declaring one of its dependencies directly (msrest -> isodate) is not
                a replacement. If A depends on (almost) only B, A is a renamed shim of B
                (pynvml -> nvidia-ml-py, bs4 -> beautifulsoup4): that is a real migration, kept.
    Returns {A: [(B, support, lift), ...]} sorted by lift (and the full audit rows if asked).
    """
    support = defaultdict(set)
    drops, adds, transitions = Counter(), Counter(), 0
    for p in projects.values():
        adoptions, dr, migrations = events_of(p)
        drops.update(a for t, a in dr if t < cut)
        adds.update(b for t, b, nd in adoptions if t < cut and not nd)
        valid = [r for r in p.releases if r[2] is not None]
        transitions += sum(1 for r in valid[1:] if r[0] < cut)
        for t, a, b in migrations:
            if t < cut:
                support[(a, b)].add(p.name)
    cands = {k: len(v) for k, v in support.items() if len(v) >= min_support}
    # coexistence, on releases before the cut only
    involved = {x for ab in cands for x in ab}
    both, together = Counter(), Counter()
    for p in projects.values():
        ever, sim = set(), set()
        for t, _, d in p.releases:
            if d is None or t >= cut:
                continue
            ever |= d & involved
            sim |= {(a, b) for a, b in cands if a in d and b in d}
        for a, b in cands:
            if a in ever and b in ever:
                both[(a, b)] += 1
        together.update(sim)
    table, audit = defaultdict(list), []
    for (a, b), n in cands.items():
        lift = (n / drops[a]) / (adds[b] / max(transitions, 1)) if drops[a] and adds[b] else 0.0
        coexist = together[(a, b)] / both[(a, b)] if both[(a, b)] else 0.0
        own = OWN_DEPS.get(a, set())
        inlined = b in own and len(own) >= 3
        keep = lift >= min_lift and coexist <= max_coexist and not inlined
        audit.append({"from": a, "to": b, "support": n, "lift": round(lift, 1),
                      "coexist": round(coexist, 2), "inlined": inlined, "kept": keep})
        if keep:
            table[a].append((b, n, round(lift, 1)))
    table = {a: sorted(v, key=lambda x: -x[2]) for a, v in table.items()}
    return (table, audit) if with_audit else table


def test_migrations(projects, cut, end):
    """Clean release transitions in [cut, end) that dropped a package: (project, dropped A,
    packages adopted in that same transition, number of packages dropped)."""
    rows = []
    for p in projects.values():
        by_t = defaultdict(lambda: [set(), set()])
        adoptions, dr, migrations = events_of(p)
        for t, a, b in migrations:
            if cut <= t < end:
                by_t[t][0].add(a)
                by_t[t][1].add(b)
        for t, (dropped, added) in by_t.items():
            rows += [(p.name, a, frozenset(added), len(dropped)) for a in sorted(dropped)]
    return rows


def popularity(interactions, cut, window_days=365):
    from datetime import date, timedelta
    start = (date.fromisoformat(cut) - timedelta(days=window_days)).isoformat()
    return Counter(d for _, d, t in interactions if t >= start)


def summary_stats(projects):
    rel = [r for p in projects.values() for r in p.releases]
    items, dated, undated, drops, migs = set(), 0, 0, 0, 0
    for p in projects.values():
        a, d, m = events_of(p)
        items |= {x for _, x, _ in a}
        first_t = next((t for t, _, dd in p.releases if dd is not None), None)
        dated += sum(1 for t, _, nd in a if not nd)
        undated += sum(1 for t, _, nd in a if nd and t != first_t)
        drops += len(d)
        migs += len(m)
    return {"projects": len(projects), "releases": len(rel),
            "releases_without_metadata": sum(1 for r in rel if r[2] is None),
            "packages": len(items), "dated_adoptions": dated, "undated_adoptions_after_gap": undated,
            "persistent_drops": drops, "migration_pairs": migs}
