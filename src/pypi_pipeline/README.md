# src/pypi_pipeline — PyPI dependency histories, cleaned

Collects the release-by-release dependencies of popular PyPI projects, cleans them, extracts dated adoptions and
"better tool" migrations, then merges them with the team's GitHub developers dataset.

User = a PyPI project. Item = a package it declares as a runtime dependency. Interaction = the first release
that declares it.

## Run

From the repository root (Python 3.10+; needs `requests` and `packaging` in addition to `requirements.txt`):

| step | command | time | output |
|---|---|---|---|
| 1. collect | `python -m src.pypi_pipeline.collect_pypi --top 15000 --workers 32` | ~20 min | `data/raw/pypi/pypi_histories.jsonl` |
| 2. repair missing metadata | `python -m src.pypi_pipeline.repair_metadata --workers 32` | ~15 min | `data/raw/pypi/pypi_histories_repaired.jsonl.gz` |
| 3. clean + export PyPI part | `python -m src.pypi_pipeline.export_pypi` | ~1 min | `data/interim/pypi/` |
| 4. merge with GitHub data | `python -m src.pypi_pipeline.merge_datasets --github-dir <GitHub dataset folder>` | ~1 min | `data/interim/combined/` |
| tests | `pytest -q tests/test_pypi_events.py` | 1 s | |

Steps 1–2 call the public PyPI API: no key needed. Step 1 resumes where it stopped if interrupted.
`data/raw/` and `data/interim/` are git-ignored: raw histories and intermediate files are rebuilt, not committed.
The merged file is the source of `data/processed/interactions.csv` and `items.csv` (PyPI projects + GitHub developers).
Step 4 needs `items.csv`, `interactions.csv`, `stars.csv` and `forks.csv` from the GitHub dataset. Extra columns
are ignored, and the columns it reads must be there: `developer`, `item`, `date`, `still_in_use`, `n_repos`;
`item`, `members`, `ecosystems`, `summary`, `is_ai`.

## Files

| file | role |
|---|---|
| `collect_pypi.py` | for each project: stable releases since 2016 (40 most recent), upload date, runtime dependencies (`requires_dist`, extras excluded) |
| `repair_metadata.py` | releases with no dependency list in the JSON API: read `Requires-Dist` from the wheel metadata (PEP 658) |
| `pypi_events.py` | cleaning rules, adoptions, drops, migrations, migration table |
| `export_pypi.py` | cleaned PyPI interactions + migration rules + audit as CSV |
| `merge_datasets.py` | one schema for PyPI projects and GitHub developers, one temporal split, checks |
| `migration_manual_check.csv` | manual verdict on each migration rule kept |
| `DATASET.md` | description of every file and column of the merged dataset |

## Cleaning rules (`pypi_events.py`)

1. **Missing metadata.** 22% of releases (75,516) have no dependency list in the PyPI JSON API. 59% of them are
   recovered from the wheel metadata. The rest are *unknown*: skipped and never compared against, so they are not
   read as "no dependencies". An empty list between two non-empty releases is treated as a packaging glitch.
2. **Dated adoptions.** An adoption is dated only if the previous release has metadata. The first release's dependencies
   and adoptions right after a gap are kept as history, never as targets (`dated = False`).
3. **Real drops.** A package removed and re-added later is not a drop (5,364 false drops removed).
4. **Projects.** Only projects with at least 3 dependencies are kept (6,644 projects).
5. **Migrations A → B.** One release drops A and adopts B, in a project with at least 3 dependencies, and the change
   is small (≤ 3 drops and ≤ 3 adds). A rule is kept if all of these hold:
   - it is seen in ≥ 3 projects;
   - lift ≥ 100;
   - A and B are declared together in ≤ 50% of the projects using both;
   - B is not a dependency of a wrapper A (dropping msrest and declaring its dependency isodate is not a replacement).

   Manual check of the 45 rules kept: **39 real replacements (87%)**, for example toml → tomli,
   appdirs → platformdirs, chardet → charset-normalizer, bleach → nh3, retry → tenacity and httpx → httpx2.
   The other 6 are side effects.

## Result after cleaning

| projects | releases | unknown releases | packages | dated adoptions | drops | migration pairs | rules kept |
|---|---|---|---|---|---|---|---|
| 6,644 | 188,568 | 11,698 | 8,500 | 19,126 | 13,993 | 3,892 | 45 |
