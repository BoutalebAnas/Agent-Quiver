# AgentQuiver — final dataset

Two sources merged into one schema (built by `scripts/merge_datasets.py`, checks included).

| source | user | interaction | team member |
|---|---|---|---|
| `pypi_projects` | a PyPI project (6,644) | the project declares a package as a runtime dependency in a release | Ahmed Amine Zahraoui |
| `github_devs` | a GitHub developer (322) | the developer starts using an npm/PyPI package in their repositories | team GitHub collection |

## Files

| file | rows | content |
|---|---|---|
| `interactions.csv` | 63,088 | one row per (user, item): first adoption |
| `users.csv` | 6,966 | `user_id`, `user` (`pypi:<project>` or `gh:<developer>`), type, source, counts, first/last date |
| `items.csv` | 8,928 | `item_id`, `item`, `members`, ecosystems, summary, `is_ai`, users per source |
| `migrations.csv` | 45 | "better tool" rules A → B (support, lift, coexistence) with a manual verdict |
| `github_stars.csv`, `github_forks.csv` | 14,310 / 2,948 | side information on the GitHub developers (not interactions) |
| `dataset_info.json` | | all counts below, split cuts, cleaning statistics |

## interactions.csv

| column | meaning |
|---|---|
| `user_id`, `item_id` | integer ids (join with users.csv / items.csv) |
| `user`, `item` | readable ids |
| `date` | adoption date, UTC, `YYYY-MM-DDTHH:MM:SS` |
| `dated` | **True = real adoption date.** False for a PyPI project's first collected dependency set and for adoptions right after a release without metadata: known, but not when. Use them as history, **never as targets**. |
| `split` | one global temporal split: `train` < 2025-01-01 ≤ `valid` < 2025-07-01 ≤ `test` < 2026-11-01 ≤ `after_test` |
| `source`, `user_type` | `pypi_projects`/`project` or `github_devs`/`developer` |
| `is_ai` | the item is in the team's curated list of 43 AI tools/libraries |
| `still_in_use` | PyPI: still a dependency in the latest release. GitHub: still used in the developer's repositories. |
| `n_repos` | GitHub: number of the developer's repositories using it (PyPI: 1) |

## Items

- Ids: `pypi:<name>`, `npm:<name>`, or `tool:<name>` when one tool exists in several ecosystems:
  `tool:openai` = `pypi:openai` + `npm:openai`. There are 17 such tools, and the grouping is applied to both sources.
- 179 items are used by both projects and developers (requests, numpy, pydantic, openai, mcp…).
- 8,489 PyPI, 422 npm, 17 cross-ecosystem tools. 43 items are tagged AI, covering 2,211 interactions.

## Counts

| | PyPI projects | GitHub developers | total |
|---|---|---|---|
| users | 6,644 | 322 | 6,966 |
| interactions | 58,477 | 4,611 | 63,088 |
| dated interactions | 19,126 | 4,611 | 23,737 |
| train / valid / test / after_test | 34,088 / 4,941 / 18,602 / 846 | 1,057 / 336 / 3,076 / 142 | 35,145 / 5,277 / 21,678 / 988 |

Every user has at least 3 interactions; the median is 6. Density is 0.10%.

## How each source was cleaned

**PyPI projects.**
- **Collection:** every stable release since 2016 (40 most recent at most) of the 15,000 most downloaded PyPI projects, with its runtime dependencies (optional extras excluded).
- **Missing metadata:** 22% of releases had no dependency list in the PyPI JSON API. 59% of them were recovered from the wheel metadata (PEP 658). The rest are unknown: skipped, never read as "no dependencies".
- **Packaging glitches:** an empty list between two non-empty releases is treated as a glitch, not a real change.
- **Drops:** a package removed and later re-added is not a drop.
- **Minimum size:** projects need at least 3 dependencies.

**GitHub developers.**
- **Collection:** the team's collection (npm and PyPI manifests of the developers' repositories).
- **Filters:** items used by at least 3 developers; developers with at least 5 interactions.
- **Grouping:** cross-ecosystem tools grouped under `tool:` ids.
- **Their original splits:** kept in `data/external/github_devs/interactions.csv` (`split_loo`, `split_temporal`).

## Migrations ("better tools")

A rule A → B comes from PyPI release transitions that drop A and adopt B. It is kept only if all of these hold:
- the change is small (≤ 3 drops and ≤ 3 adds);
- it is seen in at least 3 projects;
- lift ≥ 100;
- A and B are declared together in at most 50% of the projects using both;
- B is not a dependency of a wrapper A.

A manual check of the 45 rules found 39 real replacements (87%) and 6 side effects (`verdict` column).

## Known limits

- The two user types differ: a project's adoptions are maintainer decisions; a developer's are personal usage. Keep `source`
  as a feature, or evaluate per source.
- GitHub interactions are recent: 67% are in `test`. The GitHub part alone is too small to train on before 2025-07.
- `is_ai` comes from a curated list: an AI package missing from it is tagged False.
