# AgentQuiver

Recommending the next PyPI packages to a Python project, from the dated history of the dependencies it adopted.

The system has four parts: a dataset pipeline, a sequence model (`src/`), an API with a SQLite feedback log (`src/api/`), and a Streamlit front-end (`app/`).

---

## Quick start

Requires Python 3.10+.

```powershell
git clone https://github.com/BoutalebAnas/Agent-Quiver.git
cd Agent-Quiver
pip install -r requirements.txt
```

### 1. Data

The data is built from the public PyPI API (no key needed). **A user is a PyPI project, an item is a package it declares as a runtime dependency, an interaction is the first release that declares it** (dated by the upload date of that release). Loading and cleaning take about 1 hour. Datasets are not in Git (`data/raw/` and `data/interim/` are git-ignored).

```powershell
pip install requests packaging
python -m src.pypi_pipeline.collect_pypi --top 15000 --workers 32      # ~40 min, resumes if interrupted
python -m src.pypi_pipeline.repair_metadata --workers 32               # fill missing dependency lists
python -m src.pypi_pipeline.merge_datasets                             # clean, dated adoptions -> data/interim/combined
python -m src.pypi_pipeline.make_processed data/interim/combined data/processed keep_undated
```

| Step | What it does | Output |
|---|---|---|
| `collect_pypi` | For each of the 15,000 most downloaded projects: stable releases since 2016 (40 most recent at most), upload date, runtime dependencies (optional extras excluded). | `data/raw/pypi/pypi_histories.jsonl` |
| `repair_metadata` | 75,528 releases had no dependency list in the JSON API; 44,764 of them (59%) are recovered from the wheel metadata (PEP 658). | `data/raw/pypi/pypi_histories_repaired.jsonl.gz` |
| `merge_datasets` | Cleaning rules below, one row per (project, package), `dated` flag, checks that fail loudly. | `data/interim/combined/` |
| `make_processed` | Keeps users with at least 3 events of which 2 are dated, re-indexes ids, builds the two evaluation splits and checks them. | `data/processed/` |

Cleaning rules (`src/pypi_pipeline/pypi_events.py`):

1. **Missing metadata.** The 30,764 releases that stay without a dependency list are skipped and never read as "no dependencies". An empty list between two non-empty releases is treated as a packaging glitch.
2. **Dated adoptions.** An adoption is dated only if the previous release has metadata. A project's first dependency set and adoptions right after a metadata gap are known but not dated (`dated = False`): they are used as history, **never as targets**. Of 58,512 interactions, 19,140 are dated.
3. **Real adoptions only.** A package removed and added back later is not counted as a new adoption or a drop.
4. **Projects.** Only projects that declare at least 3 dependencies are kept: 6,644 projects, 8,507 packages.

`make_processed` then removes 3,113 users with fewer than 3 events or fewer than 2 dated events. **Final dataset: 40,202 interactions, 3,531 users, 7,142 packages, sparsity 0.9984**; 4,533 packages (63%) have no description. The last line of its output should read `FINAL: 40202 interactions | 3531 users | 7142 items` (small differences are normal if PyPI changed since our run).

Evaluation splits built by `make_processed`:
- **Leave-last-out (LOO):** last dated adoption of each user = test, the one before = validation.
- **Temporal:** cut dates are the 70% and 85% quantiles of the dated adoptions (2025-11-14 and 2026-06-24 in our run). Undated rows dated after the first cut are dropped (4,615 rows), so training never sees the future. The script stops with an error if a training row is dated after the cut or if a target is undated.

### 2. Model checkpoint

Train the model (about 16 minutes on CPU) with:

```powershell
python -m src.train --protocol loo --no_ranker
```

The checkpoint is written to `models/`.

### 3. Run the system (two terminals)

Terminal 1, the API (http://127.0.0.1:8000/docs):

```powershell
$env:AQ_RECOMMENDER="model"
python -m uvicorn src.api.main:app
```
macOS/Linux: `AQ_RECOMMENDER=model python -m uvicorn src.api.main:app`

Terminal 2, the front-end (http://localhost:8501):

```powershell
python -m streamlit run app/streamlit_app.py
```

Pick a user, click **Recommend**, then **Useful / Not useful**. The table at the bottom shows the feedback stored in the database.

Check that it really runs the model: `GET /health` must return `"is_mock": false` and `"model_version": "ckpt_loo_c4.pt"`.

### 4. Tests

```powershell
python -m pytest -q
```

Tests that need the checkpoint or the data are skipped when those are missing; make sure they are not skipped on your machine.

---

## API

| Route | What it does |
|---|---|
| `GET /health` | Status, model version, `is_mock` |
| `GET /users?limit=` | Users the model can serve (most active first) |
| `GET /recommend?username=&k=` | Top-k packages grouped by interest, with an explanation per interest. User ids look like `pypi:<project>`; an unknown user returns 404 |
| `POST /log_interaction` | Stores `user_id, repo_id, interest_id, rank, action` in SQLite (`data/agentquiver.db`, override with the `AQ_DB` variable) |
| `GET /interactions?user_id=&limit=` | Reads the stored feedback back (newest first) |

Scores are similarities, **not probabilities** (`score_type` says so). To reset the demo database, stop the API and delete the file set in `AQ_DB`.
The feedback is stored for a future training run; nothing reads it back into the model yet.

---

## The model

`src/core_model.py`, trained by `src/train.py`.

- **Sequence encoder.** A small Transformer over the user's dated history. An item vector is an ID embedding plus a TF-IDF/SVD vector of its description; an embedding of each event's age is added.
- **Interest extractor.** Four attention channels (inspired by the OMIE module of DiffuMIN, Lai et al., RecSys 2025) with an orthogonality penalty.
- **Score.** Cosine similarity with a learned scale, best match over the interests. Already-adopted packages are removed.
- **Tested but not used in the final system.** A target-guided ranker (`--ranker` is the default of `src.train`; the final system uses `--no_ranker`) and a diffusion module (`--diffusion`).

## Evaluation

Two protocols, both over the full catalogue, seen items masked, with 95% bootstrap intervals:

- **LOO**: each user's last dated adoption is the test target, the one before it the validation target.
- **Temporal**: train before 2025-11-14, validation until 2026-06-24, test after.

Results on the PyPI dataset (40,202 interactions, 3,531 users, 7,142 packages, 99.84% sparse):

| | LOO Recall@20 | LOO NDCG@10 | Temporal Recall@20 | Temporal NDCG@10 |
|---|---|---|---|---|
| Random | 0.0017 | 0.0004 | 0.0064 | 0.0006 |
| Most popular | 0.309 | 0.108 | 0.241 | 0.084 |
| Retriever (final) | 0.313 [0.299, 0.329] | 0.129 [0.120, 0.137] | 0.253 [0.234, 0.272] | 0.096 [0.085, 0.106] |

- The model is above popularity everywhere, but the gap is small: +0.004 Recall@20 in LOO and +0.011 in temporal, inside the model's interval. Only the LOO NDCG@10 gain (+0.021) is clearly outside it. No paired significance test was run.
- Cold start (packages absent from training) is **not solved**: Recall@20 is 0 on the 460 cold targets of LOO (of 3,531) and the 397 of temporal (of 1,881).
- Training stops early (patience 8): LOO best validation at epoch 20 (Recall@20 0.302, NDCG@10 0.134), temporal best validation NDCG@10 at epoch 5.

Ablations (LOO test, one seed; full model 0.313 Recall@20 / 0.129 NDCG@10):

| Variant | Recall@20 | NDCG@10 |
|---|---|---|
| One interest instead of four | 0.300 | 0.130 |
| Without text | 0.331 | 0.131 |
| Without event age | 0.309 | 0.127 |

No variant is clearly different from the full model on this dataset: neither text, nor event age, nor several interests bring a measurable gain (single seed, differences within run-to-run noise). A manual check shows the four interest channels recommend almost the same packages, mostly very common ones (`packaging`, `typing-extensions`, `protobuf`). On an earlier version of the data, the two-stage ranker lowered validation Recall@20 from about 0.30 to 0.08 and the diffusion module did not help, so neither is part of the final system.

Reproduce:

```powershell
python -m scripts.evaluate_baselines --protocol both        # baselines + sample-integrity checks
python -m src.train --protocol loo --no_ranker
python -m src.train --protocol temporal --no_ranker
python -m src.ablate retr                                    # ablations without the ranker
python -m src.manual_check models/ckpt_loo_c4.pt             # what each interest looks at
python scripts/measure_latency.py                            # API running
```

---

## Known limitations

- Interest explanations can rely on infrastructure dependencies or on the project's own sub-packages.
- Some packages have no description; `stars`, `language` and `created_at` are empty; `is_new` is always `false`.
- Only users present in the dataset can be served (no live PyPI lookup yet).
- Only PyPI is covered: users are projects, not individual developers, and a project's adoptions are maintainer decisions.
- Recommendations are often generic, very common packages.

## Team

| Member | Role |
|---|---|
| Ahmed Amine Zahraoui | Project Lead, Data Engineer |
| Anas Boutaleb | Lead ML Engineer (MLE-Core) |
| Ahmed El Monaser | ML Engineer (MLE-Ops) |
| Lionel Monteiro Wora | Systems Engineer |
