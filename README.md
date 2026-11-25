# AgentQuiver

Recommending the next open-source packages (npm / PyPI) to a developer or project, from their dated history of adopted dependencies.

The system has four parts: a dataset pipeline, a sequence model (`src/`), an API with a SQLite feedback log (`src/api/`), and a Streamlit front-end (`app/`).

---

## Quick start (about 5 minutes, no retraining)

The processed dataset (`data/processed/`) and the trained checkpoint (`models/ckpt_loo_c4.pt`) are in the repository,
so the system runs right after installation. Requires Python 3.10+ (tested with Python 3.13).

### Windows (PowerShell)

```powershell
git clone https://github.com/BoutalebAnas/Agent-Quiver.git
cd Agent-Quiver
python -m venv .venv                                   # or: py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt  # 2-3 minutes (torch)
```

Terminal 1, the API (http://127.0.0.1:8000/docs), in the `Agent-Quiver` folder:

```powershell
$env:AQ_RECOMMENDER="model"; .venv\Scripts\python -m uvicorn src.api.main:app
```

Terminal 2, the front-end (http://localhost:8501), in the `Agent-Quiver` folder:

```powershell
.venv\Scripts\python -m streamlit run app/streamlit_app.py
```

These commands call the environment's Python directly, so there is no activation step (PowerShell often blocks
`Activate.ps1` scripts).

### macOS / Linux

```bash
git clone https://github.com/BoutalebAnas/Agent-Quiver.git
cd Agent-Quiver
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt                        # 2-3 minutes (torch)
AQ_RECOMMENDER=model python -m uvicorn src.api.main:app  # terminal 1, the API
python -m streamlit run app/streamlit_app.py             # terminal 2 (run `source .venv/bin/activate` first)
```

### Use it

Pick a user (or type one, e.g. `pypi:prefect`), click **Recommend**, then **Useful / Not useful**. Below the recommendations, *Better tools* lists the replacements other projects made; the table at the bottom shows the feedback stored in SQLite.

Check that the real model is served: http://127.0.0.1:8000/health must return `"is_mock": false` and
`"model_version": "ckpt_loo_c4.pt"`.

**Tests:** `python -m pytest -q` (Windows: `.venv\Scripts\python -m pytest -q`; 32 tests: data cleaning, evaluation, model, API, database, better tools).

---

## Rebuild everything from scratch (optional)

### 1. Data

The dataset merges two sources: PyPI projects (about 90% of the interactions) and GitHub developers.

| Step | Command | Time |
|---|---|---|
| GitHub developers | run `notebooks/01_extract_data.ipynb`, then `notebooks/02_prepare_dataset.ipynb` | 1-2 h (GraphQL quota) |
| PyPI histories | `python -m src.pypi_pipeline.collect_pypi --top 15000 --workers 32` | ~20 min |
| Repair missing metadata | `python -m src.pypi_pipeline.repair_metadata --workers 32` | ~15 min |
| Merge both sources | `python -m src.pypi_pipeline.merge_datasets --github-dir data/processed` (reads the GitHub files written by notebook 02) | ~1 min |
| Convert to the training format | `python -m scripts.convert_new_data data/interim/combined data/processed keep_undated` (overwrites `interactions.csv` and `items.csv` with the merged data) | ~1 min |

Details of the PyPI cleaning (releases without metadata, dated adoptions, "better tool" migrations):
[`src/pypi_pipeline/README.md`](src/pypi_pipeline/README.md).

### 2. Model checkpoint

```bash
python -m src.train --protocol loo --no_ranker        # ~16 min on CPU, writes models/ckpt_loo_c4.pt
python -m src.train --protocol temporal --no_ranker   # temporal evaluation
```

---

## API

| Route | What it does |
|---|---|
| `GET /health` | Status, model version, `is_mock` |
| `GET /users?limit=` | Users the model can serve (most active first) |
| `GET /recommend?username=&k=` | Top-k packages grouped by interest, with an explanation per interest, restricted to the ecosystems already in the user's history (a PyPI project is not offered npm packages; cross-ecosystem `tool:` items stay). User ids look like `pypi:<project>` or `gh:<login>`; an unknown user returns 404 |
| `GET /better_tools?username=` | Packages the user still uses that other projects replaced in one release, with the replacement most of them chose (`data/processed/migrations.csv`, rules checked by hand) |
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

Two protocols, both over the full catalogue, with 95% bootstrap intervals:

- **LOO**: each user's last dated adoption is the test target, the one before it the validation target.
- **Temporal**: train before 2025-10-17, validation until 2026-05-12, test after. Undated rows (for example a project's initial
  dependencies) are never targets; those dated after the train cut are left out (`none`), so the model never trains on the future.

Results on the merged dataset (44,703 adoptions, 3,840 users, 7,561 packages, 99.85% sparse):

| | LOO Recall@20 | LOO NDCG@10 | Temporal Recall@20 | Temporal NDCG@10 |
|---|---|---|---|---|
| Most popular | 0.278 [0.264, 0.292] | 0.095 | 0.189 [0.176, 0.202] | 0.068 [0.061, 0.075] |
| Retriever (final) | **0.316** [0.302, 0.331] | **0.139** [0.130, 0.147] | 0.188 [0.175, 0.200] | 0.071 [0.064, 0.078] |

- In LOO the model is significantly better than popularity.
- In the temporal protocol it is tied with popularity in Recall@20.
- Cold start (packages absent from training) is **not solved**: Recall@20 is 0 for both, and 59% of packages have no text description.
- Run-to-run variation is about 0.003-0.005.

Ablations (LOO, retriever only, one seed each, `python -m src.ablate retr loo`, re-run on the current data):

| Variant | Recall@20 | NDCG@10 |
|---|---|---|
| Full model (served checkpoint `ckpt_loo_c4.pt`) | **0.316** | **0.139** |
| One interest instead of four | 0.300 | 0.132 |
| Without text | 0.299 | 0.130 |
| Without event age | 0.274 | 0.115 |

Every component helps, and event age matters most. An earlier run of the same ablation showed no gain from the four
interests (one interest 0.312 / 0.133 against 0.312 / 0.131 for the full model), so the interest gain needs more seeds before being
claimed. The two-stage ranker lowered validation Recall@20 from about 0.30 to 0.08, so it is not part of the final system. The diffusion
module was only tested on an earlier, smaller dataset, where it did not help.

Reproduce:

```powershell
python -m scripts.evaluate_baselines --protocol both        # baselines + sample-integrity checks
python -m src.train --protocol loo --no_ranker
python -m src.train --protocol temporal --no_ranker
python -m src.ablate retr                                    # ablations without the ranker
python -m src.manual_check models/ckpt_loo_c4.pt             # what each interest looks at
python scripts/measure_latency.py                            # API running; median 0.026 s, max 0.105 s (reports/week3/latency.csv)
```

---

## Known limitations

- The API keeps recommendations in the user's ecosystems. Measured offline, this filter raises temporal Recall@20 for both the model (0.189 → 0.207) and popularity (0.189 → 0.215), so it improves the product but does not change the comparison.
- Interest explanations can rely on infrastructure dependencies or on the project's own sub-packages.
- Some packages have no description; `stars`, `language` and `created_at` are empty; `is_new` is always `false`.
- Only users present in the dataset can be served (no live GitHub lookup yet).
- A user is either a project or a developer; the data is dominated by PyPI projects (about 90% of interactions).

## Team

| Member | Role |
|---|---|
| Ahmed Amine Zahraoui | Project Lead, Data Engineer |
| Anas Boutaleb | Lead ML Engineer (MLE-Core) |
| Ahmed El Monaser | ML Engineer (MLE-Ops) |
| Lionel Monteiro Wora | Systems Engineer |
