# AgentQuiver API

## Run
```bash
uv pip install -r requirements.txt
# mock answers (no model needed)
uv run uvicorn src.api.main:app --reload
# trained model: needs models/ckpt_loo_c4.pt and data/processed/interactions.csv
AQ_RECOMMENDER=model uv run uvicorn src.api.main:app --reload
```
Interactive docs: http://127.0.0.1:8000/docs
Other checkpoint: `AQ_CHECKPOINT=models/other.pt`.

## Front-end (Streamlit)
In a second terminal, with the API running:
```bash
uv run streamlit run app/streamlit_app.py      # opens http://localhost:8501
```
The app only calls the API over HTTP (env `AQ_API`, default http://127.0.0.1:8000).

## Feedback database (SQLite)
`POST /log_interaction` writes to `data/agentquiver.db` (env `AQ_DB`), table `interactions`.
Check it: `sqlite3 data/agentquiver.db "SELECT * FROM interactions ORDER BY id DESC LIMIT 5;"`
or `GET /interactions?user_id=<user>`.

## Endpoints
| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Status, `model_version`, `is_mock` (false with the trained model) |
| GET | `/users?limit=20` | Users the model can serve (most active first) |
| GET | `/recommend?username=<user>&k=10` | Top-k items grouped by interest, with explanation. 404 if unknown user |
| POST | `/log_interaction` | Save feedback: `user_id, repo_id, interest_id, rank, action, timestamp` |
| GET | `/interactions?user_id=&limit=` | Recorded feedback from SQLite, newest first |

## With the trained model
- Uses `ModelPredictor` (scripts/predict.py): weights loaded once at start-up.
- Items are **npm/PyPI packages** (`kind="package"`, id like `pypi:requests`), not GitHub repos.
- `score` is cosine similarity x learned scale, **not a probability** (`score_type` says which).
- Each item is placed under the interest (OMIE channel) it matches best; each interest is
  explained by the 2 history items its attention weights point to most.
- Only users present in `interactions.csv` can be served (no live GitHub lookup yet).

## Tests
```bash
uv run python -m pytest tests/test_api.py tests/test_api_model.py tests/test_db.py -q
```
`test_api_model.py` is skipped if the checkpoint or the data are missing.
