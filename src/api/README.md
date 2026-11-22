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

## Endpoints
| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Status, `model_version`, `is_mock` (false with the trained model) |
| GET | `/users?limit=20` | Users the model can serve (most active first) |
| GET | `/recommend?username=<user>&k=10` | Top-k items grouped by interest, with explanation. 404 if unknown user |
| POST | `/log_interaction` | Save feedback: `user_id, repo_id, interest_id, rank, action, timestamp` |
| GET | `/interactions?user_id=` | Show recorded feedback (demo/debug) |

## With the trained model
- Uses `ModelPredictor` (scripts/predict.py): weights loaded once at start-up.
- Items are **npm/PyPI packages** (`kind="package"`, id like `pypi:requests`), not GitHub repos.
- `score` is cosine similarity x learned scale, **not a probability** (`score_type` says which).
- Each item is placed under the interest (OMIE channel) it matches best; each interest is
  explained by the 2 history items its attention weights point to most.
- Only users present in `interactions.csv` can be served (no live GitHub lookup yet).

## Tests
```bash
uv run python -m pytest tests/test_api.py tests/test_api_model.py -q
```
`test_api_model.py` is skipped if the checkpoint or the data are missing.
