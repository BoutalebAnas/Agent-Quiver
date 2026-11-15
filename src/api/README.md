# AgentQuiver API (week 2: mock answers)

## Run
```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```
Interactive docs: http://127.0.0.1:8000/docs

## Endpoints
| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Status + model version (`is_mock`) |
| GET | `/recommend?username=<gh_user>&k=10` | Top-k repos, grouped by 4 interests, with explanation |
| POST | `/log_interaction` | Save feedback: `user_id, repo_id, interest_id, rank, action, timestamp` |
| GET | `/interactions?user_id=` | Show recorded feedback (demo/debug) |

`action` ∈ `useful`, `not_useful`, `open`, `star`.

## Plugging in the real model (week 3)
The API only calls `Recommender.recommend(username, k) -> List[Interest]` (`src/api/recommender.py`).
Write a `ModelRecommender` with this method (calls the predict function from MLE-Ops),
register it in `get_recommender()`, then start with `AQ_RECOMMENDER=model`.
The in-memory feedback store is to be replaced by SQLite (Data Engineer).

## Tests
```bash
python -m pytest tests/test_api.py -q
```
