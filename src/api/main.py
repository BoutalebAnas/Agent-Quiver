"""AgentQuiver backend (FastAPI).

Run (mock):  uvicorn src.api.main:app --reload
Run (model): AQ_RECOMMENDER=model uvicorn src.api.main:app --reload
             (checkpoint: AQ_CHECKPOINT, default models/ckpt_loo_c4.pt)
Docs:  http://127.0.0.1:8000/docs
"""
import os
import re
import threading
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, HTTPException, Query

from .recommender import UnknownUserError, get_recommender
from .schemas import Health, InteractionIn, InteractionOut, RecommendResponse

# Any dataset id (GitHub login, PyPI/npm project...): 1-100 printable chars, no spaces.
# Unknown users are rejected later with a 404 by the recommender.
_GH_USER = re.compile(r"^\S{1,100}$")

recommender = get_recommender(os.getenv("AQ_RECOMMENDER", "mock"))

app = FastAPI(
    title="AgentQuiver API",
    version="0.1.0",
    description="Recommends the next open-source AI repositories to a GitHub developer.",
)

# Temporary in-memory store. The Data Engineer replaces it with SQLite (SQLAlchemy).
_interactions: List[InteractionOut] = []
_lock = threading.Lock()


@app.get("/health", response_model=Health, tags=["system"])
def health() -> Health:
    return Health(status="ok", model_version=recommender.model_version, is_mock=recommender.is_mock)


@app.get("/recommend", response_model=RecommendResponse, tags=["recommendations"])
def recommend(
    username: str = Query(..., description="GitHub username", examples=["torvalds"]),
    k: int = Query(10, ge=1, le=50, description="Total number of repositories"),
) -> RecommendResponse:
    if not _GH_USER.match(username):
        raise HTTPException(status_code=422, detail="Invalid GitHub username")
    try:
        interests = recommender.recommend(username, k)
    except UnknownUserError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return RecommendResponse(
        username=username, k=k,
        model_version=recommender.model_version, is_mock=recommender.is_mock,
        score_type=recommender.score_type,
        generated_at=datetime.now(timezone.utc), interests=interests,
    )


@app.get("/users", response_model=List[str], tags=["recommendations"])
def users(limit: int = Query(20, ge=1, le=500)) -> List[str]:
    """Usernames the model can serve (most active first). Used by the demo drop-down."""
    return recommender.known_users(limit)


@app.post("/log_interaction", response_model=InteractionOut, status_code=201, tags=["feedback"])
def log_interaction(event: InteractionIn) -> InteractionOut:
    with _lock:
        out = InteractionOut(
            **event.model_dump(exclude={"timestamp"}),
            interaction_id=len(_interactions) + 1,
            timestamp=event.timestamp or datetime.now(timezone.utc),
        )
        _interactions.append(out)
    return out


@app.get("/interactions", response_model=List[InteractionOut], tags=["feedback"])
def list_interactions(user_id: str | None = None) -> List[InteractionOut]:
    """Debug endpoint for the demo: show recorded feedback live."""
    return [e for e in _interactions if user_id is None or e.user_id == user_id]
