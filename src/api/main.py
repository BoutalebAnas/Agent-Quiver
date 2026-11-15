"""AgentQuiver backend (FastAPI).

Run:   uvicorn src.api.main:app --reload
Docs:  http://127.0.0.1:8000/docs
"""
import os
import re
import threading
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, HTTPException, Query

from .recommender import get_recommender
from .schemas import Health, InteractionIn, InteractionOut, RecommendResponse

# GitHub username rules: 1-39 chars, alphanumeric or single hyphens, no leading/trailing hyphen
_GH_USER = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")

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
    interests = recommender.recommend(username, k)
    return RecommendResponse(
        username=username, k=k,
        model_version=recommender.model_version, is_mock=recommender.is_mock,
        generated_at=datetime.now(timezone.utc), interests=interests,
    )


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
