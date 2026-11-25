"""AgentQuiver backend (FastAPI).

Run (mock):  uvicorn src.api.main:app --reload
Run (model): AQ_RECOMMENDER=model uvicorn src.api.main:app --reload
             (checkpoint: AQ_CHECKPOINT, default models/ckpt_loo_c4.pt)
Docs:  http://127.0.0.1:8000/docs
"""
import os
import re
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, HTTPException, Query

from .better_tools import BetterTools
from .db import InteractionStore
from .stack import Stacks, parse_packages
from .recommender import UnknownUserError, get_recommender
from .schemas import (BetterTool, Health, InteractionIn, InteractionOut, RecommendResponse, StackIn,
                      StackItem, StackResponse)

# Any dataset id (GitHub login, PyPI/npm project...): 1-100 printable chars, no spaces.
# Unknown users are rejected later with a 404 by the recommender.
_GH_USER = re.compile(r"^\S{1,100}$")

recommender = get_recommender(os.getenv("AQ_RECOMMENDER", "mock"))

app = FastAPI(
    title="AgentQuiver API",
    version="0.1.0",
    description="Recommends the next open-source AI repositories to a GitHub developer.",
)

# Feedback is stored in SQLite (env AQ_DB, default data/agentquiver.db)
store = InteractionStore()

# "Better tools": replacements other projects made (data/processed/migrations.csv)
better_tools = BetterTools()

# Users' histories and package-name lookup (data/processed/interactions.csv, items.csv)
stacks = Stacks()


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
        # items already rated by this user in the app are not recommended again
        interests = recommender.recommend(username, k, exclude=store.rated_items(username))
    except UnknownUserError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return RecommendResponse(
        username=username, k=k,
        model_version=recommender.model_version, is_mock=recommender.is_mock,
        score_type=recommender.score_type,
        generated_at=datetime.now(timezone.utc), interests=interests,
    )


@app.get("/better_tools", response_model=List[BetterTool], response_model_by_alias=True, tags=["recommendations"])
def get_better_tools(username: str = Query(..., description="User id, e.g. pypi:prefect")) -> List[BetterTool]:
    """Packages the user still uses that other projects replaced, with what they switched to."""
    if not _GH_USER.match(username):
        raise HTTPException(status_code=422, detail="Invalid username")
    try:
        return [BetterTool(**s) for s in better_tools.suggest(username)]
    except UnknownUserError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/history", response_model=List[StackItem], tags=["recommendations"])
def history(username: str = Query(..., description="User id, e.g. pypi:prefect")) -> List[StackItem]:
    """The packages in the user's history, oldest first (what the model reads)."""
    if not _GH_USER.match(username):
        raise HTTPException(status_code=422, detail="Invalid username")
    try:
        return [StackItem(**h) for h in stacks.history(username)]
    except UnknownUserError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/recommend_stack", response_model=StackResponse, response_model_by_alias=True, tags=["recommendations"])
def recommend_stack(body: StackIn) -> StackResponse:
    """Recommendations for any stack: paste a requirements.txt, a package.json or package names.
    Known packages become the history (all treated as just adopted); unknown names are reported."""
    if not hasattr(recommender, "recommend_items"):
        raise HTTPException(status_code=503, detail="Needs the trained model (start the API with AQ_RECOMMENDER=model)")
    matched, unknown = stacks.match(parse_packages(body.text))
    if not matched:
        raise HTTPException(status_code=404, detail="None of these packages is in our catalogue")
    try:
        interests = recommender.recommend_items(matched, body.k)
    except UnknownUserError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return StackResponse(matched=matched, unknown=unknown, model_version=recommender.model_version,
                         is_mock=recommender.is_mock, score_type=recommender.score_type, interests=interests,
                         better_tools=[BetterTool(**t) for t in better_tools.suggest_for_items(matched)])


@app.get("/users", response_model=List[str], tags=["recommendations"])
def users(limit: int = Query(20, ge=1, le=500)) -> List[str]:
    """Usernames the model can serve (most active first). Used by the demo drop-down."""
    return recommender.known_users(limit)


@app.post("/log_interaction", response_model=InteractionOut, status_code=201, tags=["feedback"])
def log_interaction(event: InteractionIn) -> InteractionOut:
    ts = event.timestamp or datetime.now(timezone.utc)
    new_id = store.add(event.user_id, event.repo_id, event.interest_id, event.rank,
                       event.action.value, ts.isoformat())
    return InteractionOut(**event.model_dump(exclude={"timestamp"}), interaction_id=new_id, timestamp=ts)


@app.get("/interactions", response_model=List[InteractionOut], tags=["feedback"])
def list_interactions(user_id: str | None = None,
                      limit: int = Query(100, ge=1, le=1000)) -> List[InteractionOut]:
    """Recorded feedback, newest first (read from the SQLite database)."""
    return [InteractionOut(interaction_id=r["id"], user_id=r["user_id"], repo_id=r["repo_id"],
                           interest_id=r["interest_id"], rank=r["rank"], action=r["action"],
                           timestamp=r["timestamp"])
            for r in store.list(user_id, limit)]
