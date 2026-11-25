"""API contract for AgentQuiver (request / response models).

These models are the contract between the backend, the Streamlit app and the
model team. The mock and the real model must both return these exact shapes.
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Action(str, Enum):
    """What the user did with a recommendation in the app."""
    useful = "useful"
    not_useful = "not_useful"
    open = "open"      # clicked to open the repo
    star = "star"      # starred the repo on GitHub afterwards


class RepoCard(BaseModel):
    """One recommended item. With the trained model, items are npm/PyPI packages
    (kind="package", repo_id like "pypi:requests"), not GitHub repositories."""
    repo_id: str = Field(..., examples=["pypi:requests"])
    kind: str = Field("repository", description='"repository" (mock) or "package" (trained model)')
    description: Optional[str] = None
    language: Optional[str] = None
    topics: List[str] = Field([], description="Topics, or ecosystems (npm, pypi) for packages")
    stars: Optional[int] = None
    created_at: Optional[datetime] = None
    url: Optional[str] = None


class RecommendedRepo(BaseModel):
    rank: int = Field(..., ge=1, description="Global rank, 1 = best")
    score: float = Field(..., description="Model score. Not a probability: cosine similarity x learned scale")
    is_new: bool = Field(False, description="True if the repo is a cold-start repo")
    repo: RepoCard


class Interest(BaseModel):
    interest_id: int = Field(..., ge=0, description="Index of the interest channel (OMIE)")
    label: str = Field(..., examples=["LLM serving"])
    explanation: str = Field(..., examples=["Because you work on LLM serving"])
    evidence: List[str] = Field([], description="Repos from the history behind this interest")
    recommendations: List[RecommendedRepo]


class RecommendResponse(BaseModel):
    username: str
    k: int
    model_version: str
    is_mock: bool
    score_type: str = "mock"
    generated_at: datetime
    interests: List[Interest]


class InteractionIn(BaseModel):
    user_id: str = Field(..., examples=["torvalds"])
    repo_id: str = Field(..., examples=["vllm-project/vllm"])
    interest_id: Optional[int] = Field(None, ge=0)
    rank: Optional[int] = Field(None, ge=1)
    action: Action
    timestamp: Optional[datetime] = None


class InteractionOut(InteractionIn):
    interaction_id: int
    timestamp: datetime


class Health(BaseModel):
    status: str
    model_version: str
    is_mock: bool


class BetterTool(BaseModel):
    """A replacement that other projects made: they dropped `replace` and adopted `with_` in one release."""
    replace: str = Field(..., examples=["pypi:toml"])
    with_: str = Field(..., alias="with", examples=["pypi:tomli"])
    projects: int = Field(..., description="Number of projects that made this switch")
    lift: float = Field(..., description="How much dropping `replace` raises the chance of adopting `with`")
    note: str = ""

    model_config = {"populate_by_name": True}
