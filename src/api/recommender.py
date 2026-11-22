"""Recommender interface + mock implementation.

The API only talks to a `Recommender`. In week 2 it is `MockRecommender`
(fake but realistic answers). In week 3, MLE-Ops plugs the real model by
writing a class with the same `recommend()` signature, e.g. `ModelRecommender`
that calls the predict function, then sets AQ_RECOMMENDER=model.
"""
import hashlib
import random
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Protocol

from .schemas import Interest, RecommendedRepo, RepoCard

N_INTERESTS = 4  # c = 4 channels, as in the project plan


class Recommender(Protocol):
    model_version: str
    is_mock: bool

    def recommend(self, username: str, k: int = 10, exclude: Iterable[str] = ()) -> List[Interest]:
        """Return up to `k` items in total, grouped by interest, never items in `exclude`.
        Raise UnknownUserError if the user cannot be served."""
        ...

    def known_users(self, limit: int = 20) -> List[str]:
        """Usernames that can be served (for the demo drop-down)."""
        ...


class UnknownUserError(ValueError):
    """The user is not in the dataset or has no history."""


# Small hand-made catalogue, only used by the mock.
_CATALOGUE = {
    "LLM serving & fine-tuning": [
        ("vllm-project/vllm", "High-throughput LLM inference engine", "Python", ["llm", "inference"]),
        ("huggingface/peft", "Parameter-efficient fine-tuning", "Python", ["llm", "fine-tuning"]),
        ("unslothai/unsloth", "Faster LLM fine-tuning", "Python", ["llm", "fine-tuning"]),
        ("sgl-project/sglang", "Fast serving framework for LLMs", "Python", ["llm", "inference"]),
        ("ggml-org/llama.cpp", "LLM inference in C/C++", "C++", ["llm", "inference"]),
    ],
    "AI agents & MCP": [
        ("langchain-ai/langgraph", "Build agents as graphs", "Python", ["agents"]),
        ("modelcontextprotocol/servers", "Model Context Protocol servers", "TypeScript", ["mcp"]),
        ("crewAIInc/crewAI", "Multi-agent framework", "Python", ["agents"]),
        ("microsoft/autogen", "Multi-agent conversation framework", "Python", ["agents"]),
        ("pydantic/pydantic-ai", "Agent framework by Pydantic", "Python", ["agents"]),
    ],
    "Python web backends": [
        ("fastapi/fastapi", "Modern Python web framework", "Python", ["api", "web"]),
        ("pydantic/pydantic", "Data validation with type hints", "Python", ["validation"]),
        ("encode/httpx", "Async HTTP client", "Python", ["http"]),
        ("sqlalchemy/sqlalchemy", "SQL toolkit and ORM", "Python", ["database"]),
        ("astral-sh/uv", "Fast Python package manager", "Rust", ["packaging"]),
    ],
    "DevOps & containers": [
        ("docker/compose", "Multi-container Docker apps", "Go", ["docker"]),
        ("kubernetes/kubernetes", "Container orchestration", "Go", ["k8s"]),
        ("jesseduffield/lazydocker", "Terminal UI for Docker", "Go", ["docker", "tui"]),
        ("hashicorp/terraform", "Infrastructure as code", "Go", ["iac"]),
        ("prometheus/prometheus", "Monitoring and alerting", "Go", ["monitoring"]),
    ],
    "Vector search & RAG": [
        ("facebookresearch/faiss", "Similarity search on dense vectors", "C++", ["vector-search"]),
        ("qdrant/qdrant", "Vector database", "Rust", ["vector-db"]),
        ("chroma-core/chroma", "Embedding database", "Python", ["vector-db"]),
        ("run-llama/llama_index", "Data framework for LLM apps", "Python", ["rag"]),
        ("UKPLab/sentence-transformers", "Text embeddings", "Python", ["embeddings"]),
    ],
    "Data engineering": [
        ("duckdb/duckdb", "In-process analytical database", "C++", ["sql", "olap"]),
        ("pola-rs/polars", "Fast DataFrame library", "Rust", ["dataframe"]),
        ("apache/arrow", "Columnar in-memory format", "C++", ["data"]),
        ("dbt-labs/dbt-core", "Data transformation", "Python", ["sql"]),
        ("apache/airflow", "Workflow orchestration", "Python", ["orchestration"]),
    ],
}


class MockRecommender:
    """Fake recommendations, deterministic per username.

    Same username -> same answer (useful for tests and demos);
    different usernames -> different interests and rankings.
    """
    model_version = "mock-0.1"
    is_mock = True
    score_type = "mock"

    def known_users(self, limit: int = 20) -> List[str]:
        return ["torvalds", "gvanrossum", "MonteLion05"][:limit]

    def recommend(self, username: str, k: int = 10, exclude: Iterable[str] = ()) -> List[Interest]:
        exclude = set(exclude)
        seed = int(hashlib.sha256(username.lower().encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        now = datetime.now(timezone.utc)

        labels = rng.sample(sorted(_CATALOGUE), N_INTERESTS)
        # split k across interests (e.g. k=10 -> 3,3,2,2)
        per = [k // N_INTERESTS + (1 if i < k % N_INTERESTS else 0) for i in range(N_INTERESTS)]

        scored = []  # (score, interest_idx, repo tuple, is_new)
        evidence = {}
        for idx, label in enumerate(labels):
            repos = [r for r in _CATALOGUE[label] if r[0] not in exclude]
            rng.shuffle(repos)
            evidence[idx] = [r[0] for r in repos[per[idx]:per[idx] + 2]]
            for r in repos[: per[idx]]:
                scored.append((round(rng.uniform(0.35, 0.98), 4), idx, r, rng.random() < 0.15))

        scored.sort(key=lambda x: x[0], reverse=True)
        by_interest = {i: [] for i in range(N_INTERESTS)}
        for rank, (score, idx, (rid, desc, lang, topics), is_new) in enumerate(scored, start=1):
            created = now - timedelta(days=rng.randint(1, 6)) if is_new else \
                now - timedelta(days=rng.randint(200, 3000))
            card = RepoCard(
                repo_id=rid, description=desc, language=lang, topics=topics,
                stars=rng.randint(0, 40) if is_new else rng.randint(2_000, 90_000),
                created_at=created, url=f"https://github.com/{rid}",
            )
            by_interest[idx].append(RecommendedRepo(rank=rank, score=score, is_new=is_new, repo=card))

        return [
            Interest(
                interest_id=i, label=labels[i],
                explanation=f"Because you work on {labels[i]}",
                evidence=evidence[i], recommendations=by_interest[i],
            )
            for i in range(N_INTERESTS) if by_interest[i]
        ]


def get_recommender(name: str = "mock") -> Recommender:
    if name == "mock":
        return MockRecommender()
    if name == "model":
        import os
        from .model_recommender import ModelRecommender   # imported here so the mock needs no torch
        return ModelRecommender(os.getenv("AQ_CHECKPOINT", "models/ckpt_loo_c4.pt"),
                                os.getenv("AQ_DATA_DIR") or None)
    raise ValueError(f"Unknown recommender '{name}'")
