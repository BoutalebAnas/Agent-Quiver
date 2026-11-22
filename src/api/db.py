"""SQLite storage for user feedback (POST /log_interaction).

One file, no server to install. Path: env AQ_DB (default data/agentquiver.db).
Table `interactions`: id, user_id, repo_id, interest_id, rank, action, timestamp (ISO, UTC).
These rows are new interactions that can be appended to the history for the next training run.
"""
import os
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    NOT NULL,
    repo_id     TEXT    NOT NULL,
    interest_id INTEGER,
    rank        INTEGER,
    action      TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id);
"""


class InteractionStore:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or os.getenv("AQ_DB", "data/agentquiver.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    RATINGS = ("useful", "not_useful")

    def add(self, user_id: str, repo_id: str, interest_id: Optional[int], rank: Optional[int],
            action: str, timestamp: str) -> int:
        """Insert a feedback row. A user has at most ONE rating (useful / not_useful) per item:
        rating the same item again updates that row instead of adding a duplicate."""
        with self._lock, self._connect() as con:
            if action in self.RATINGS:
                row = con.execute(
                    "SELECT id FROM interactions WHERE user_id = ? AND repo_id = ? AND action IN (?, ?)",
                    (user_id, repo_id, *self.RATINGS)).fetchone()
                if row:
                    con.execute("UPDATE interactions SET action = ?, interest_id = ?, rank = ?, timestamp = ? "
                                "WHERE id = ?", (action, interest_id, rank, timestamp, row["id"]))
                    return int(row["id"])
            cur = con.execute(
                "INSERT INTO interactions (user_id, repo_id, interest_id, rank, action, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, repo_id, interest_id, rank, action, timestamp),
            )
            return int(cur.lastrowid)

    def rated_items(self, user_id: str) -> set:
        """Items the user already rated useful / not_useful (hidden from new recommendations)."""
        with self._connect() as con:
            rows = con.execute("SELECT repo_id FROM interactions WHERE user_id = ? AND action IN (?, ?)",
                               (user_id, *self.RATINGS))
            return {r["repo_id"] for r in rows}

    def list(self, user_id: Optional[str] = None, limit: int = 100) -> List[dict]:
        sql = "SELECT * FROM interactions"
        args: tuple = ()
        if user_id is not None:
            sql += " WHERE user_id = ?"
            args = (user_id,)
        sql += " ORDER BY id DESC LIMIT ?"
        with self._connect() as con:
            return [dict(r) for r in con.execute(sql, args + (limit,))]
