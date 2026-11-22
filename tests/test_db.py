"""SQLite feedback store: rows are written to disk and survive a restart (new store on same file)."""
from src.api.db import InteractionStore


def test_rows_persist_after_restart(tmp_path):
    path = tmp_path / "aq.db"
    s1 = InteractionStore(str(path))
    i = s1.add("alice", "pypi:requests", 0, 1, "useful", "2026-11-22T12:00:00+00:00")
    s2 = InteractionStore(str(path))          # simulates an API restart
    rows = s2.list("alice")
    assert rows[0]["id"] == i and rows[0]["repo_id"] == "pypi:requests" and rows[0]["action"] == "useful"
    assert s2.list("bob") == []


def test_same_item_rated_twice_gives_one_row(tmp_path):
    s = InteractionStore(str(tmp_path / "aq.db"))
    a = s.add("alice", "pypi:requests", 0, 1, "useful", "2026-11-22T12:00:00+00:00")
    b = s.add("alice", "pypi:requests", 0, 1, "useful", "2026-11-22T12:00:01+00:00")
    c = s.add("alice", "pypi:requests", 0, 1, "not_useful", "2026-11-22T12:00:02+00:00")   # changes mind
    rows = s.list("alice")
    assert a == b == c and len(rows) == 1 and rows[0]["action"] == "not_useful"
    assert s.rated_items("alice") == {"pypi:requests"}
