"""Rules of the PyPI cleaning: metadata gaps, temporary removals, dated adoptions, migrations.

    pytest -q tests/test_pypi_events.py
"""
from src.pypi_pipeline import pypi_events as D


def toy_project():
    rel = [("2024-01-01", "1.0", frozenset({"a", "b"})),
           ("2024-06-01", "1.1", frozenset({"a", "b", "c"})),
           ("2025-03-01", "1.2", frozenset({"a", "c", "d"})),     # drops b, adopts d
           ("2025-09-01", "2.0", frozenset({"a", "c", "d", "e"}))]
    return D.Project("toy", "", "toy", rel)


def test_events_and_migrations():
    adoptions, drops, migrations = D.events_of(toy_project())
    assert [(d, undated) for _, d, undated in adoptions] == [("a", True), ("b", True), ("c", False), ("d", False), ("e", False)]
    assert drops == [("2025-03-01", "b")]
    assert migrations == [("2025-03-01", "b", "d")]


def test_metadata_gap_is_not_a_drop_and_adoption_is_not_dated():
    rel = [("2024-01-01", "1.0", frozenset({"a", "b", "c"})),
           ("2024-03-01", "1.1", None),                         # no metadata
           ("2024-06-01", "1.2", frozenset()),                   # empty glitch between non-empty releases
           ("2024-09-01", "1.3", frozenset({"a", "b", "c", "x"})),
           ("2024-12-01", "1.4", frozenset({"a", "b", "c", "x", "y"}))]
    p = D.Project("gap", "", "gap", D._mark_glitches(rel))
    adoptions, drops, migrations = D.events_of(p)
    assert drops == []                                          # nothing dropped because of missing metadata
    assert ("2024-09-01", "x", True) in adoptions              # after a gap: known, not dated
    assert ("2024-12-01", "y", False) in adoptions             # dated
    assert migrations == []


def test_temporary_removal_is_not_a_drop():
    rel = [("2024-01-01", "1.0", frozenset({"a", "b", "c"})),
           ("2024-02-01", "1.1", frozenset({"a", "c", "d"})),
           ("2024-03-01", "1.2", frozenset({"a", "b", "c", "d"}))]
    adoptions, drops, migrations = D.events_of(D.Project("t", "", "t", rel))
    assert drops == [] and migrations == []


def test_big_rewrite_is_not_a_migration():
    rel = [("2024-01-01", "1.0", frozenset({"a", "b", "c", "d", "e"})),
           ("2024-02-01", "2.0", frozenset({"a", "v", "w", "x", "y"}))]   # 4 drops, 4 adds
    _, drops, migrations = D.events_of(D.Project("r", "", "r", rel))
    assert len(drops) == 4 and migrations == []


def test_small_project_has_no_migration():
    rel = [("2024-01-01", "1.0", frozenset({"a", "b"})),       # fewer than MIN_DEPS dependencies
           ("2024-02-01", "1.1", frozenset({"a", "c"}))]
    _, _, migrations = D.events_of(D.Project("s", "", "s", rel))
    assert migrations == []
