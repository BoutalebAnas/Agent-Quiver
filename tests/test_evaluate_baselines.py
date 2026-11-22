"""Smoke tests for the baseline evaluation and sample-integrity utilities."""

import numpy as np

from scripts.evaluate_baselines import integrity_checks, sample_key


def test_sample_key_uses_timestamp_target_and_user():
    s = (12, np.array([2]), np.array([1.]), 3., 5, None)
    assert sample_key(s) == (12, 3.0, 5)


def test_leakage_checks_detect_future_and_overlap():
    train = (1, np.array([2]), np.array([5.]), 7., 3, None)
    test = (1, np.array([2]), np.array([8.]), 7., 3, {2})
    res = integrity_checks({"train": [train], "valid": [], "test": [test]})
    assert res["train_test_duplicate_keys"] == 1
    assert res["future_history_samples"]["test"] == 1


def test_clean_example_passes_structural_checks():
    train = (1, np.array([2]), np.array([1.]), 2., 3, None)
    test = (1, np.array([2]), np.array([1.]), 4., 5, {2})
    res = integrity_checks({"train": [train], "valid": [], "test": [test]})
    assert res["train_test_duplicate_keys"] == 0
    assert res["future_history_samples"]["test"] == 0
