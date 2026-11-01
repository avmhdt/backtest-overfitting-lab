import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import overfitlab as ol

ROWS = st.integers(40, 300)
HORIZON = st.integers(0, 15)
EMBARGO = st.floats(0.0, 0.05)


def overlaps(train, test, t0, t1):
    """Brute force: does any train label interval touch any test label interval?"""
    return any(
        t0[i] <= t1[test].max() and t0[test].min() <= t1[i] and _touches(i, test, t0, t1)
        for i in train
    )


def _touches(i, test, t0, t1):
    return bool(((t0[i] <= t1[test]) & (t0[test] <= t1[i])).any())


@settings(max_examples=50)
@given(ROWS, HORIZON, EMBARGO, st.integers(2, 8))
def test_purged_intervals_never_overlap(n_rows, horizon, embargo, folds):
    t0 = np.arange(n_rows)
    t1 = t0 + horizon
    for train, test in ol.PurgedKFold(folds, t1=t1, embargo_pct=embargo).split(np.zeros(n_rows)):
        assert not overlaps(train, test, t0, t1)
        assert np.intersect1d(train, test).size == 0


@settings(max_examples=50)
@given(ROWS, HORIZON, EMBARGO, st.integers(2, 8))
def test_embargo_window_is_excluded(n_rows, horizon, embargo, folds):
    t1 = np.arange(n_rows) + horizon
    width = math.ceil(embargo * n_rows)
    for train, test in ol.PurgedKFold(folds, t1=t1, embargo_pct=embargo).split(np.zeros(n_rows)):
        after = np.arange(test[-1] + 1, test[-1] + 1 + horizon + width)
        assert np.intersect1d(train, after).size == 0


@settings(max_examples=30)
@given(st.integers(3, 8), st.integers(1, 3), HORIZON, EMBARGO)
def test_cpcv_intervals_never_overlap(n_groups, k, horizon, embargo):
    k = min(k, n_groups - 1)
    n_rows = n_groups * 12
    t0 = np.arange(n_rows)
    t1 = t0 + horizon
    cv = ol.CombinatorialPurgedCV(n_groups, k, t1=t1, embargo_pct=embargo)
    for train, test in cv.split(np.zeros(n_rows)):
        assert not overlaps(train, test, t0, t1)


@settings(max_examples=30)
@given(st.integers(3, 9), st.integers(1, 4))
def test_each_group_is_tested_in_the_right_number_of_splits(n_groups, k):
    k = min(k, n_groups - 1)
    cv = ol.CombinatorialPurgedCV(n_groups, k, t1=np.arange(n_groups * 6))
    counts = np.zeros(n_groups, dtype=int)
    for split in range(cv.n_splits):
        counts[list(cv.test_groups(split))] += 1
    assert (counts == math.comb(n_groups - 1, k - 1)).all()
