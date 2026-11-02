import math

import numpy as np
import pytest
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

import overfitlab as ol


def fold_rows(cv, n_rows):
    return list(cv.split(np.zeros(n_rows)))


def test_folds_are_contiguous_and_cover_every_row():
    cv = ol.PurgedKFold(4, t1=np.arange(103), embargo_pct=0.0)
    tests = [test for _, test in fold_rows(cv, 103)]
    assert np.array_equal(np.concatenate(tests), np.arange(103))
    assert all((np.diff(test) == 1).all() for test in tests)


def test_point_labels_lose_only_the_test_rows():
    cv = ol.PurgedKFold(5, t1=np.arange(100), embargo_pct=0.0)
    for train, test in fold_rows(cv, 100):
        assert train.size + test.size == 100


def test_rows_before_the_fold_are_purged():
    t1 = np.arange(100) + 5
    train, test = fold_rows(ol.PurgedKFold(5, t1=t1, embargo_pct=0.0), 100)[1]
    assert test[0] == 20
    assert train[train < 20].max() == 14


def test_rows_after_the_fold_are_purged():
    t1 = np.arange(100) + 5
    train, test = fold_rows(ol.PurgedKFold(5, t1=t1, embargo_pct=0.0), 100)[1]
    assert test[-1] == 39
    assert train[train > 39].min() == 45


def test_embargo_extends_the_purge():
    t1 = np.arange(100) + 5
    train, _ = fold_rows(ol.PurgedKFold(5, t1=t1, embargo_pct=0.03), 100)[1]
    assert train[train > 39].min() == 48


def test_embargo_uses_the_ceiling():
    t1 = np.arange(101)
    train, _ = fold_rows(ol.PurgedKFold(5, t1=t1, embargo_pct=0.01), 101)[0]
    assert train.min() == 21 + 2


def test_embargo_applies_only_after_the_fold():
    t1 = np.arange(100)
    train, test = fold_rows(ol.PurgedKFold(5, t1=t1, embargo_pct=0.05), 100)[1]
    assert train[train < test[0]].max() == test[0] - 1


def test_datetime_labels_match_integer_labels():
    days = np.datetime64("2024-01-01", "D") + np.arange(60)
    by_date = ol.PurgedKFold(3, t0=days, t1=days + np.timedelta64(4, "D"), embargo_pct=0.02)
    by_row = ol.PurgedKFold(3, t1=np.arange(60) + 4, embargo_pct=0.02)
    for (a_train, a_test), (b_train, b_test) in zip(
        fold_rows(by_date, 60), fold_rows(by_row, 60), strict=True
    ):
        assert np.array_equal(a_train, b_train)
        assert np.array_equal(a_test, b_test)


def test_get_n_splits():
    assert ol.PurgedKFold(4, t1=np.arange(10)).get_n_splits() == 4


def test_works_as_a_scikit_learn_cv():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 2))
    labels = rng.integers(0, 2, 60)
    cv = ol.PurgedKFold(3, t1=np.arange(60) + 2)
    scores = cross_val_score(DecisionTreeClassifier(random_state=0), features, labels, cv=cv)
    assert scores.size == 3
    cpcv = ol.CombinatorialPurgedCV(4, 2, t1=np.arange(60) + 2)
    scores = cross_val_score(DecisionTreeClassifier(random_state=0), features, labels, cv=cpcv)
    assert scores.size == 6


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_splits": 1, "t1": np.arange(10)}, "n_splits"),
        ({"n_splits": 2, "t1": np.arange(10), "embargo_pct": 1.0}, "embargo_pct"),
        ({"n_splits": 20, "t1": np.arange(10)}, "fewer rows"),
        ({"n_splits": 2, "t1": np.arange(10).reshape(5, 2)}, "one-dimensional"),
        ({"n_splits": 2, "t1": np.arange(10), "t0": np.arange(9)}, "differ in length"),
        ({"n_splits": 2, "t1": np.arange(10) + 1, "t0": np.arange(10)[::-1]}, "sorted"),
        ({"n_splits": 2, "t1": np.arange(10) - 1}, "precede"),
        ({"n_splits": 2, "t1": np.arange(10.0)}, "integers"),
        (
            {"n_splits": 2, "t1": np.datetime64("2024-01-01", "D") + np.arange(10)},
            "t0 is required",
        ),
        (
            {
                "n_splits": 2,
                "t1": np.datetime64("2024-01-01", "D") + np.arange(10),
                "t0": np.arange(10),
            },
            "both",
        ),
    ],
)
def test_purged_kfold_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ol.PurgedKFold(**kwargs)


def test_split_rejects_wrong_length():
    with pytest.raises(ValueError, match="differ"):
        list(ol.PurgedKFold(2, t1=np.arange(10)).split(np.zeros(9)))
    with pytest.raises(ValueError, match="differ"):
        list(ol.CombinatorialPurgedCV(4, 2, t1=np.arange(10)).split(np.zeros(9)))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_groups": 1, "n_test_groups": 1}, "n_groups"),
        ({"n_groups": 4, "n_test_groups": 4}, "n_test_groups"),
        ({"n_groups": 4, "n_test_groups": 0}, "n_test_groups"),
        ({"n_groups": 4, "n_test_groups": 2, "embargo_pct": -0.1}, "embargo_pct"),
    ],
)
def test_cpcv_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ol.CombinatorialPurgedCV(t1=np.arange(10), **kwargs)


def test_cpcv_needs_enough_rows():
    with pytest.raises(ValueError, match="fewer rows"):
        ol.CombinatorialPurgedCV(6, 2, t1=np.arange(3))


@pytest.mark.parametrize(("n", "k"), [(6, 2), (10, 2), (10, 3), (5, 1), (8, 7)])
def test_cpcv_counts_match_the_formula(n, k):
    cv = ol.CombinatorialPurgedCV(n, k, t1=np.arange(n * 3))
    assert cv.n_splits == math.comb(n, k)
    assert cv.n_paths == k * math.comb(n, k) / n
    assert cv.get_n_splits() == cv.n_splits
    assert len(fold_rows(cv, n * 3)) == cv.n_splits


def test_cpcv_six_two_is_fifteen_splits_and_five_paths():
    # G-07, also in the golden cases: AFML, section 12.4.1, p. 164.
    cv = ol.CombinatorialPurgedCV(6, 2, t1=np.arange(60))
    assert (cv.n_splits, cv.n_paths) == (15, 5)


def test_path_map_uses_each_split_on_its_test_groups_only():
    cv = ol.CombinatorialPurgedCV(6, 2, t1=np.arange(60))
    mapping = cv.path_map()
    assert mapping.shape == (5, 6)
    for path in range(5):
        for group in range(6):
            assert group in cv.test_groups(int(mapping[path, group]))


def test_every_split_appears_once_per_test_group_in_the_path_map():
    cv = ol.CombinatorialPurgedCV(6, 2, t1=np.arange(60))
    counts = np.bincount(cv.path_map().ravel(), minlength=cv.n_splits)
    assert (counts == 2).all()


def test_adjacent_test_groups_are_one_block():
    cv = ol.CombinatorialPurgedCV(4, 2, t1=np.arange(40) + 2, embargo_pct=0.0)
    train, test = fold_rows(cv, 40)[0]
    assert np.array_equal(test, np.arange(20))
    assert train.min() == 22


def test_separate_test_groups_are_purged_on_both_sides():
    cv = ol.CombinatorialPurgedCV(4, 2, t1=np.arange(40) + 2, embargo_pct=0.05)
    train, test = fold_rows(cv, 40)[1]
    assert np.array_equal(test, np.r_[0:10, 20:30])
    assert np.array_equal(train, np.r_[14:18, 34:40])


def test_assemble_paths_rebuilds_each_row_from_its_test_split():
    t1 = np.arange(60) + 3
    cv = ol.CombinatorialPurgedCV(6, 2, t1=t1)
    predictions = [test * 1.0 + 1000.0 * split for split, (_, test) in enumerate(cv.split(t1))]
    paths = ol.assemble_paths(cv, predictions)
    assert paths.shape == (5, 60)
    assert np.array_equal(paths % 1000.0, np.tile(np.arange(60.0), (5, 1)))


def test_assemble_paths_checks_the_input():
    cv = ol.CombinatorialPurgedCV(4, 2, t1=np.arange(8))
    with pytest.raises(ValueError, match="prediction arrays"):
        ol.assemble_paths(cv, [])
    with pytest.raises(ValueError, match="needs"):
        ol.assemble_paths(cv, [np.zeros(1)] * cv.n_splits)
