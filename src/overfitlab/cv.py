"""Purged k-fold and combinatorial purged cross-validation.

Implements the schemes of Lopez de Prado (2018, chapters 7 and 12). A label observed at
row ``i`` spans the interval ``[t0[i], t1[i]]``, so two rows can share information even when
they are not adjacent. The splitters follow the scikit-learn ``split`` protocol, so they can
be passed as ``cv=`` without importing scikit-learn.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]


def _label_times(t1: ArrayLike, t0: ArrayLike | None) -> tuple[I64, I64]:
    """Return ``t0`` and ``t1`` as int64, with integer or datetime64 inputs."""
    end = np.asarray(t1)
    if end.ndim != 1:
        raise ValueError("t1 must be one-dimensional")
    is_datetime = np.issubdtype(end.dtype, np.datetime64)
    if t0 is None:
        if is_datetime:
            raise ValueError("t0 is required when t1 holds datetimes")
        start = np.arange(end.size)
    else:
        start = np.asarray(t0)
    if start.shape != end.shape:
        raise ValueError("t0 and t1 differ in length")
    if is_datetime != np.issubdtype(start.dtype, np.datetime64):
        raise ValueError("t0 and t1 must both be integers or both be datetimes")
    if is_datetime:
        start = start.astype("datetime64[ns]").astype(np.int64)
        end = end.astype("datetime64[ns]").astype(np.int64)
    elif not (np.issubdtype(start.dtype, np.integer) and np.issubdtype(end.dtype, np.integer)):
        raise ValueError("t0 and t1 must be integers or datetime64")
    if np.any(np.diff(start) < 0):
        raise ValueError("t0 must be sorted in ascending order")
    if np.any(end < start):
        raise ValueError("t1 must not precede t0")
    return start.astype(np.int64), end.astype(np.int64)


def _check_embargo(embargo_pct: float) -> None:
    if not 0.0 <= embargo_pct < 1.0:
        raise ValueError("embargo_pct must be in [0, 1)")


def _contiguous_blocks(test: I64) -> list[I64]:
    """Split sorted row numbers into runs of consecutive rows."""
    return np.split(test, np.flatnonzero(np.diff(test) > 1) + 1)


def _train_rows(test: I64, t0: I64, t1: I64, embargo: int) -> I64:
    """Return the rows left after removing, purging and embargoing around ``test``.

    A row is purged if its label interval overlaps any test label interval. The embargo
    covers ``embargo`` rows after each test block, counted from the first row whose label
    starts after that block's labels end, so that it adds to the purge instead of falling
    inside it.
    """
    n_rows = t0.size
    keep = np.ones(n_rows, dtype=bool)
    keep[test] = False
    for block in _contiguous_blocks(test):
        block_end = int(t1[block].max())
        keep &= (t1 < int(t0[block[0]])) | (t0 > block_end)
        first_after = int(np.searchsorted(t0, block_end, side="right"))
        keep[first_after : first_after + embargo] = False
    return np.flatnonzero(keep).astype(np.int64)


class PurgedKFold:
    """K-fold over contiguous test folds, with purging and an embargo.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds, at least 2.
    t1 : array_like
        End of each label interval, integers or ``datetime64``.
    t0 : array_like, optional
        Start of each label interval, ascending. Defaults to row positions, which needs
        integer ``t1``.
    embargo_pct : float, default 0.01
        Embargo length as a share of the rows: ``ceil(embargo_pct * T)`` rows.

    Raises
    ------
    ValueError
        If an argument is out of range or the label times are inconsistent.

    Examples
    --------
    >>> t1 = np.arange(10) + 2
    >>> cv = PurgedKFold(2, t1=t1, embargo_pct=0.0)
    >>> [train.tolist() for train, _ in cv.split(np.zeros(10))]
    [[7, 8, 9], [0, 1, 2]]
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        t1: ArrayLike,
        t0: ArrayLike | None = None,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        _check_embargo(embargo_pct)
        self.n_splits = n_splits
        self.t0, self.t1 = _label_times(t1, t0)
        self.embargo_pct = embargo_pct
        if self.t0.size < n_splits:
            raise ValueError("fewer rows than folds")

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        """Return the number of folds.

        Parameters
        ----------
        X, y, groups : object, optional
            Ignored, for scikit-learn compatibility.

        Returns
        -------
        int
            Number of folds.
        """
        return self.n_splits

    def split(
        self, X: ArrayLike, y: object = None, groups: object = None
    ) -> Iterator[tuple[I64, I64]]:
        """Yield train and test row numbers for each fold.

        Parameters
        ----------
        X : array_like
            Data, used only for its length.
        y, groups : object, optional
            Ignored, for scikit-learn compatibility.

        Yields
        ------
        train, test : numpy.ndarray
            Row numbers.

        Raises
        ------
        ValueError
            If ``X`` and ``t1`` differ in length.
        """
        n_rows = len(X)  # type: ignore[arg-type]
        if n_rows != self.t1.size:
            raise ValueError("X and t1 differ in length")
        embargo = math.ceil(self.embargo_pct * n_rows)
        for test in np.array_split(np.arange(n_rows, dtype=np.int64), self.n_splits):
            yield _train_rows(test, self.t0, self.t1, embargo), test


class CombinatorialPurgedCV:
    """Combinatorial purged cross-validation.

    Rows are divided into ``N`` contiguous groups and every combination of ``k`` groups is
    a test set, so there are ``C(N, k)`` splits. Each group is tested ``k/N * C(N, k)``
    times, which yields that many complete backtest paths.

    Parameters
    ----------
    n_groups : int, default 6
        Number of groups ``N``.
    n_test_groups : int, default 2
        Number of test groups ``k``, below ``N``.
    t1 : array_like
        End of each label interval, integers or ``datetime64``.
    t0 : array_like, optional
        Start of each label interval, ascending. Defaults to row positions.
    embargo_pct : float, default 0.01
        Embargo length as a share of the rows.

    Raises
    ------
    ValueError
        If an argument is out of range or the label times are inconsistent.

    Examples
    --------
    >>> cv = CombinatorialPurgedCV(6, 2, t1=np.arange(60))
    >>> cv.n_splits, cv.n_paths
    (15, 5)
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        *,
        t1: ArrayLike,
        t0: ArrayLike | None = None,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_groups < 2:
            raise ValueError("n_groups must be at least 2")
        if not 1 <= n_test_groups < n_groups:
            raise ValueError("n_test_groups must be at least 1 and below n_groups")
        _check_embargo(embargo_pct)
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.t0, self.t1 = _label_times(t1, t0)
        self.embargo_pct = embargo_pct
        if self.t0.size < n_groups:
            raise ValueError("fewer rows than groups")
        self._groups = np.array_split(np.arange(self.t0.size, dtype=np.int64), n_groups)
        self._combinations = list(combinations(range(n_groups), n_test_groups))

    @property
    def n_splits(self) -> int:
        """Number of splits, ``C(N, k)``."""
        return math.comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """Number of complete backtest paths, ``k/N * C(N, k)``."""
        return self.n_splits * self.n_test_groups // self.n_groups

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        """Return the number of splits.

        Parameters
        ----------
        X, y, groups : object, optional
            Ignored, for scikit-learn compatibility.

        Returns
        -------
        int
            Number of splits.
        """
        return self.n_splits

    def split(
        self, X: ArrayLike, y: object = None, groups: object = None
    ) -> Iterator[tuple[I64, I64]]:
        """Yield train and test row numbers for each split.

        Parameters
        ----------
        X : array_like
            Data, used only for its length.
        y, groups : object, optional
            Ignored, for scikit-learn compatibility.

        Yields
        ------
        train, test : numpy.ndarray
            Row numbers.

        Raises
        ------
        ValueError
            If ``X`` and ``t1`` differ in length.
        """
        n_rows = len(X)  # type: ignore[arg-type]
        if n_rows != self.t1.size:
            raise ValueError("X and t1 differ in length")
        embargo = math.ceil(self.embargo_pct * n_rows)
        for chosen in self._combinations:
            test = np.concatenate([self._groups[group] for group in chosen])
            yield _train_rows(test, self.t0, self.t1, embargo), test

    def path_map(self) -> I64:
        """Return which split supplies each group of each path.

        Returns
        -------
        numpy.ndarray
            Integer array of shape ``(n_paths, n_groups)`` holding split ids.

        Examples
        --------
        >>> CombinatorialPurgedCV(3, 2, t1=np.arange(9)).path_map().tolist()
        [[0, 0, 1], [1, 2, 2]]
        """
        mapping = np.zeros((self.n_paths, self.n_groups), dtype=np.int64)
        seen = [0] * self.n_groups
        for split_id, chosen in enumerate(self._combinations):
            for group in chosen:
                mapping[seen[group], group] = split_id
                seen[group] += 1
        return mapping

    def group_rows(self, group: int) -> I64:
        """Return the row numbers of a group.

        Parameters
        ----------
        group : int
            Group id.

        Returns
        -------
        numpy.ndarray
            Row numbers.
        """
        return self._groups[group]

    def test_groups(self, split_id: int) -> tuple[int, ...]:
        """Return the test groups of a split.

        Parameters
        ----------
        split_id : int
            Split id.

        Returns
        -------
        tuple of int
            Group ids.
        """
        return self._combinations[split_id]


def assemble_paths(cv: CombinatorialPurgedCV, oos_predictions: Sequence[Any]) -> F64:
    """Assemble per-split out-of-sample predictions into complete backtest paths.

    The paths are those of combinatorial purged CV in Lopez de Prado (2018).

    Parameters
    ----------
    cv : CombinatorialPurgedCV
        The splitter that produced the predictions.
    oos_predictions : sequence of array_like
        One array per split, aligned with that split's test rows.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_paths, T)``.

    Raises
    ------
    ValueError
        If the number of arrays or the length of one is wrong.

    Examples
    --------
    >>> cv = CombinatorialPurgedCV(3, 2, t1=np.arange(3), embargo_pct=0.0)
    >>> preds = [test.astype(float) * 10 for _, test in cv.split(np.zeros(3))]
    >>> assemble_paths(cv, preds).tolist()
    [[0.0, 10.0, 20.0], [0.0, 10.0, 20.0]]
    """
    if len(oos_predictions) != cv.n_splits:
        raise ValueError(f"expected {cv.n_splits} prediction arrays")
    paths = np.zeros((cv.n_paths, cv.t0.size), dtype=np.float64)
    mapping = cv.path_map()
    for path in range(cv.n_paths):
        for group in range(cv.n_groups):
            split_id = int(mapping[path, group])
            chosen = cv.test_groups(split_id)
            values = np.asarray(oos_predictions[split_id], dtype=np.float64)
            expected = sum(cv.group_rows(item).size for item in chosen)
            if values.size != expected:
                raise ValueError(f"split {split_id} needs {expected} predictions")
            offset = sum(cv.group_rows(item).size for item in chosen[: chosen.index(group)])
            rows = cv.group_rows(group)
            paths[path, rows] = values[offset : offset + rows.size]
    return paths
