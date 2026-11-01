"""Input coercion shared by every public function.

Array arguments accept NumPy arrays, nested lists and, when Polars is installed,
``pl.Series`` and ``pl.DataFrame``. Polars is duck-typed through ``.to_numpy()`` so
it is never imported here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


def as_float_array(x: Any, *, name: str = "x", allow_nan: bool = False) -> F64:
    """Convert ``x`` to a ``float64`` array and reject non-finite values.

    Parameters
    ----------
    x : array_like
        NumPy array, nested list or an object with a ``to_numpy()`` method.
    name : str, default "x"
        Argument name used in error messages.
    allow_nan : bool, default False
        Keep NaN and inf. Only functions that document their NaN handling set this.

    Returns
    -------
    numpy.ndarray
        Array of dtype ``float64``.

    Raises
    ------
    ValueError
        If ``x`` holds NaN or inf and ``allow_nan`` is false.

    Examples
    --------
    >>> as_float_array([1, 2, 3]).tolist()
    [1.0, 2.0, 3.0]
    """
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    arr = np.asarray(x, dtype=np.float64)
    if not allow_nan and not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def as_vector(x: Any, *, name: str = "x", allow_nan: bool = False) -> F64:
    """Convert ``x`` to a one-dimensional ``float64`` array.

    A single-column matrix is flattened, so a one-column ``pl.DataFrame`` is accepted.

    Parameters
    ----------
    x : array_like
        Input values.
    name : str, default "x"
        Argument name used in error messages.
    allow_nan : bool, default False
        Keep NaN and inf.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of dtype ``float64``.

    Raises
    ------
    ValueError
        If ``x`` is not one-dimensional or holds non-finite values.

    Examples
    --------
    >>> as_vector([[1.0], [2.0]]).tolist()
    [1.0, 2.0]
    """
    arr = as_float_array(x, name=name, allow_nan=allow_nan)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr


def as_matrix(x: Any, *, name: str = "returns", allow_nan: bool = False) -> F64:
    """Convert ``x`` to a two-dimensional ``float64`` array of shape ``(T, N)``.

    A one-dimensional input becomes a single column.

    Parameters
    ----------
    x : array_like
        Input values, rows are observations and columns are trials.
    name : str, default "returns"
        Argument name used in error messages.
    allow_nan : bool, default False
        Keep NaN and inf.

    Returns
    -------
    numpy.ndarray
        Two-dimensional array of dtype ``float64``.

    Raises
    ------
    ValueError
        If ``x`` has more than two dimensions or holds non-finite values.

    Examples
    --------
    >>> as_matrix([1.0, 2.0]).shape
    (2, 1)
    """
    arr = as_float_array(x, name=name, allow_nan=allow_nan)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {arr.shape}")
    return arr
