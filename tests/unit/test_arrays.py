import numpy as np
import polars as pl
import pytest

from overfitlab._arrays import as_float_array, as_matrix, as_vector


def test_list_and_array_agree():
    assert np.array_equal(as_float_array([[1, 2], [3, 4]]), np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_output_is_float64():
    assert as_float_array(np.array([1, 2], dtype=np.int32)).dtype == np.float64


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_rejected_with_argument_name(bad):
    with pytest.raises(ValueError, match="rets contains NaN or inf"):
        as_float_array([1.0, bad], name="rets")


def test_allow_nan_keeps_values():
    assert np.isnan(as_float_array([np.nan], allow_nan=True)[0])


def test_polars_series_parity():
    values = [0.01, -0.02, 0.03]
    assert np.array_equal(as_vector(pl.Series(values)), np.array(values))


def test_polars_dataframe_parity():
    frame = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert np.array_equal(as_matrix(frame), np.array([[1.0, 3.0], [2.0, 4.0]]))


def test_vector_flattens_single_column():
    assert as_vector([[1.0], [2.0]]).shape == (2,)


def test_vector_rejects_matrix():
    with pytest.raises(ValueError, match="one-dimensional"):
        as_vector([[1.0, 2.0], [3.0, 4.0]])


def test_matrix_promotes_vector_to_column():
    assert as_matrix([1.0, 2.0, 3.0]).shape == (3, 1)


def test_matrix_rejects_three_dimensions():
    with pytest.raises(ValueError, match="two-dimensional"):
        as_matrix(np.zeros((2, 2, 2)))
