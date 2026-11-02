import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import overfitlab as ol


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 10_000), st.integers(2, 12), st.sampled_from([2, 4, 6]))
def test_pbo_is_a_probability(seed, n_strategies, n_partitions):
    returns = np.random.default_rng(seed).normal(0.0, 0.01, (48, n_strategies))
    result = ol.pbo_cscv(returns, n_partitions=n_partitions)
    assert 0.0 <= result.pbo <= 1.0
    assert ((result.oos_rank > 0.0) & (result.oos_rank < 1.0)).all()
