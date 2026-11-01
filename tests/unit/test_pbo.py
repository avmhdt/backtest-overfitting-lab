import json
import math
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

import overfitlab as ol
from overfitlab import pbo as pbo_module
from overfitlab.triallog import read_wide_returns

FIXTURE = Path(__file__).parents[1] / "fixtures" / "pbo_s4.csv"


def reference_pbo(returns, n_partitions, metric):
    """Pure-NumPy CSCV, written from the paper's steps."""
    n_obs, n_strategies = returns.shape
    size = n_obs // n_partitions
    blocks = [np.arange(b * size, (b + 1) * size) for b in range(n_partitions)]
    ranks = []
    is_perf = []
    oos_perf = []
    for chosen in combinations(range(n_partitions), n_partitions // 2):
        rest = [b for b in range(n_partitions) if b not in chosen]
        is_rows = np.concatenate([blocks[b] for b in chosen])
        oos_rows = np.concatenate([blocks[b] for b in rest])
        is_values = metric(returns[is_rows])
        oos_values = metric(returns[oos_rows])
        winner = int(np.argmax(is_values))
        below = (oos_values < oos_values[winner]).sum()
        equal = (oos_values == oos_values[winner]).sum()
        ranks.append(below + (equal + 1) / 2)
        is_perf.append(is_values[winner])
        oos_perf.append(oos_values[winner])
    omega = np.array(ranks) / (n_strategies + 1)
    return np.log(omega / (1 - omega)), np.array(is_perf), np.array(oos_perf)


def mean_metric(values):
    return values.mean(axis=0)


def sharpe_metric(values):
    return values.mean(axis=0) / values.std(axis=0, ddof=1)


def test_hand_case_with_four_blocks():
    _, matrix = read_wide_returns(FIXTURE)
    result = ol.pbo_cscv(matrix, n_partitions=4, metric="mean")
    # Worked by hand in docs/validation.md: logits 0, 0, ln 3, ln 3, -ln 3, 0.
    assert result.n_combinations == 6
    assert result.pbo == pytest.approx(4 / 6, abs=1e-15)
    third = math.log(3.0)
    assert result.logits == pytest.approx([0.0, 0.0, third, third, -third, 0.0], abs=1e-15)
    assert result.is_perf == pytest.approx([4.5, 4.0, 3.5, 3.5, 4.5, 3.5])
    assert result.oos_perf == pytest.approx([2.5, 3.0, 3.5, 3.5, 0.5, 1.5])


@pytest.mark.parametrize(("name", "metric"), [("sharpe", sharpe_metric), ("mean", mean_metric)])
def test_kernel_matches_the_reference(name, metric):
    returns = np.random.default_rng(0).normal(0.0, 0.01, (403, 25))
    result = ol.pbo_cscv(returns, n_partitions=8, metric=name)
    logits, is_perf, oos_perf = reference_pbo(returns, 8, metric)
    assert result.logits == pytest.approx(logits, abs=1e-12)
    assert result.is_perf == pytest.approx(is_perf, abs=1e-10)
    assert result.oos_perf == pytest.approx(oos_perf, abs=1e-10)


def test_jitted_and_python_runs_agree():
    code = (
        "import json, numpy as np, overfitlab as ol;"
        "r = np.random.default_rng(5).normal(0, 0.01, (320, 30));"
        "x = ol.pbo_cscv(r, n_partitions=8);"
        "print(json.dumps(x.logits.tolist() + x.is_perf.tolist() + x.oos_perf.tolist()))"
    )
    outputs = []
    for flag in ("0", "1"):
        env = {**os.environ, "NUMBA_DISABLE_JIT": flag}
        run = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        outputs.append(np.array(json.loads(run.stdout)))
    assert np.allclose(outputs[0], outputs[1], rtol=1e-12, atol=1e-12)


def test_callable_metric_matches_the_named_one():
    returns = np.random.default_rng(1).normal(0.0, 0.01, (160, 10))
    named = ol.pbo_cscv(returns, n_partitions=8, metric="mean")
    custom = ol.pbo_cscv(returns, n_partitions=8, metric=mean_metric)
    assert custom.logits == pytest.approx(named.logits, abs=1e-12)


def test_noise_gives_about_one_half_on_average():
    # A single path is very noisy, so this averages over paths.
    generator = np.random.default_rng(2)
    values = [
        ol.pbo_cscv(generator.normal(0.0, 0.01, (400, 40)), n_partitions=8).pbo for _ in range(40)
    ]
    assert np.mean(values) == pytest.approx(0.5, abs=0.1)


def test_strong_signal_gives_a_low_pbo():
    result = ol.planted_signal_matrix(1260, 50, true_sharpe_annual=4.0, rng=0)
    assert ol.pbo_cscv(result.returns, n_partitions=8).pbo < 0.05


def test_degradation_is_negative_on_noise():
    returns = np.random.default_rng(3).normal(0.0, 0.01, (1000, 100))
    assert ol.pbo_cscv(returns, n_partitions=10).degradation_slope < 0.0


def test_probability_of_loss_on_a_losing_set():
    returns = np.random.default_rng(4).normal(-0.01, 0.002, (160, 10))
    assert ol.pbo_cscv(returns, n_partitions=8).prob_oos_loss == 1.0


def test_constant_performance_has_no_slope():
    # Every combination selects the same in-sample value, so the regression has no spread.
    result = ol.pbo_cscv(np.ones((8, 2)), n_partitions=2, metric="mean")
    assert result.degradation_slope == 0.0


def test_rows_past_the_last_block_are_dropped():
    returns = np.random.default_rng(5).normal(0.0, 0.01, (80, 5))
    longer = np.vstack((returns, np.ones((3, 5))))
    assert ol.pbo_cscv(longer, n_partitions=8).pbo == ol.pbo_cscv(returns, n_partitions=8).pbo


def test_flat_strategy_scores_zero():
    sums = np.zeros((2, 1))
    out = np.ones(1)
    blocks = np.array([0, 1])
    pbo_module._metric_values(sums, sums.copy(), blocks, 2, True, out)
    assert out[0] == 0.0


def test_average_rank_on_ties():
    assert pbo_module._winner_rank(np.array([1.0, 2.0, 2.0, 3.0]), 1) == 2.5


@pytest.mark.parametrize(
    ("shape", "kwargs", "message"),
    [
        ((100, 5), {"n_partitions": 3}, "even"),
        ((100, 5), {"n_partitions": 0}, "even"),
        ((100, 1), {}, "two trials"),
        ((10, 5), {"n_partitions": 16}, "too few"),
        ((100, 5), {"metric": "sortino"}, "metric"),
    ],
)
def test_argument_validation(shape, kwargs, message):
    returns = np.random.default_rng(0).normal(size=shape)
    with pytest.raises(ValueError, match=message):
        ol.pbo_cscv(returns, **kwargs)
