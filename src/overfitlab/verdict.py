"""Verdict gate: combine the evidence into PASS, FAIL or INCONCLUSIVE."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from overfitlab._arrays import as_matrix, as_vector
from overfitlab.bootstrap import optimal_block_length, stationary_bootstrap_indices
from overfitlab.deflated import dsr_from_returns
from overfitlab.haircut import haircut_sharpe
from overfitlab.pbo import PBOResult, pbo_cscv
from overfitlab.results import DSRResult, HaircutResult
from overfitlab.sharpe import annualize_sharpe, sharpe_ratio
from overfitlab.snooping import SPAResult, StepMResult, romano_wolf_stepdown, spa_test

RNGLike = np.random.Generator | int | None
Verdict = Literal["PASS", "FAIL", "INCONCLUSIVE"]
MIN_OBS_PER_BLOCK = 20


@dataclass(frozen=True)
class VerdictConfig:
    """Thresholds of the verdict gate, with the defaults of decision D-06.

    Attributes
    ----------
    dsr_min : float
        Minimum Deflated Sharpe Ratio.
    pbo_max : float
        Maximum Probability of Backtest Overfitting.
    snoop_alpha : float
        Significance level of the SPA and stepdown tests.
    haircut_method : {"bonferroni", "holm", "bhy"}
        Multiple-testing adjustment of the haircut.
    pbo_partitions : int
        Number of CSCV partitions ``S``.
    n_boot : int
        Number of bootstrap resamples.
    periods_per_year : int
        Periods per year of the returns.
    """

    dsr_min: float = 0.95
    pbo_max: float = 0.20
    snoop_alpha: float = 0.05
    haircut_method: Literal["bonferroni", "holm", "bhy"] = "bhy"
    pbo_partitions: int = 16
    n_boot: int = 1000
    periods_per_year: int = 252


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


@dataclass(frozen=True)
class VerdictReport:
    """Outcome of the gate.

    Attributes
    ----------
    verdict : {"PASS", "FAIL", "INCONCLUSIVE"}
        Overall result.
    reasons : mapping
        Criterion to ``"pass"``, ``"fail: ..."`` or ``"not computed: ..."``.
    dsr : DSRResult
        Deflated Sharpe Ratio.
    pbo : PBOResult or None
        PBO, if the sample allows it.
    spa : SPAResult or None
        SPA test, if the sample allows it.
    stepm : StepMResult or None
        Stepdown, if the sample allows it.
    haircut : HaircutResult
        Harvey-Liu haircut.
    selected : int
        Column of the selected trial.
    """

    verdict: Verdict
    reasons: Mapping[str, str]
    dsr: DSRResult
    pbo: PBOResult | None
    spa: SPAResult | None
    stepm: StepMResult | None
    haircut: HaircutResult
    selected: int

    def to_json(self) -> str:
        """Return the report as one JSON object, with NaN as null.

        Returns
        -------
        str
            JSON text.
        """
        return json.dumps(_plain(asdict(self)), sort_keys=True)


def decide(reasons: Mapping[str, str]) -> Verdict:
    """Combine per-criterion outcomes.

    Parameters
    ----------
    reasons : mapping
        Criterion to ``"pass"``, ``"fail: ..."`` or ``"not computed: ..."``.

    Returns
    -------
    {"PASS", "FAIL", "INCONCLUSIVE"}
        FAIL if any computed criterion fails, otherwise INCONCLUSIVE if any was not
        computed, otherwise PASS.

    Examples
    --------
    >>> decide({"dsr": "pass", "pbo": "not computed: too few observations"})
    'INCONCLUSIVE'
    """
    if any(reason.startswith("fail") for reason in reasons.values()):
        return "FAIL"
    if any(reason.startswith("not computed") for reason in reasons.values()):
        return "INCONCLUSIVE"
    return "PASS"


def judge(
    config: VerdictConfig,
    *,
    dsr: DSRResult,
    pbo: PBOResult | None,
    spa: SPAResult | None,
    stepm: StepMResult | None,
    haircut_sr_annual: float | None,
    selected: int,
) -> dict[str, str]:
    """Apply the thresholds to the evidence, criterion by criterion.

    Parameters
    ----------
    config : VerdictConfig
        Thresholds.
    dsr : DSRResult
        Deflated Sharpe Ratio.
    pbo : PBOResult or None
        PBO, if computed.
    spa : SPAResult or None
        SPA test, if computed.
    stepm : StepMResult or None
        Stepdown, if computed.
    haircut_sr_annual : float or None
        Haircut Sharpe ratio, if computed.
    selected : int
        Column of the selected trial in ``stepm``.

    Returns
    -------
    dict
        Criterion to outcome.
    """
    reasons: dict[str, str] = {}
    reasons["dsr"] = (
        "pass" if dsr.dsr >= config.dsr_min else f"fail: {dsr.dsr:.4f} < {config.dsr_min}"
    )
    if pbo is None:
        reasons["pbo"] = "not computed: no aligned matrix with enough observations"
    elif pbo.pbo <= config.pbo_max:
        reasons["pbo"] = "pass"
    else:
        reasons["pbo"] = f"fail: {pbo.pbo:.4f} > {config.pbo_max}"
    if spa is None:
        reasons["spa"] = "not computed: no aligned matrix"
    elif spa.pvalue_consistent <= config.snoop_alpha:
        reasons["spa"] = "pass"
    else:
        reasons["spa"] = f"fail: consistent p {spa.pvalue_consistent:.4f} > {config.snoop_alpha}"
    if stepm is None:
        reasons["stepm"] = "not computed: no aligned matrix"
    elif stepm.rejected[selected]:
        reasons["stepm"] = "pass"
    else:
        reasons["stepm"] = "fail: the selected trial is not rejected"
    if haircut_sr_annual is None:
        reasons["haircut"] = "not computed: the sample is too short"
    elif haircut_sr_annual > 0.0:
        reasons["haircut"] = "pass"
    else:
        reasons["haircut"] = "fail: haircut Sharpe ratio is not positive"
    return reasons


def run_snooping(
    rel_perf: NDArray[np.float64], config: VerdictConfig, rng: RNGLike
) -> tuple[SPAResult, StepMResult]:
    """Run SPA and the stepdown on the same resamples.

    Parameters
    ----------
    rel_perf : numpy.ndarray
        ``T x N`` performance relative to the benchmark.
    config : VerdictConfig
        Number of resamples and significance level.
    rng : numpy.random.Generator or int or None
        Source of randomness.

    Returns
    -------
    spa : SPAResult
        SPA test.
    stepm : StepMResult
        Stepdown.
    """
    block = float(
        np.median([optimal_block_length(rel_perf[:, j]) for j in range(rel_perf.shape[1])])
    )
    indices = stationary_bootstrap_indices(
        rel_perf.shape[0], config.n_boot, mean_block=block, rng=rng
    )
    spa = spa_test(rel_perf, indices=indices, block_length=block)
    stepm = romano_wolf_stepdown(
        rel_perf, alpha=config.snoop_alpha, indices=indices, block_length=block
    )
    return spa, stepm


def evaluate(
    returns: ArrayLike,
    *,
    selected: int | None = None,
    n_trials: float | None = None,
    benchmark: ArrayLike | None = None,
    config: VerdictConfig = VerdictConfig(),
    rng: RNGLike = 0,
) -> VerdictReport:
    """Judge the selected trial of a ``T x N`` return matrix.

    PASS needs DSR at or above ``dsr_min``, PBO at or below ``pbo_max``, a consistent SPA
    p-value at or below ``snoop_alpha``, the stepdown rejecting the selected trial and a
    positive haircut Sharpe ratio. FAIL means a computed criterion failed. INCONCLUSIVE means
    a criterion could not be computed and none failed.

    Parameters
    ----------
    returns : array_like
        ``T x N`` per-period returns.
    selected : int, optional
        Column of the selected trial. Defaults to the highest Sharpe ratio.
    n_trials : float, optional
        Trials for the DSR and the haircut. Defaults to ``N``.
    benchmark : array_like, optional
        ``T`` benchmark returns. Defaults to zero.
    config : VerdictConfig, optional
        Thresholds.
    rng : numpy.random.Generator or int or None, default 0
        Source of randomness.

    Returns
    -------
    VerdictReport
        Verdict, reasons and evidence.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> r = np.random.default_rng(0).normal(0.0, 0.01, (400, 20))
    >>> evaluate(r, config=VerdictConfig(n_boot=100)).verdict
    'FAIL'
    """
    matrix = as_matrix(returns)
    n_obs, n_strategies = matrix.shape
    dsr = dsr_from_returns(matrix, selected=selected, n_trials=n_trials)
    srs = np.atleast_1d(sharpe_ratio(matrix))
    index = int(np.argmax(srs)) if selected is None else selected

    pbo: PBOResult | None = None
    if n_strategies >= 2 and n_obs >= config.pbo_partitions * MIN_OBS_PER_BLOCK:
        pbo = pbo_cscv(matrix, n_partitions=config.pbo_partitions)

    rel_perf = matrix
    if benchmark is not None:
        rel_perf = matrix - as_vector(benchmark, name="benchmark")[:, np.newaxis]
    spa, stepm = run_snooping(rel_perf, config, rng)

    tests = n_strategies if n_trials is None else max(1, round(n_trials))
    haircut = haircut_sharpe(
        annualize_sharpe(dsr.sr, config.periods_per_year),
        n_obs,
        periods_per_year=config.periods_per_year,
        n_tests=tests,
        method=config.haircut_method,
        rng=rng,
    )
    reasons = judge(
        config,
        dsr=dsr,
        pbo=pbo,
        spa=spa,
        stepm=stepm,
        haircut_sr_annual=haircut.haircut_sr_annual,
        selected=index,
    )
    return VerdictReport(decide(reasons), reasons, dsr, pbo, spa, stepm, haircut, index)
