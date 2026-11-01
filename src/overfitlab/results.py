"""Result dataclasses returned by the public API."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReturnMoments:
    """Moments of a return series.

    Attributes
    ----------
    sr : float
        Per-period Sharpe ratio with ``ddof=1``.
    skew : float
        Sample skewness (gamma_3).
    kurtosis : float
        Non-excess kurtosis (gamma_4; normal = 3).
    n_obs : int
        Number of observations.
    """

    sr: float
    skew: float
    kurtosis: float
    n_obs: int


@dataclass(frozen=True)
class DSRResult:
    """Deflated Sharpe Ratio and its inputs. All Sharpe quantities are per period.

    Attributes
    ----------
    dsr : float
        Deflated Sharpe Ratio, a probability.
    sr : float
        Sharpe ratio of the selected trial.
    sr0 : float
        Rejection threshold, the expected maximum Sharpe under the null.
    z : float
        Standard normal score whose CDF is ``dsr``.
    n_trials : float
        Number of trials, possibly effective and non-integer.
    var_trials : float
        Variance of the trial Sharpe ratios.
    n_obs : int
        Number of observations of the selected trial.
    skew : float
        Skewness of the selected trial.
    kurtosis : float
        Non-excess kurtosis of the selected trial.
    """

    dsr: float
    sr: float
    sr0: float
    z: float
    n_trials: float
    var_trials: float
    n_obs: int
    skew: float
    kurtosis: float

    def to_dict(self) -> dict[str, float]:
        """Return the fields as a plain dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class HaircutResult:
    """Harvey-Liu haircut of an annual Sharpe ratio.

    Attributes
    ----------
    method : str
        ``"bonferroni"``, ``"holm"`` or ``"bhy"``.
    sr_annual : float
        Annual Sharpe ratio after the autocorrelation correction.
    haircut_sr_annual : float
        Annual Sharpe ratio implied by the adjusted p-value.
    haircut_pct : float
        Haircut as a percentage of ``sr_annual``.
    tstat : float
        t-statistic of ``sr_annual``.
    pvalue : float
        Two-sided single-test p-value.
    adj_pvalue : float
        Multiple-testing adjusted p-value.
    """

    method: str
    sr_annual: float
    haircut_sr_annual: float
    haircut_pct: float
    tstat: float
    pvalue: float
    adj_pvalue: float
