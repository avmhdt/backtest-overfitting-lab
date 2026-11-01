"""overfitlab: statistics that separate a real trading signal from a lucky one."""

from overfitlab._normal import EULER_GAMMA
from overfitlab.deflated import (
    deflated_sharpe_ratio,
    dsr_from_returns,
    expected_max_sharpe,
    min_backtest_length,
)
from overfitlab.haircut import haircut_sharpe, sharpe_to_tstat
from overfitlab.results import DSRResult, HaircutResult, ReturnMoments
from overfitlab.sharpe import (
    annualize_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    psr_from_returns,
    return_moments,
    sharpe_ratio,
    sharpe_std,
)
from overfitlab.triallog import (
    TRIAL_LOG_COLUMNS,
    TRIAL_LOG_VERSION,
    TrialLog,
    TrialLogError,
    TrialLogger,
    TrialLogIssue,
    TrialLogReport,
    TrialSharpes,
    add_manual_trial,
    canonical_params_json,
    evaluate_trial_log,
    read_trial_log,
    validate_trial_log,
)
from overfitlab.verdict import VerdictConfig

__version__ = "0.1.0"

__all__ = [
    "EULER_GAMMA",
    "DSRResult",
    "HaircutResult",
    "ReturnMoments",
    "TRIAL_LOG_COLUMNS",
    "TRIAL_LOG_VERSION",
    "TrialLog",
    "TrialLogError",
    "TrialLogIssue",
    "TrialLogReport",
    "TrialLogger",
    "TrialSharpes",
    "VerdictConfig",
    "__version__",
    "add_manual_trial",
    "annualize_sharpe",
    "canonical_params_json",
    "deflated_sharpe_ratio",
    "dsr_from_returns",
    "evaluate_trial_log",
    "expected_max_sharpe",
    "haircut_sharpe",
    "min_backtest_length",
    "min_track_record_length",
    "probabilistic_sharpe_ratio",
    "psr_from_returns",
    "read_trial_log",
    "return_moments",
    "sharpe_ratio",
    "sharpe_std",
    "sharpe_to_tstat",
    "validate_trial_log",
]
