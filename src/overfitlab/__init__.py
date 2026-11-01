"""overfitlab: statistics that separate a real trading signal from a lucky one."""

from overfitlab._normal import EULER_GAMMA
from overfitlab.bootstrap import optimal_block_length, stationary_bootstrap_indices
from overfitlab.cv import CombinatorialPurgedCV, PurgedKFold, assemble_paths
from overfitlab.deflated import (
    deflated_sharpe_ratio,
    dsr_from_returns,
    expected_max_sharpe,
    min_backtest_length,
)
from overfitlab.factory import StrategySpec, run_strategies, strategy_grid
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
from overfitlab.synth import (
    SynthResult,
    block_bootstrap_returns,
    gaussian_noise,
    gbm_prices,
    planted_signal_matrix,
    planted_signal_prices,
    student_t_noise,
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
    "CombinatorialPurgedCV",
    "DSRResult",
    "HaircutResult",
    "PurgedKFold",
    "ReturnMoments",
    "StrategySpec",
    "SynthResult",
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
    "assemble_paths",
    "block_bootstrap_returns",
    "canonical_params_json",
    "deflated_sharpe_ratio",
    "dsr_from_returns",
    "evaluate_trial_log",
    "expected_max_sharpe",
    "gaussian_noise",
    "gbm_prices",
    "haircut_sharpe",
    "min_backtest_length",
    "min_track_record_length",
    "optimal_block_length",
    "planted_signal_matrix",
    "planted_signal_prices",
    "probabilistic_sharpe_ratio",
    "psr_from_returns",
    "read_trial_log",
    "return_moments",
    "run_strategies",
    "sharpe_ratio",
    "sharpe_std",
    "sharpe_to_tstat",
    "stationary_bootstrap_indices",
    "strategy_grid",
    "student_t_noise",
    "validate_trial_log",
]
