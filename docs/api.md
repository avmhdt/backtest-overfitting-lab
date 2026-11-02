# API reference

Every public name is re-exported from `overfitlab`. Sharpe ratios are per period unless a name ends in `_annual`, and kurtosis is non-excess (normal = 3).

## Sharpe statistics

::: overfitlab.sharpe
    options:
      members:
        - sharpe_ratio
        - return_moments
        - annualize_sharpe
        - sharpe_std
        - probabilistic_sharpe_ratio
        - psr_from_returns
        - min_track_record_length

## Deflated Sharpe Ratio

::: overfitlab.deflated
    options:
      members:
        - expected_max_sharpe
        - deflated_sharpe_ratio
        - dsr_from_returns
        - min_backtest_length

## Haircut

::: overfitlab.haircut
    options:
      members:
        - sharpe_to_tstat
        - haircut_sharpe

## Cross-validation

::: overfitlab.cv
    options:
      members:
        - PurgedKFold
        - CombinatorialPurgedCV
        - assemble_paths

## PBO

::: overfitlab.pbo
    options:
      members:
        - PBOResult
        - pbo_cscv

## Bootstrap

::: overfitlab.bootstrap
    options:
      members:
        - optimal_block_length
        - stationary_bootstrap_indices

## Data-snooping tests

::: overfitlab.snooping
    options:
      members:
        - SnoopResult
        - SPAResult
        - StepMResult
        - reality_check
        - spa_test
        - romano_wolf_stepdown

## Synthetic data

::: overfitlab.synth
    options:
      members:
        - SynthResult
        - gaussian_noise
        - student_t_noise
        - gbm_prices
        - planted_signal_matrix
        - planted_signal_prices
        - block_bootstrap_returns

## Strategy factory

::: overfitlab.factory
    options:
      members:
        - StrategySpec
        - strategy_grid
        - run_strategies

## Verdict gate

::: overfitlab.verdict
    options:
      members:
        - VerdictConfig
        - VerdictReport
        - evaluate

## Trial log

::: overfitlab.triallog
    options:
      members:
        - TRIAL_LOG_COLUMNS
        - TRIAL_LOG_VERSION
        - TrialLogger
        - TrialLog
        - TrialLogError
        - TrialLogIssue
        - TrialLogReport
        - TrialSharpes
        - add_manual_trial
        - canonical_params_json
        - evaluate_trial_log
        - read_trial_log
        - validate_trial_log

## Results

::: overfitlab.results
    options:
      members:
        - ReturnMoments
        - DSRResult
        - HaircutResult

## Constants

`overfitlab.EULER_GAMMA` is the Euler–Mascheroni constant, used by the expected maximum. `overfitlab.__version__` is the installed version.
