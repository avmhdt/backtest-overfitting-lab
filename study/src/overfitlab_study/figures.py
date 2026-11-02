"""Theory figures: each answers one interview question with generated numbers (FR-37)."""

from __future__ import annotations

import math
import shutil
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

import overfitlab as ol
from overfitlab_study.config import experiment_seeds
from overfitlab_study.reports import GRID, INK, MUTED, SERIES_1, SERIES_2, SURFACE

F64 = NDArray[np.float64]
PERIODS_PER_YEAR = 252
N_OBS = 1260
PBO_SEEDS = 20
SERIES_3 = "#1baf7a"


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save(fig: Figure, path: Path) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def dsr_vs_variance(path: Path) -> None:
    """The DSR of a fixed backtest falls with the spread of the trials, at any trial count."""
    sr = 1.5 / math.sqrt(PERIODS_PER_YEAR)
    spreads = np.linspace(0.01, 1.2, 120)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    _style(ax)
    for n_trials, colour in zip((10, 100, 1000), (SERIES_1, SERIES_2, SERIES_3), strict=True):
        values = [
            ol.deflated_sharpe_ratio(
                sr, N_OBS, n_trials=n_trials, var_trials=s * s / PERIODS_PER_YEAR
            ).dsr
            for s in spreads
        ]
        ax.plot(spreads, values, color=colour, linewidth=2, label=f"N = {n_trials:,}")
    ax.axhline(0.95, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xlabel("Standard deviation of annual Sharpe ratios across the trials", color=MUTED)
    ax.set_ylabel("DSR of an annual Sharpe of 1.5, 5 years", color=MUTED)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, labelcolor=MUTED, fontsize=9)
    _save(fig, path)


def _noise(rng: np.random.Generator) -> F64:
    return ol.gaussian_noise(N_OBS, 50, rng=rng)


def _equally_good(rng: np.random.Generator) -> F64:
    return ol.planted_signal_matrix(N_OBS, 50, true_sharpe_annual=1.0, n_true=50, rng=rng).returns


def _correlated(rng: np.random.Generator) -> F64:
    return ol.planted_signal_matrix(
        N_OBS, 50, true_sharpe_annual=1.5, n_true=1, corr=0.95, rng=rng
    ).returns


def _regime_break(rng: np.random.Generator) -> F64:
    returns = ol.planted_signal_matrix(N_OBS, 50, true_sharpe_annual=0.0, rng=rng).returns
    returns[: N_OBS // 2, 0] += 0.003
    return returns


def _few_observations(rng: np.random.Generator) -> F64:
    return ol.planted_signal_matrix(160, 50, true_sharpe_annual=2.0, rng=rng).returns


def _mixed_risk(rng: np.random.Generator) -> F64:
    returns = ol.planted_signal_matrix(N_OBS, 50, true_sharpe_annual=1.5, rng=rng).returns
    return returns * np.linspace(0.2, 5.0, 50)


def _strong(rng: np.random.Generator) -> F64:
    return ol.planted_signal_matrix(N_OBS, 50, true_sharpe_annual=3.0, rng=rng).returns


SCENARIOS: tuple[tuple[str, Callable[[np.random.Generator], F64]], ...] = (
    ("Pure noise", _noise),
    ("One strong strategy", _strong),
    ("All equally good", _equally_good),
    ("Near-duplicates, one good", _correlated),
    ("Edge only in the first half", _regime_break),
    ("Good strategy, 8 bars per block", _few_observations),
    ("Good strategy, mixed volatility", _mixed_risk),
)


def pbo_scenarios(master_seed: int, seeds: int = PBO_SEEDS) -> list[tuple[str, F64]]:
    """PBO per seed in each scenario."""
    out: list[tuple[str, F64]] = []
    for name, generate in SCENARIOS:
        values = [
            ol.pbo_cscv(generate(np.random.default_rng(seed))).pbo
            for seed in experiment_seeds(master_seed, f"figures-{name}", seeds)
        ]
        out.append((name, np.array(values)))
    return out


def pbo_failure_modes(path: Path, master_seed: int, seeds: int = PBO_SEEDS) -> None:
    """Where PBO misleads: its value depends on the set, not only on the winner's edge."""
    rows = pbo_scenarios(master_seed, seeds)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    _style(ax)
    for position, (_, values) in enumerate(reversed(rows)):
        low, mid, high = np.quantile(values, [0.25, 0.5, 0.75])
        ax.plot([low, high], [position, position], color=SERIES_1, linewidth=3, alpha=0.45)
        ax.scatter([mid], [position], color=SERIES_1, s=36, zorder=3)
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_yticks(range(len(rows)), [name for name, _ in reversed(rows)], color=INK)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel(f"PBO: median and interquartile range over {seeds} seeds", color=MUTED)
    _save(fig, path)


def trial_counting(path: Path) -> None:
    """The bar a selected Sharpe ratio must clear, against how many trials are admitted to."""
    variance = 1.0 / (N_OBS - 1)
    counts = np.unique(np.geomspace(1, 10_000, 120).astype(int))
    bars = [
        ol.annualize_sharpe(ol.expected_max_sharpe(float(n), variance), PERIODS_PER_YEAR)
        for n in counts
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), facecolor=SURFACE)
    _style(ax)
    ax.plot(counts, bars, color=SERIES_1, linewidth=2)
    marks = (
        (1, "Reported as a single test"),
        (20, "20 hand edits"),
        (1034, "The whole factory grid"),
    )
    for count, label in marks:
        value = ol.annualize_sharpe(ol.expected_max_sharpe(float(count), variance), 252)
        ax.scatter([count], [value], color=INK, s=28, zorder=3)
        ax.annotate(
            f"{label}: {value:.2f}",
            (count, value),
            xytext=(8, -4),
            textcoords="offset points",
            color=INK,
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.12, None)
    ax.set_xlabel("Trials admitted to, on five years of noise", color=MUTED)
    ax.set_ylabel("Expected best annual Sharpe", color=MUTED)
    _save(fig, path)


def pbo_logit_hist(path: Path, master_seed: int) -> None:
    """Logits of the in-sample winner: centred on noise, far to the right on a real edge."""
    noise_seed, signal_seed = experiment_seeds(master_seed, "figures-logits", 2)
    noise = ol.pbo_cscv(ol.gaussian_noise(N_OBS, 100, rng=np.random.default_rng(noise_seed)))
    signal = ol.pbo_cscv(
        ol.planted_signal_matrix(
            N_OBS, 100, true_sharpe_annual=3.0, rng=np.random.default_rng(signal_seed)
        ).returns
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True, facecolor=SURFACE)
    bins = np.linspace(-5.0, 5.0, 41)
    panels = ((noise, "Noise", SERIES_1), (signal, "One strategy with a real edge", SERIES_2))
    for ax, (result, title, colour) in zip(axes, panels, strict=True):
        _style(ax)
        ax.hist(np.clip(result.logits, -5, 5), bins=bins, color=colour)
        ax.axvline(0.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(f"{title}, PBO = {result.pbo:.2f}", color=INK, fontsize=10)
        ax.set_xlabel("Logit of out-of-sample rank", color=MUTED)
    axes[0].set_ylabel("Combinations", color=MUTED)
    _save(fig, path)


def run(reports_dir: Path, *, master_seed: int, seeds: int = PBO_SEEDS) -> list[Path]:
    folder = reports_dir / "figures"
    outputs: dict[str, Callable[[Path], None]] = {
        "dsr_vs_variance.png": dsr_vs_variance,
        "pbo_failure_modes.png": lambda path: pbo_failure_modes(path, master_seed, seeds),
        "trial_counting.png": trial_counting,
        "pbo_logit_hist_noise_vs_signal.png": lambda path: pbo_logit_hist(path, master_seed),
    }
    for name, draw in outputs.items():
        draw(folder / name)
    return [folder / name for name in outputs]


def copy_to_docs(reports_dir: Path, docs_dir: Path) -> int:
    """Copy every report figure to the docs site."""
    target = docs_dir / "assets" / "figures"
    target.mkdir(parents=True, exist_ok=True)
    figures = sorted((reports_dir / "figures").glob("*.png"))
    for figure in figures:
        shutil.copy(figure, target / figure.name)
    return len(figures)
