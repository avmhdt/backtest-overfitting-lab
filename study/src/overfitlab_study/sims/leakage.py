"""Purged against standard k-fold on overlapping labels, on pure-noise prices (FR-24)."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from overfitlab.cv import CombinatorialPurgedCV, PurgedKFold
from overfitlab_study.config import experiment_seeds
from overfitlab_study.reports import GRID, INK, MUTED, SERIES_1, SURFACE, write_table

F64 = NDArray[np.float64]
SCHEMES = ("kfold_unshuffled", "kfold_shuffled", "purged_kfold", "cpcv")
LABELS = {
    "kfold_unshuffled": "K-fold, unshuffled",
    "kfold_shuffled": "K-fold, shuffled",
    "purged_kfold": "Purged k-fold",
    "cpcv": "CPCV",
}
MAX_WORKERS = 8


@dataclass(frozen=True)
class LeakageConfig:
    n_obs: int
    label_horizon: int
    n_features: int
    feature_ar1: float
    n_splits: int
    embargo_pct: float
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> LeakageConfig:
        model = config["model"]
        return cls(
            n_obs=config["n_obs"],
            label_horizon=config["label_horizon"],
            n_features=config["n_features"],
            feature_ar1=config["feature_ar1"],
            n_splits=config["n_splits"],
            embargo_pct=config["embargo_pct"],
            n_estimators=model["n_estimators"],
            max_depth=model["max_depth"],
            min_samples_leaf=model["min_samples_leaf"],
        )


def make_dataset(
    config: LeakageConfig, rng: np.random.Generator
) -> tuple[F64, NDArray[np.int64], NDArray[np.int64]]:
    """Features with no relation to the labels, and overlapping forward-return labels."""
    h = config.label_horizon
    phi = config.feature_ar1
    returns = rng.standard_normal(config.n_obs + h)
    cumulative = np.concatenate(([0.0], np.cumsum(returns)))
    forward = cumulative[h : config.n_obs + h] - cumulative[: config.n_obs]
    labels = (forward > 0.0).astype(np.int64)
    shocks = rng.standard_normal((config.n_obs, config.n_features)) * math.sqrt(1.0 - phi * phi)
    features = np.zeros_like(shocks)
    features[0] = rng.standard_normal(config.n_features)
    for t in range(1, config.n_obs):
        features[t] = phi * features[t - 1] + shocks[t]
    t1 = np.arange(config.n_obs, dtype=np.int64) + h
    return features, labels, t1


def _accuracy(
    config: LeakageConfig,
    features: F64,
    labels: NDArray[np.int64],
    splits: Sequence[tuple[NDArray[np.int64], NDArray[np.int64]]],
    random_state: int,
) -> float:
    correct = 0
    total = 0
    for train, test in splits:
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            random_state=random_state,
            n_jobs=1,
        )
        model.fit(features[train], labels[train])
        correct += int((model.predict(features[test]) == labels[test]).sum())
        total += test.size
    return correct / total


def run_seed(config: LeakageConfig, seed: np.random.SeedSequence) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    features, labels, t1 = make_dataset(config, rng)
    state = int(seed.generate_state(1)[0])
    schemes = {
        "kfold_unshuffled": KFold(config.n_splits, shuffle=False),
        "kfold_shuffled": KFold(config.n_splits, shuffle=True, random_state=state),
        "purged_kfold": PurgedKFold(config.n_splits, t1=t1, embargo_pct=config.embargo_pct),
        "cpcv": CombinatorialPurgedCV(6, 2, t1=t1, embargo_pct=config.embargo_pct),
    }
    return {
        name: _accuracy(config, features, labels, list(scheme.split(features)), state)
        for name, scheme in schemes.items()
    }


def _mean_ci(values: F64) -> tuple[float, float, float]:
    mean = float(values.mean())
    half = float(stats.t.ppf(0.975, values.size - 1) * values.std(ddof=1) / math.sqrt(values.size))
    return mean, mean - half, mean + half


def summarise(accuracy: Mapping[str, F64]) -> list[dict[str, Any]]:
    """One row per scheme, then the paired gaps to purged k-fold."""
    rows: list[dict[str, Any]] = []
    for scheme in SCHEMES:
        mean, low, high = _mean_ci(accuracy[scheme])
        rows.append(
            {
                "row": scheme,
                "kind": "accuracy",
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "sd": float(accuracy[scheme].std(ddof=1)),
                "n_seeds": int(accuracy[scheme].size),
            }
        )
    for scheme in ("kfold_shuffled", "kfold_unshuffled"):
        gap = accuracy[scheme] - accuracy["purged_kfold"]
        mean, low, high = _mean_ci(gap)
        rows.append(
            {
                "row": f"{scheme}_minus_purged",
                "kind": "paired_gap",
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "sd": float(gap.std(ddof=1)),
                "n_seeds": int(gap.size),
            }
        )
    return rows


def check_criteria(rows: Sequence[Mapping[str, Any]], min_gap: float = 0.05) -> dict[str, bool]:
    """AC-W-04: the shuffled gap and the purged accuracy interval."""
    by_row = {row["row"]: row for row in rows}
    gap = by_row["kfold_shuffled_minus_purged"]
    purged = by_row["purged_kfold"]
    return {
        "gap_at_least_min": bool(gap["mean"] >= min_gap),
        "gap_ci_excludes_zero": bool(gap["ci_low"] > 0.0),
        "purged_ci_contains_half": bool(purged["ci_low"] <= 0.5 <= purged["ci_high"]),
    }


def plot(accuracy: Mapping[str, F64], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    jitter = np.random.default_rng(0)
    for position, scheme in enumerate(reversed(SCHEMES)):
        values = accuracy[scheme]
        offsets = jitter.uniform(-0.16, 0.16, values.size)
        ax.scatter(values, position + offsets, s=14, color=SERIES_1, alpha=0.45, linewidths=0)
        mean = float(values.mean())
        ax.plot([mean, mean], [position - 0.3, position + 0.3], color=INK, linewidth=2)
        ax.annotate(
            f"{mean:.3f}",
            (mean, position + 0.3),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            color=INK,
            fontsize=9,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0},
            zorder=3,
        )
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_yticks(range(len(SCHEMES)), [LABELS[s] for s in reversed(SCHEMES)], color=INK)
    ax.set_ylim(-0.6, len(SCHEMES) - 0.3)
    ax.set_xlabel("Out-of-sample accuracy on pure noise (one dot per seed)", color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def run(
    config: LeakageConfig,
    *,
    seeds: int,
    master_seed: int,
    reports_dir: Path,
    results_dir: Path,
    config_key: str,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    sequences = experiment_seeds(master_seed, "leakage", seeds)
    workers = min(MAX_WORKERS, os.cpu_count() or 1, seeds)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(run_seed, [config] * seeds, sequences))
    accuracy = {s: np.array([o[s] for o in outcomes]) for s in SCHEMES}

    rows = summarise(accuracy)
    write_table(rows, reports_dir / "tables" / "leakage.csv")
    plot(accuracy, reports_dir / "figures" / "leakage_kfold_vs_purged.png")

    raw = pl.DataFrame(
        {
            "experiment": ["leakage"] * (seeds * len(SCHEMES)),
            "seed": [i for i in range(seeds) for _ in SCHEMES],
            "test": [s for _ in range(seeds) for s in SCHEMES],
            "statistic": [float(accuracy[s][i]) for i in range(seeds) for s in SCHEMES],
        }
    )
    target = results_dir / "sims" / "leakage" / f"{config_key}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    raw.write_parquet(target, compression="zstd")
    return rows, check_criteria(rows)
