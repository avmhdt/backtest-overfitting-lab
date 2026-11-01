"""Experiment configuration and seed derivation."""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

STUDY_CONFIG = Path("experiments/study.yaml")


def load_study_config(path: Path = STUDY_CONFIG) -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))


def experiment_seeds(master: int, experiment: str, count: int) -> list[np.random.SeedSequence]:
    """Independent seeds per replication, stable for a given master seed and experiment."""
    root = np.random.SeedSequence(master, spawn_key=(zlib.crc32(experiment.encode()),))
    return root.spawn(count)
