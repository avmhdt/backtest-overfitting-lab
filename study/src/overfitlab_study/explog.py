"""Append-only experiment log and JSON-lines run logs.

Every study command runs inside :func:`study_run`, which appends one row to
``results/experiment_log.csv`` even when the command fails. That file is the honest
count of what was run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import subprocess
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from overfitlab_study.settings import Settings

EXPERIMENT_LOG_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp",
    "git_sha",
    "command",
    "config_hash",
    "params_json",
    "metrics_json",
    "duration_s",
)


def canonical_json(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()[:16]


def git_sha(repo_dir: Path = Path(".")) -> str:
    """Return HEAD, suffixed ``-dirty`` if tracked files have uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if status else sha


def append_experiment_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EXPERIMENT_LOG_COLUMNS, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


class JsonLinesFormatter(logging.Formatter):
    def __init__(self, run_id: str, command: str, seed: int) -> None:
        super().__init__()
        self.context = {"run_id": run_id, "command": command, "seed": seed}

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                **self.context,
                "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )


@dataclass
class StudyRun:
    run_id: str
    command: str
    settings: Settings
    metrics: dict[str, Any] = field(default_factory=dict)


@contextmanager
def study_run(
    command: str,
    settings: Settings,
    *,
    params: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> Iterator[StudyRun]:
    """Time a study command, log it to JSON lines and append its experiment-log row."""
    params = dict(params or {})
    run = StudyRun(run_id=str(uuid.uuid4()), command=command, settings=settings)
    log_path = settings.olab_results_dir / "logs" / f"{run.run_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(JsonLinesFormatter(run.run_id, command, settings.olab_seed))
    logger = logging.getLogger("overfitlab_study")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    started = time.perf_counter()
    logger.info("run started")
    status = "ok"
    try:
        yield run
    except BaseException:
        status = "failed"
        logger.exception("run failed")
        raise
    finally:
        duration = time.perf_counter() - started
        logger.info("run finished in %.3f s", duration)
        logger.removeHandler(handler)
        handler.close()
        append_experiment_row(
            settings.olab_results_dir / "experiment_log.csv",
            {
                "run_id": run.run_id,
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "git_sha": git_sha(),
                "command": command,
                "config_hash": config_hash(config if config is not None else params),
                "params_json": canonical_json(params),
                "metrics_json": canonical_json({"status": status, **run.metrics}),
                "duration_s": f"{duration:.3f}",
            },
        )
