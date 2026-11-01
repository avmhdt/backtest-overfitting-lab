"""MANIFEST.json for every ingest (DQ-09)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    raw_dir: Path,
    *,
    source: str,
    files: Sequence[Path],
    row_counts: Mapping[str, int],
    units: str,
    origin: str,
    release: str | None = None,
    notes: Mapping[str, Any] | None = None,
) -> Path:
    """Merge the given files into ``raw_dir/MANIFEST.json`` and return its path."""
    path = raw_dir / "MANIFEST.json"
    manifest: dict[str, Any] = {"source": source, "files": {}}
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    downloaded_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    for file in files:
        manifest["files"][file.name] = {
            "origin": origin,
            "downloaded_at_utc": downloaded_at,
            "sha256": sha256_file(file),
            "bytes": file.stat().st_size,
        }
    manifest["row_counts"] = {**manifest.get("row_counts", {}), **row_counts}
    manifest["units"] = units
    manifest["release"] = release
    if notes:
        manifest["notes"] = {**manifest.get("notes", {}), **notes}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
