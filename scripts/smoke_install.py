"""Check an installed overfitlab: the README quickstart, the CLI and a light import.

Takes the expected version and the README path. Uses only the standard library and the
installed package, so it runs from a fresh venv.
"""

from __future__ import annotations

import contextlib
import io
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

HEAVY = {"pyarrow", "typer", "polars", "arch", "sklearn", "matplotlib"}


def run_quickstart(readme: Path) -> str:
    """Run the Python block of the README quickstart and return what it prints."""
    text = readme.read_text(encoding="utf-8")
    section = text.split("## Quickstart", 1)[1].split("\n## ", 1)[0]
    code = re.findall(r"```python\n(.*?)```", section, flags=re.DOTALL)
    if len(code) != 1:
        raise SystemExit(f"expected one Python block in the quickstart, found {len(code)}")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code[0], {})  # noqa: S102
    return out.getvalue().strip()


def main() -> None:
    """Run every check, stopping at the first failure."""
    version, readme = sys.argv[1], Path(sys.argv[2])
    installed = metadata.version("overfitlab")
    if installed != version:
        raise SystemExit(f"installed {installed}, expected {version}")

    printed = run_quickstart(readme)
    if printed != "0.9004":
        raise SystemExit(f"the quickstart printed {printed!r}")

    command = [sys.executable, "-c", "from overfitlab.cli import app; app()", "--version"]
    cli = subprocess.run(command, capture_output=True, text=True, check=True)
    if cli.stdout.strip() != version:
        raise SystemExit(f"the CLI reported {cli.stdout.strip()!r}")

    probe = "import sys, overfitlab; print(*sorted({m.split('.')[0] for m in sys.modules}))"
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    found = HEAVY & set(loaded.stdout.split())
    if found:
        raise SystemExit(f"import overfitlab loaded {sorted(found)}")
    print(f"overfitlab {version} works")


if __name__ == "__main__":
    main()
