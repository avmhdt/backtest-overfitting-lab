"""Command-line interface ``overfitlab``."""

from __future__ import annotations

from typing import Annotated

import typer

from overfitlab import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
trials_app = typer.Typer(no_args_is_help=True, help="Trial-log commands (contract v1.0).")
app.add_typer(trials_app, name="trials")


def _print_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_print_version, is_eager=True, help="Print the version."
        ),
    ] = False,
) -> None:
    """Tell a real trading signal from a lucky one."""
