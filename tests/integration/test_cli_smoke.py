from importlib.metadata import entry_points

import pytest
from typer.testing import CliRunner

import overfitlab
from overfitlab.cli import app
from overfitlab_study.cli import app as study_app

runner = CliRunner()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == overfitlab.__version__


@pytest.mark.parametrize("args", [["--help"], ["trials", "--help"]])
def test_library_help(args):
    assert runner.invoke(app, args).exit_code == 0


@pytest.mark.parametrize("args", [["--help"], ["data", "--help"], ["data", "smoke", "--help"]])
def test_study_help(args):
    assert runner.invoke(study_app, args).exit_code == 0


@pytest.mark.slow
def test_study_entry_point_resolves():
    names = {ep.name for ep in entry_points(group="console_scripts")}
    assert {"overfitlab", "olab-study"} <= names
