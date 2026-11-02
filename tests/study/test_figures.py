import numpy as np
import pytest
from typer.testing import CliRunner

import overfitlab as ol
from overfitlab_study import figures
from overfitlab_study.cli import app


def test_each_figure_is_drawn(tmp_path):
    paths = figures.run(tmp_path, master_seed=1, seeds=1)
    assert sorted(path.name for path in paths) == [
        "dsr_vs_variance.png",
        "pbo_failure_modes.png",
        "pbo_logit_hist_noise_vs_signal.png",
        "trial_counting.png",
    ]
    assert all(path.stat().st_size > 1000 for path in paths)


def test_copy_to_docs(tmp_path):
    source = tmp_path / "reports" / "figures"
    source.mkdir(parents=True)
    (source / "a.png").write_bytes(b"x")
    assert figures.copy_to_docs(tmp_path / "reports", tmp_path / "docs") == 1
    assert (tmp_path / "docs/assets/figures/a.png").read_bytes() == b"x"


def test_scenarios_are_reproducible():
    first = figures.pbo_scenarios(1, seeds=1)
    second = figures.pbo_scenarios(1, seeds=1)
    assert [name for name, _ in first] == [name for name, _ in second]
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(first, second, strict=True))


@pytest.mark.parametrize(
    ("generate", "shape"),
    [(figures._regime_break, (1260, 50)), (figures._few_observations, (160, 50))],
)
def test_scenario_shapes(generate, shape):
    assert generate(np.random.default_rng(0)).shape == shape


def test_regime_break_has_its_edge_only_in_the_first_half():
    returns = figures._regime_break(np.random.default_rng(0))
    first = ol.sharpe_ratio(returns[:630, 0])
    second = ol.sharpe_ratio(returns[630:, 0])
    assert first > 0.2
    assert abs(second) < 0.1


def test_figures_command(tmp_path, monkeypatch):
    monkeypatch.setenv("OLAB_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("OLAB_REPORTS_DIR", str(tmp_path / "reports"))
    drawn = []
    monkeypatch.setattr(figures, "run", lambda *a, **k: drawn.append(k) or [tmp_path / "a.png"])
    monkeypatch.setattr(figures, "copy_to_docs", lambda *a: 3)
    result = CliRunner().invoke(app, ["figures", "--docs", "--seeds", "2"])
    assert result.exit_code == 0, result.output
    assert drawn[0]["seeds"] == 2
