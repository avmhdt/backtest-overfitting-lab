from overfitlab_study import bench


def test_hardware_fields():
    info = bench.hardware()
    assert info["cores"] >= 1
    assert info["ram_gb"] > 0
    assert set(info) == {"cpu", "cores", "ram_gb", "os", "python", "numba"}


def test_median_time_runs_the_action():
    calls = []
    assert bench.median_time(lambda: calls.append(1), repeats=3) >= 0.0
    assert len(calls) == 4


def test_row_flags_a_miss():
    assert bench.row("NFR", "case", 2.0, 1.0)["met"] is False


def test_run_writes_the_table(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "jit_compile_seconds", lambda: 1.0)
    monkeypatch.setattr(bench, "median_time", lambda action: 0.5)
    rows = bench.run(tmp_path)
    assert [row["requirement"] for row in rows] == [
        "NFR-01",
        "NFR-02",
        "NFR-02",
        "NFR-02",
        "NFR-03",
        "NFR-03",
    ]
    assert all(row["met"] for row in rows)
    assert (tmp_path / "tables" / "benchmarks.csv").exists()
