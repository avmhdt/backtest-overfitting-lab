import subprocess

from overfitlab_study.explog import git_sha


def git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def make_repo(path):
    (path / "results").mkdir()
    (path / "results" / "experiment_log.csv").write_text("a\n", encoding="utf-8")
    (path / "code.py").write_text("x = 1\n", encoding="utf-8")
    git(path, "init")
    git(path, "add", ".")
    git(path, "commit", "-m", "start")


def test_study_outputs_do_not_dirty_the_tree(tmp_path):
    make_repo(tmp_path)
    (tmp_path / "results" / "experiment_log.csv").write_text("a\nb\n", encoding="utf-8")
    assert not git_sha(tmp_path).endswith("-dirty")


def test_code_changes_dirty_the_tree(tmp_path):
    make_repo(tmp_path)
    (tmp_path / "code.py").write_text("x = 2\n", encoding="utf-8")
    assert git_sha(tmp_path).endswith("-dirty")
