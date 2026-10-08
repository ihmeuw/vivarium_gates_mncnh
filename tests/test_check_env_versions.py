"""Tests for tools/check_env_versions.py, mostly through its command line."""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

from vivarium_gates_mncnh.tools import check_env_versions
from vivarium_gates_mncnh.tools.check_env_versions import parse_requirements

from .conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "src" / "vivarium_gates_mncnh" / "tools" / "check_env_versions.py"

# A line for a test overrides.txt. Nothing is ever installed from it, so the commit is made up.
OVERRIDES = (
    "vivarium-public-health @ "
    "git+https://github.com/ihmeuw/vivarium-suite@abc1234#subdirectory=libs/public-health\n"
)

SIM_V1 = "numpy==1.26.4\npandas==2.0.0\n"
SIM_V2 = "numpy==1.26.4\npandas==2.1.0\n"
SIM_V3 = "numpy==1.26.4\npandas==2.2.0\n"

GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "Test Author",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "Test Author",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        env={**os.environ, **GIT_IDENTITY},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _write(repo: Path, name: str, text: str) -> None:
    (repo / "requirements").mkdir(parents=True, exist_ok=True)
    (repo / "requirements" / name).write_text(text)


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _make_repo(path: Path, simulation: str = SIM_V1) -> Path:
    """A git repo on ``main`` with a committed simulation version file."""
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "symbolic-ref", "HEAD", "refs/heads/main")  # git 2.25 has no `init -b`
    _write(path, "simulation.txt", simulation)
    _commit(path, "Initial version files")
    return path


def _run(*args: str, cwd: Path, python: str = sys.executable) -> subprocess.CompletedProcess:
    """Run check_env_versions.py by file path in an isolated interpreter, as environment.sh does."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    return subprocess.run(
        [python, "-I", str(SCRIPT), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _out(result: subprocess.CompletedProcess) -> str:
    return result.stdout + result.stderr


def _record(repo: Path, dest: Path) -> None:
    result = _run(
        "record", "--repo", ".", "--type", "simulation", "--dest", str(dest), cwd=repo
    )
    assert result.returncode == 0, _out(result)


def _venv(path: Path, installed: Dict[str, str]) -> str:
    """A bare venv whose site-packages holds fake distributions; returns its python."""
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(path)], check=True)
    python = str(path / "bin" / "python")
    site = subprocess.run(
        [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    for name, version in installed.items():
        dist_info = Path(site) / "{}-{}.dist-info".format(name.replace("-", "_"), version)
        dist_info.mkdir(parents=True)
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: {}\nVersion: {}\n".format(name, version)
        )
    return python


class TestCompare:
    """``compare`` decides whether environment.sh rebuilds a conda environment."""

    @staticmethod
    def _matches(repo: Path, record: Path) -> None:
        _record(repo, record)

    @staticmethod
    def _version_changed(repo: Path, record: Path) -> None:
        _record(repo, record)
        _write(repo, "simulation.txt", SIM_V2)
        _commit(repo, "Bump pandas")

    @staticmethod
    def _overrides_added(repo: Path, record: Path) -> None:
        _record(repo, record)
        _write(repo, "overrides.txt", OVERRIDES)

    @staticmethod
    def _no_record(repo: Path, record: Path) -> None:
        record.mkdir()

    @staticmethod
    def _malformed_checkout(repo: Path, record: Path) -> None:
        _record(repo, record)
        _write(repo, "simulation.txt", "pandas>=2.0\n")

    @pytest.mark.parametrize(
        "setup, env_type, expected_code, expected_output",
        [
            ("_matches", "simulation", check_env_versions.EXIT_MATCH, []),
            (
                "_version_changed",
                "simulation",
                check_env_versions.EXIT_DIFFERENT,
                ["pandas: 2.0.0 -> 2.1.0", "Bump pandas"],
            ),
            (
                "_overrides_added",
                "simulation",
                check_env_versions.EXIT_DIFFERENT,
                ["override"],
            ),
            ("_no_record", "simulation", check_env_versions.EXIT_NO_RECORD, []),
            (
                "_malformed_checkout",
                "simulation",
                check_env_versions.EXIT_ERROR,
                ["pandas>=2.0"],
            ),
            ("_matches", "bogus", check_env_versions.EXIT_ERROR, ["bogus"]),
        ],
    )
    def test_exit_code_and_report(
        self,
        tmp_path: Path,
        setup: str,
        env_type: str,
        expected_code: int,
        expected_output: List[str],
    ) -> None:
        """Each situation gets its own exit code, and a change names the package and commit."""
        repo = _make_repo(tmp_path / "repo")
        record = tmp_path / "record"
        getattr(self, setup)(repo, record)

        result = _run(
            "compare",
            "--repo",
            ".",
            "--type",
            env_type,
            "--record-dir",
            str(record),
            cwd=repo,
        )

        assert result.returncode == expected_code, _out(result)
        for text in expected_output:
            assert text in _out(result)
        if expected_code == check_env_versions.EXIT_MATCH:
            assert _out(result) == ""


class TestExplainSharedMismatch:
    """``explain-shared-mismatch`` explains, but never blocks, a mismatch with the shared env."""

    @staticmethod
    def _branch_behind_main(repo: Path, record: Path) -> None:
        base = _git(repo, "rev-parse", "HEAD")
        _write(repo, "simulation.txt", SIM_V2)
        main = _commit(repo, "Main bumps pandas")
        _git(repo, "update-ref", "refs/remotes/origin/main", main)
        _git(repo, "checkout", "-q", "-b", "feature", base)
        record.mkdir()  # the shared env was built from main
        (record / "simulation.txt").write_text(SIM_V2)

    @staticmethod
    def _branch_changed_versions(repo: Path, record: Path) -> None:
        _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
        _record(repo, record)
        _git(repo, "checkout", "-q", "-b", "feature")
        _write(repo, "simulation.txt", SIM_V3)
        _commit(repo, "Feature bumps pandas")

    @staticmethod
    def _shared_env_behind_main(repo: Path, record: Path) -> None:
        _record(repo, record)
        _write(repo, "simulation.txt", SIM_V2)
        _git(repo, "update-ref", "refs/remotes/origin/main", _commit(repo, "Bump pandas"))

    @staticmethod
    def _no_record(repo: Path, record: Path) -> None:
        record.mkdir()

    @staticmethod
    def _malformed_checkout(repo: Path, record: Path) -> None:
        _record(repo, record)
        _write(repo, "simulation.txt", "pandas>=2.0\n")

    @pytest.mark.parametrize(
        "setup, advice",
        [
            ("_branch_behind_main", "merge"),
            ("_branch_changed_versions", "source environment.sh -t simulation"),
            ("_shared_env_behind_main", "catch up"),
            ("_no_record", "no record"),
            ("_malformed_checkout", "WARNING"),
        ],
    )
    def test_advice(self, tmp_path: Path, setup: str, advice: str) -> None:
        """Each kind of mismatch gets its own advice, and the command always exits 0."""
        repo = _make_repo(tmp_path / "repo")
        record = tmp_path / "record"
        getattr(self, setup)(repo, record)

        result = _run(
            "explain-shared-mismatch",
            "--repo",
            ".",
            "--type",
            "simulation",
            "--record-dir",
            str(record),
            cwd=repo,
        )

        assert result.returncode == 0, _out(result)
        assert advice in _out(result)


def test_installed_versions_that_differ_are_reported(tmp_path: Path) -> None:
    """``installed-matches-version-file`` flags installed pins at the wrong version, and nothing else."""
    repo = tmp_path / "repo"
    _write(
        repo,
        "simulation.txt",
        "attrs==23.1.0\nnumpy==1.26.4\npywin32==306\nvivarium-public-health==6.7.0\n",
    )
    _write(repo, "overrides.txt", OVERRIDES)
    python = _venv(
        tmp_path / "venv",
        {
            "attrs": "24.2.0",  # pinned, wrong version: reported
            "numpy": "1.26.4",  # matches
            "vivarium-public-health": "6.8.0.dev3",  # overridden: skipped
            "wheel": "0.47.0",  # not pinned: skipped (pywin32 is pinned, not installed)
        },
    )
    args = ("installed-matches-version-file", "--repo", str(repo), "--type", "simulation")

    result = _run(*args, cwd=repo, python=python)
    assert result.returncode == 1, _out(result)
    assert re.findall(r"^\s+(\S.*)$", result.stdout, re.M) == [
        "attrs: pinned 23.1.0, installed 24.2.0"
    ]

    _write(repo, "simulation.txt", "attrs==24.2.0\nnumpy==1.26.4\n")
    assert _run(*args, cwd=repo, python=python).returncode == 0


@pytest.mark.parametrize("env_type", check_env_versions.ENV_TYPES)
def test_version_files_pin_the_build_tools(env_type: str) -> None:
    """build-env installs uv and vivarium_build_utils from the version file first."""
    pins = parse_requirements((REPO_ROOT / "requirements" / f"{env_type}.txt").read_text())
    assert {"uv", "vivarium-build-utils"} <= set(pins)
