"""Unit tests for ``vivarium_gates_mncnh.tools.utilities``.

These are the repository's first unit tests for ``tools/``. They deliberately
need no cluster, no conda and no network: every collaborator that would reach
outside the process is either monkeypatched or replaced by a scratch git
repository under ``tmp_path``.

They live in the simulation environment's half of the ``conftest.py``
collection split (``tests/`` root), so they are collected there and ignored in
the artifact environment.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

import vivarium_gates_mncnh
from vivarium_gates_mncnh.tools import utilities

#: Where the package under test actually lives, computed the same way the rest
#: of the repository computes it (``constants/paths.BASE_DIR``). Used to state
#: what ``repo_root`` is expected to find without reaching into its internals.
PACKAGE_DIR = Path(vivarium_gates_mncnh.__file__).resolve().parent

#: The name of the module global in ``tools.utilities`` that records the
#: installed package's location -- resolution step 1 of ``repo_root``. Named
#: once here because two tests need to defeat that step in order to reach
#: step 2, and the name is the only thing they need to know about it.
PACKAGE_DIR_ATTR = "PACKAGE_DIR"


def write_executable(path: Path, body: str) -> Path:
    """Write *body* as an executable shell script at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}")
    path.chmod(0o755)
    return path


def make_prefix(root: Path, **executables: str) -> Path:
    """Build a directory that looks like an environment prefix.

    Always contains ``bin/python``, which is what marks a directory as a
    prefix. Any keyword arguments become further executables in ``bin/``, with
    the value used as the script body.
    """
    write_executable(root / "bin" / "python", "exit 0\n")
    for name, body in executables.items():
        write_executable(root / "bin" / name, body)
    return root


def empty_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Replace ``PATH`` with a single empty directory, so nothing is found on it."""
    nothing = tmp_path / "empty-bin"
    nothing.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PATH", str(nothing))
    return nothing


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Run a git command in *repo*, failing loudly."""
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )


def init_repo(path: Path) -> Path:
    """Initialise a scratch git repository with one commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--quiet", str(path)], check=True, capture_output=True)
    # Pin the identity and disable signing so the commit does not depend on
    # whatever the machine running the suite has in its global git config.
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test Runner")
    git(path, "config", "commit.gpgsign", "false")
    git(path, "commit", "--allow-empty", "--quiet", "-m", "initial")
    return path.resolve()


def make_source_repo(path: Path) -> Path:
    """A scratch repository laid out like this one, with everything committed.

    Holds one file in each of the three regions ``check_clean_tree``
    distinguishes: in-scope model code, and the two exempt subtrees.
    """
    repo = init_repo(path)
    src = repo / "src" / "vivarium_gates_mncnh"
    for relative in ("components/intrapartum.py", "validation/measures.py", "tools/cli.py"):
        target = src / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ORIGINAL = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "--quiet", "-m", "add sources")
    return repo


def point_package_dir_outside_any_checkout(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    """Defeat ``repo_root``'s first resolution step.

    Points the recorded package location at a directory with no ``.git``
    anywhere above it, so resolution has to fall through to git.
    """
    path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(utilities, PACKAGE_DIR_ATTR, path)


class TestRepoRoot:
    """``repo_root`` locates the checkout without consulting the cwd."""

    def test_finds_checkout_from_installed_package(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resolves via the installed package's location, ignoring cwd."""
        # Stand somewhere that is not a checkout at all, so the git fallback
        # could not possibly supply the answer.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        root = utilities.repo_root()

        # ``.git`` is a directory in a clone and a file in a worktree; both are
        # checkouts, so this asks only that it be there.
        assert (root / ".git").exists()
        assert root == PACKAGE_DIR.parent.parent

    def test_falls_back_to_git_toplevel_from_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falls back to ``git rev-parse --show-toplevel`` when the package is not in a checkout."""
        scratch = init_repo(tmp_path / "scratch")
        point_package_dir_outside_any_checkout(monkeypatch, tmp_path / "installed")
        monkeypatch.chdir(scratch)

        assert utilities.repo_root() == scratch

    def test_raises_when_no_checkout_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raises rather than guessing when neither strategy finds a ``.git``."""
        point_package_dir_outside_any_checkout(monkeypatch, tmp_path / "installed")
        nowhere = tmp_path / "nowhere"
        nowhere.mkdir()
        monkeypatch.chdir(nowhere)

        with pytest.raises(RuntimeError):
            utilities.repo_root()


class TestCheckCleanTree:
    """``check_clean_tree`` guards a run's provenance, independent of cwd."""

    def test_raises_on_a_dirty_tracked_file(self, tmp_path: Path) -> None:
        """An uncommitted change under ``src/vivarium_gates_mncnh`` aborts the run."""
        repo = make_source_repo(tmp_path / "repo")
        dirty = repo / "src" / "vivarium_gates_mncnh" / "components" / "intrapartum.py"
        dirty.write_text("ORIGINAL = 2\n")

        with pytest.raises(RuntimeError) as excinfo:
            utilities.check_clean_tree(repo)

        # Naming the offending file is what makes the abort actionable.
        assert "intrapartum.py" in str(excinfo.value)

    def test_ignores_validation_and_tools(self, tmp_path: Path) -> None:
        """Changes under ``validation/`` and ``tools/`` cannot affect results and are allowed."""
        repo = make_source_repo(tmp_path / "repo")
        src = repo / "src" / "vivarium_gates_mncnh"
        (src / "validation" / "measures.py").write_text("ORIGINAL = 2\n")
        (src / "tools" / "cli.py").write_text("ORIGINAL = 2\n")

        utilities.check_clean_tree(repo)

    def test_ignores_untracked_files(self, tmp_path: Path) -> None:
        """Untracked files are not a provenance problem."""
        repo = make_source_repo(tmp_path / "repo")
        src = repo / "src" / "vivarium_gates_mncnh"
        (src / "components" / "scratch_notes.py").write_text("NEW = 1\n")

        utilities.check_clean_tree(repo)

    def test_detects_a_dirty_file_when_run_from_a_subdirectory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: the pathspec used to resolve against cwd, so the guard passed silently
        whenever it ran from anywhere but the repository root."""
        repo = make_source_repo(tmp_path / "repo")
        components = repo / "src" / "vivarium_gates_mncnh" / "components"
        (components / "intrapartum.py").write_text("ORIGINAL = 2\n")

        monkeypatch.setattr(utilities, "repo_root", lambda: repo)
        # A pipeline step's cwd is chosen by the runner, not by the operator.
        monkeypatch.chdir(components)

        with pytest.raises(RuntimeError) as excinfo:
            utilities.check_clean_tree()

        # Naming the file proves the pathspec matched it, rather than the
        # guard having failed for some unrelated reason.
        assert "intrapartum.py" in str(excinfo.value)

    def test_passes_on_a_clean_tree(self, tmp_path: Path) -> None:
        """A clean checkout raises nothing."""
        repo = make_source_repo(tmp_path / "repo")

        assert utilities.check_clean_tree(repo) is None

    def test_raises_when_git_cannot_answer(self, tmp_path: Path) -> None:
        """A git failure is reported, never swallowed into a silent pass."""
        not_a_repo = tmp_path / "not-a-repo"
        not_a_repo.mkdir()

        with pytest.raises(RuntimeError) as excinfo:
            utilities.check_clean_tree(not_a_repo)

        assert "git" in str(excinfo.value).lower()


def psimulate_output(completed: int, total: int, results_dir: str) -> str:
    """A plausible tail of psimulate's output."""
    return (
        "Running jobs...\n"
        f"({completed} of {total} total jobs completed successfully overall)\n"
        f"Results written to: {results_dir}\n"
    )


class TestCheckPsimulateFinished:
    """Adjacent parser, previously untested."""

    def test_true_when_all_jobs_completed(self) -> None:
        """``(N of N total jobs completed successfully overall)`` means success."""
        assert utilities.check_psimulate_finished(psimulate_output(250, 250, "/out")) is True

    def test_false_on_partial_completion(self) -> None:
        """``(M of N ...)`` with M < N means failure."""
        assert utilities.check_psimulate_finished(psimulate_output(249, 250, "/out")) is False

    @pytest.mark.parametrize(
        "output",
        [
            pytest.param("", id="empty"),
            pytest.param("   \n\n  ", id="whitespace-only"),
            pytest.param("psimulate died before it said anything useful", id="one-line"),
        ],
    )
    def test_handles_output_too_short_to_index(self, output: str) -> None:
        """Output with fewer than two lines must not raise IndexError."""
        # A crashed psimulate is exactly when this gets called, so the parser
        # has to survive a truncated tail and report the run as unfinished.
        assert utilities.check_psimulate_finished(output) is False

    def test_false_when_the_line_cannot_be_parsed(self) -> None:
        """Unrecognised output is reported as failure rather than assumed success."""
        unparseable = "some unrelated chatter\nand a bit more\nfinal line\n"

        assert utilities.check_psimulate_finished(unparseable) is False


class TestExtractResultsDir:
    """Adjacent parser, previously untested."""

    def test_parses_the_results_directory(self) -> None:
        """``Results written to: <path>`` yields that path."""
        output = psimulate_output(2, 2, "/mnt/team/results/model29.0.2/ethiopia")

        assert (
            utilities.extract_results_dir(output) == "/mnt/team/results/model29.0.2/ethiopia"
        )

    def test_strips_ansi_colour_codes(self) -> None:
        """Colourised output does not leak escape sequences into the path."""
        output = psimulate_output(2, 2, "\x1b[32m/mnt/team/results/model29.0.2\x1b[0m")

        assert utilities.extract_results_dir(output) == "/mnt/team/results/model29.0.2"

    def test_returns_none_when_absent(self) -> None:
        """Output without the marker line yields ``None``."""
        output = "Running jobs...\n(2 of 2 total jobs completed successfully overall)\n"

        assert utilities.extract_results_dir(output) is None

    @pytest.mark.parametrize(
        "output",
        [
            pytest.param("", id="empty"),
            pytest.param("   ", id="whitespace-only"),
            pytest.param("\n\n", id="newlines-only"),
        ],
    )
    def test_handles_empty_output(self, output: str) -> None:
        """Wholly empty output returns None rather than raising IndexError."""
        # Same shape as the completion parser: a psimulate that dies without
        # printing is exactly when this runs, so it must not add its own crash
        # on top of the one being reported.
        assert utilities.extract_results_dir(output) is None
