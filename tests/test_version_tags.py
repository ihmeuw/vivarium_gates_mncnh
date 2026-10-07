"""The package version is derived only from release-style git tags."""

import fnmatch
import shlex
import sys

import pytest

from .conftest import REPO_ROOT

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


def _describe_match_glob() -> str:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]
    command = config["setuptools_scm"]["git_describe_command"]
    if isinstance(command, str):
        command = shlex.split(command)
    return command[command.index("--match") + 1]


@pytest.mark.parametrize("tag", ["v40.1", "v41.0", "v43.0a", "v29.0.3", "v44.0"])
def test_release_tags_are_considered(tag: str) -> None:
    """Model-number tags keep driving the package version."""
    assert fnmatch.fnmatchcase(tag, _describe_match_glob())


@pytest.mark.parametrize(
    "tag", ["vtest_mic7347", "vget_draws_migration", "vmodel44_run2", "aph_removal"]
)
def test_stray_tags_are_ignored(tag: str) -> None:
    """Test, event and non-numbered tags can never become the version source."""
    assert not fnmatch.fnmatchcase(tag, _describe_match_glob())
