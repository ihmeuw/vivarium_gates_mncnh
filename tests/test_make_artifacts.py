"""Tests for how build_artifacts uses its requirements.txt record.

The record itself comes from psimulate's ``pip_env``, which has its own tests.
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, List

import click
import pytest

from vivarium_gates_mncnh.constants import metadata
from vivarium_gates_mncnh.tools import make_artifacts
from vivarium_gates_mncnh.utilities import sanitize_location

RECORD = "requirements.txt"


class ScriptedPrompts:
    """Stands in for ``click.confirm``: gives scripted answers and records each prompt."""

    def __init__(self, answers: List[bool]):
        self.answers = list(answers)
        self.prompts: List[str] = []

    def __call__(self, text: Any = "", *args: Any, **kwargs: Any) -> bool:
        assert self.answers, f"unexpected prompt: {text}"
        self.prompts.append(str(text))
        answer = self.answers.pop(0)
        if not answer and kwargs.get("abort"):
            raise click.exceptions.Abort()
        return answer

    @property
    def deletion_prompts(self) -> List[bool]:
        return ["Existing artifacts" in prompt for prompt in self.prompts]


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Run build_artifacts on the cluster, recording builds instead of building."""
    built: List[str] = []
    # pip_env.validate runs `pip list`; use this interpreter's pip.
    path = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    monkeypatch.setenv("PATH", path)
    monkeypatch.setattr(make_artifacts, "running_from_cluster", lambda: True)
    for name in ("build_single", "build_all_artifacts"):
        monkeypatch.setattr(make_artifacts, name, lambda *a, n=name, **k: built.append(n))
    return built


def _answer(monkeypatch: pytest.MonkeyPatch, *answers: bool) -> ScriptedPrompts:
    prompts = ScriptedPrompts(list(answers))
    # pip_env.validate calls click.confirm through the module attribute.
    monkeypatch.setattr(click, "confirm", prompts)
    return prompts


def _build(
    output_dir: Path, location: str, append: bool = False, resume: bool = False
) -> None:
    make_artifacts.build_artifacts(
        location=location,
        years=None,
        output_dir=str(output_dir),
        append=append,
        replace_keys=(),
        verbose=0,
        resume=resume,
    )


def _stale_record(output_dir: Path) -> str:
    """A record of this environment, but with a different click version."""
    output_dir.mkdir(parents=True, exist_ok=True)
    make_artifacts.check_environment_record(output_dir)
    record = output_dir / RECORD
    stale = re.sub(r"(?im)^click==.*$", "click==0.0.1", record.read_text())
    record.write_text(stale)
    return stale


def _artifact(output_dir: Path, name: str) -> Path:
    path = output_dir / f"{name}.hdf"
    path.write_text("not really an artifact")
    return path


def test_record_is_checked_before_anything_is_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, builds: List[str]
) -> None:
    """A mismatched record is raised first, so declining it leaves every artifact in place."""
    location = metadata.LOCATIONS[0]
    stale = _stale_record(tmp_path)
    artifact = _artifact(tmp_path, sanitize_location(location))
    prompts = _answer(monkeypatch, False)

    with pytest.raises(click.exceptions.Abort):
        _build(tmp_path, location)

    assert prompts.deletion_prompts == [False]  # only the record prompt was shown
    assert artifact.exists() and (tmp_path / RECORD).read_text() == stale
    assert builds == []
