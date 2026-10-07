"""Tests for the Makefile's install, build-env and lock targets.

``pip``, ``uv`` and ``conda`` are replaced with recorders, so nothing is installed.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from vivarium.build_utils.resources import get_makefiles_path

from vivarium_gates_mncnh.tools.env_versions import parse_requirements

from .conftest import OVERRIDES, REPO_ROOT

ENV_VERSIONS = Path("src") / "vivarium_gates_mncnh" / "tools" / "env_versions.py"

# Logs each call as a JSON line, with UV_CONSTRAINT's file contents (make may
# regenerate it). Exits 1 when every "+"-joined token of a group in
# RECORDER_FAIL_ON is one of its arguments.
RECORDER = """\
import json, os, sys
log, prog, args = sys.argv[1], sys.argv[2], sys.argv[3:]
constraint = os.environ.get("UV_CONSTRAINT")
content = None
if constraint and os.path.isfile(constraint):
    content = open(constraint).read()
with open(log, "a") as f:
    f.write(json.dumps({"prog": prog, "argv": args, "UV_CONSTRAINT": constraint,
        "UV_CONSTRAINT_CONTENT": content, "UV_OVERRIDE": os.environ.get("UV_OVERRIDE")}) + "\\n")
names = set(args) | {os.path.basename(a) for a in args}
for group in filter(None, os.environ.get("RECORDER_FAIL_ON", "").split(",")):
    if all(token in names for token in group.split("+")):
        sys.exit(1)
"""
_SCRUBBED_ENV = (
    "MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEFILES", "JENKINS_URL", "ENV_REQS", "UV_FLAGS",
    "CHANGED_LIBS", "UV_CONSTRAINT", "UV_OVERRIDE", "type", "name", "path", "py", "lfs",
    "force", "include_timestamp", "RECORDER_FAIL_ON",
)  # fmt: skip


class MakeSandbox:
    """A copy of the repo's build files, with recorder ``pip``, ``uv`` and ``conda``."""

    def __init__(self, root: Path, overrides: Optional[str] = None, jenkins: bool = False):
        self.project = root / "vivarium_gates_mncnh"
        self.project.mkdir(parents=True)
        for name in ("Makefile", "pyproject.toml", "python_versions.json"):
            shutil.copy(REPO_ROOT / name, self.project / name)
        shutil.copytree(REPO_ROOT / "requirements", self.project / "requirements")
        (self.project / ENV_VERSIONS).parent.mkdir(parents=True)
        shutil.copy(REPO_ROOT / ENV_VERSIONS, self.project / ENV_VERSIONS)
        self.overrides_file.unlink(missing_ok=True)
        if overrides is not None:
            self.overrides_file.write_text(overrides)
        self.jenkins = jenkins
        if jenkins:  # Jenkins puts the shared makefiles in the workspace
            for name in ("base.mk", "test.mk"):
                shutil.copy(Path(get_makefiles_path()) / name, self.project / name)
        self.fail_on = ""
        self.log = root / "calls.jsonl"
        self.bin = root / "bin"
        self.bin.mkdir()
        (root / "recorder.py").write_text(RECORDER)
        for prog in ("pip", "uv", "conda"):
            self.script(
                prog,
                f'exec "{sys.executable}" -I "{root / "recorder.py"}" "{self.log}" {prog} "$@"',
            )
        self.script("python", f'exec "{sys.executable}" "$@"')  # so make can find base.mk

    @property
    def overrides_file(self) -> Path:
        return self.project / "requirements" / "overrides.txt"

    def version_file(self, env_type: str) -> Path:
        return self.project / "requirements" / f"{env_type}.txt"

    def script(self, name: str, body: str) -> None:
        path = self.bin / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)

    def make(self, *args: str) -> "subprocess.CompletedProcess[str]":
        env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED_ENV}
        env["PATH"] = f"{self.bin}{os.pathsep}{env.get('PATH', '')}"
        env["CONDA_PREFIX"] = str(self.bin.parent / "conda_prefix")
        env.update({"JENKINS_URL": "1"} if self.jenkins else {})
        env.update({"RECORDER_FAIL_ON": self.fail_on} if self.fail_on else {})
        self.log.unlink(missing_ok=True)
        return subprocess.run(
            ["make", "-C", str(self.project), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def calls(self) -> List[Dict[str, Any]]:
        return (
            [json.loads(line) for line in self.log.read_text().splitlines()]
            if self.log.exists()
            else []
        )

    def install(self, *args: str) -> Dict[str, Any]:
        """Run ``make install`` and return the uv call that installs the package."""
        result = self.make("install", *args)
        assert result.returncode == 0, result.stdout + result.stderr
        [call] = [c for c in self.calls() if c["prog"] == "uv" and "-e" in c["argv"]]
        return call


def _pins(text: str) -> Dict[str, str]:
    return parse_requirements(text)


class TestInstall:
    @pytest.mark.parametrize(
        "args, jenkins, env_type, extra",
        [
            ((), False, "simulation", "dev"),
            (("ENV_REQS=data",), False, "artifact", "data"),
            (("ENV_REQS= data ",), False, "artifact", None),  # base.mk keeps the spaces
            (("ENV_REQS=test",), False, "simulation", "test"),
            (("UV_FLAGS=--no-cache",), True, "simulation", "dev"),  # the Jenkins PR build
        ],
    )
    def test_constrained_by_the_type_version_file(
        self, tmp_path: Path, args: tuple, jenkins: bool, env_type: str, extra: Optional[str]
    ) -> None:
        """Every install, including Jenkins', is constrained by its type's version file."""
        sandbox = MakeSandbox(tmp_path, jenkins=jenkins)

        call = sandbox.install(*args)

        if extra is not None:
            assert f".[{extra}]" in call["argv"]
        assert (
            Path(call["UV_CONSTRAINT"]).resolve() == sandbox.version_file(env_type).resolve()
        )
        assert call["UV_OVERRIDE"] is None

    @pytest.mark.parametrize(
        "env_reqs, env_type", [("dev", "simulation"), ("data", "artifact")]
    )
    @pytest.mark.parametrize(
        "overrides, dropped",
        [(OVERRIDES, {"vivarium-public-health"})],
    )
    def test_overrides_drop_only_their_own_pins(
        self, tmp_path: Path, env_reqs: str, env_type: str, overrides: str, dropped: set
    ) -> None:
        """With overrides, uv gets them plus every pin except the overridden ones."""
        sandbox = MakeSandbox(tmp_path, overrides=overrides)
        committed = _pins(sandbox.version_file(env_type).read_text())

        call = sandbox.install(f"ENV_REQS={env_reqs}")

        assert Path(call["UV_OVERRIDE"]).resolve() == sandbox.overrides_file.resolve()
        constrained = _pins(call["UV_CONSTRAINT_CONTENT"])
        assert constrained == {n: v for n, v in committed.items() if n not in dropped}
        assert "resolver-constraints" not in call["UV_CONSTRAINT"]

        # Removing the overrides puts the plain version file back.
        sandbox.overrides_file.unlink()
        call = sandbox.install(f"ENV_REQS={env_reqs}")
        assert call["UV_OVERRIDE"] is None
        assert (
            Path(call["UV_CONSTRAINT"]).resolve() == sandbox.version_file(env_type).resolve()
        )


def _steps(sandbox: MakeSandbox) -> List[str]:
    """The build-env steps that ran through conda: install, the version check, record."""
    steps = []
    for call in sandbox.calls():
        argv = call["argv"]
        if call["prog"] == "conda" and any(a.endswith("env_versions.py") for a in argv):
            steps += [s for s in ("installed-matches-version-file", "record") if s in argv]
        elif call["prog"] == "conda" and "make" in argv and "install" in argv:
            steps.append("install")
    return steps


def _drop_pin(sandbox: MakeSandbox, name: str) -> None:
    path = sandbox.version_file("simulation")
    lines = path.read_text().splitlines()
    path.write_text("\n".join(l for l in lines if not re.match(rf"{name}\s*==", l)) + "\n")


class TestBuildEnv:
    @pytest.mark.parametrize(
        "break_build, expected_steps",
        [
            (None, ["install", "installed-matches-version-file", "record"]),
            ("drop uv pin", []),
            ("drop vivarium-build-utils pin", []),
            ("make+install", ["install"]),
            (
                "env_versions.py+installed-matches-version-file",
                ["install", "installed-matches-version-file"],
            ),
        ],
    )
    def test_a_failed_step_leaves_no_record(
        self, tmp_path: Path, break_build: Optional[str], expected_steps: List[str]
    ) -> None:
        """Any failing step stops the build before the environment's versions are recorded."""
        sandbox = MakeSandbox(tmp_path)
        if break_build and break_build.startswith("drop "):
            _drop_pin(sandbox, break_build.split()[1])
        elif break_build:
            sandbox.fail_on = break_build

        result = sandbox.make("build-env", "type=simulation", f"path={tmp_path / 'env'}")

        assert (result.returncode == 0) == (break_build is None), (
            result.stdout + result.stderr
        )
        assert _steps(sandbox) == expected_steps
