"""Tests for tools/lock_versions.py, run by file path with a fake uv on the PATH."""

import os
import subprocess
import sys
from pathlib import Path

from .conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "src" / "vivarium_gates_mncnh" / "tools" / "lock_versions.py"

# Stands in for `uv pip compile`: writes $FAKE_UV_<EXTRA> to the -o file, or fails if unset.
FAKE_UV = """#!{python}
import os, sys
args = sys.argv[1:]
output = os.environ.get("FAKE_UV_" + args[args.index("--extra") + 1].upper())
if output is None:
    sys.exit("resolution failed")
open(args[args.index("-o") + 1], "w").write(output)
"""

SIM_OLD = "numpy==1.26.4\npandas==2.0.0\n"
SIM_NEW = "numpy==1.26.4\npandas==2.1.0\nscipy==1.11.0\n"
ART = "numpy==1.26.4\n"


def _lock(tmp_path: Path, **outputs: str) -> subprocess.CompletedProcess:
    """Run lock_versions.py on a repo holding SIM_OLD and ART; ``outputs`` maps extra to result."""
    repo = tmp_path / "repo"
    (repo / "requirements").mkdir(parents=True)
    (repo / "requirements" / "simulation.txt").write_text(SIM_OLD)
    (repo / "requirements" / "artifact.txt").write_text(ART)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(FAKE_UV.format(python=sys.executable))
    (bin_dir / "uv").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["PATH"] = "{}{}{}".format(bin_dir, os.pathsep, env.get("PATH", ""))
    env.update({"FAKE_UV_" + extra.upper(): text for extra, text in outputs.items()})
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--repo", str(repo), "--target", "lock-versions"]
        + [
            "--python-version",
            "{}.{}".format(*sys.version_info[:2]),
            "--extra",
            "simulation=dev",
            "--extra",
            "artifact=data",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_replaces_the_files_and_reports_moved_pins(tmp_path: Path) -> None:
    result = _lock(tmp_path, dev=SIM_NEW, data=ART)

    assert result.returncode == 0, result.stdout + result.stderr
    requirements = tmp_path / "repo" / "requirements"
    assert (requirements / "simulation.txt").read_text() == SIM_NEW
    assert "Pins that moved in simulation.txt:\n    pandas: 2.0.0 -> 2.1.0\n" in result.stdout
    assert "    scipy: added (1.11.0)\n" in result.stdout
    assert "No existing pins moved in artifact.txt." in result.stdout


def test_changes_no_file_if_any_type_fails(tmp_path: Path) -> None:
    result = _lock(tmp_path, dev=SIM_NEW)  # the artifact type fails to resolve

    assert result.returncode == 1
    assert "no version file was changed" in result.stderr
    requirements = tmp_path / "repo" / "requirements"
    assert (requirements / "simulation.txt").read_text() == SIM_OLD
    assert (requirements / "artifact.txt").read_text() == ART
