"""
Regenerate the version files, requirements/<type>.txt, with ``uv pip compile``.

``make lock-versions`` and ``make upgrade-versions`` run this. Each type is resolved
into a temporary directory, and the version files are replaced only if every type
resolves. Then it prints the pins that moved. Run it from an activated environment of
this repository, so ``uv`` is on the PATH. The files are resolved for that environment's
exact Python version, which must be the newest supported minor version.

Usage::

    python lock_versions.py --repo <repo> --target <make target> --python-version <x.y>
        --extra <type>=<extra> [--extra ...] [--type <type>] [--upgrade] [-- <uv flags>]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Run by file path, so import check_env_versions.py from this directory, not the package
# (which may not be installed, or may be another checkout's).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_env_versions import describe_changes, parse_requirements  # noqa: E402

RESOLVER_CONSTRAINTS = "resolver-constraints.txt"
# Pinned too, though nothing in pyproject.toml asks for them: they build the environment.
BUILD_TOOLS = "uv\npip\nsetuptools\n"
PACKAGE = "vivarium-gates-mncnh"  # this repository's package, which is never pinned


def _compile(args: argparse.Namespace, env_type: str, extra: str, output: Path) -> None:
    """Resolve one type into ``output``, which starts as a copy of its current version file."""
    requirements = Path(args.repo) / "requirements"
    command = [
        "uv",
        "pip",
        "compile",
        str(Path(args.repo) / "pyproject.toml"),
        "-",
        "--extra",
        extra,
        "--python-version",
        args.exact_python,
        "--no-emit-package",
        PACKAGE,
        "--custom-compile-command",
        "make {} type={}".format(args.target, env_type),
        "-o",
        str(output),
        "--quiet",  # the result is in the file; errors still print
    ]
    if (requirements / RESOLVER_CONSTRAINTS).is_file():
        command += ["-c", str(requirements / RESOLVER_CONSTRAINTS)]
    if args.upgrade:
        command.append("--upgrade")
    command += args.uv_flags
    print(
        "Resolving {}.txt (extra '{}', Python {})".format(env_type, extra, args.exact_python)
    )
    subprocess.run(command, input=BUILD_TOOLS, encoding="utf-8", check=True)


def _report(env_type: str, old: Optional[str], new: str) -> None:
    """Print the pins that moved, were added or were removed."""
    if old is None:
        print("Created {}.txt.".format(env_type))
        return
    moved = describe_changes(parse_requirements(old), parse_requirements(new))
    if not moved:
        print("No existing pins moved in {}.txt.".format(env_type))
        return
    print("Pins that moved in {}.txt:".format(env_type))
    for line in moved:
        print("    " + line)


def lock(args: argparse.Namespace) -> None:
    """Resolve every requested type, then replace their version files together."""
    extras = dict(pair.split("=", 1) for pair in args.extra)  # type: Dict[str, str]
    if args.env_type == "all":
        env_types = list(extras)  # type: List[str]
    elif args.env_type in extras:
        env_types = [args.env_type]
    else:
        raise ValueError(
            "Unknown environment type {!r}; expected one of {}.".format(
                args.env_type, ", ".join(list(extras) + ["all"])
            )
        )
    # The exact version, not just the minor version: uv treats "3.11" as 3.11.0, so a
    # dependency marked for python_full_version <= "3.11.0a6" (or >= "3.11.5") would be pinned
    # (or left unpinned) when the environment, on a later patch release, doesn't (or does)
    # install it.
    if "{}.{}".format(*sys.version_info[:2]) != args.python_version:
        raise RuntimeError(
            "This is Python {}.{}; activate an environment of this repository built for "
            "Python {}.".format(*sys.version_info[:2], args.python_version)
        )
    args.exact_python = "{}.{}.{}".format(*sys.version_info[:3])
    if shutil.which("uv") is None:
        raise FileNotFoundError(
            "uv is not on the PATH; activate this repository's environment"
        )

    requirements = Path(args.repo) / "requirements"
    old = {}  # type: Dict[str, Optional[str]]
    new = {}  # type: Dict[str, str]
    with tempfile.TemporaryDirectory() as tmp_dir:
        for env_type in env_types:
            current = requirements / (env_type + ".txt")
            output = Path(tmp_dir) / (env_type + ".txt")
            old[env_type] = current.read_text() if current.is_file() else None
            if old[env_type] is not None:
                # uv prefers the pins already in the output file.
                shutil.copyfile(str(current), str(output))
            _compile(args, env_type, extras[env_type], output)
            new[env_type] = output.read_text()

    for env_type in env_types:
        (requirements / (env_type + ".txt")).write_text(new[env_type])
    for env_type in env_types:
        _report(env_type, old[env_type], new[env_type])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--target", required=True, help="the make target, for the file header"
    )
    parser.add_argument("--python-version", required=True)
    parser.add_argument(
        "--extra", action="append", required=True, help="<type>=<extra>, once per type"
    )
    parser.add_argument("--type", dest="env_type", default="all")
    parser.add_argument("--upgrade", action="store_true", help="ignore the existing pins")
    parser.add_argument("uv_flags", nargs="*", help="passed to uv pip compile, after --")
    args = parser.parse_args(argv)
    try:
        lock(args)
    except subprocess.CalledProcessError:
        print("ERROR: uv pip compile failed; no version file was changed.", file=sys.stderr)
        return 1
    except Exception as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
