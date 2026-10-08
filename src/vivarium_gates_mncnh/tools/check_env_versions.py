"""
Check an environment's package versions against requirements/<type>.txt.

``make build-env`` copies the version files into the environment it builds
(its *record*); ``environment.sh`` uses this script to see if the environment
still matches. It may run before the package is installed, so it uses only the
standard library.

Usage::

    python check_env_versions.py record --repo <repo> --type <type> [--dest <dir>]
    python check_env_versions.py compare --repo <repo> --type <type> [--record-dir <dir>]
    python check_env_versions.py explain-shared-mismatch --repo <repo> --type <type> --record-dir <dir>
    python check_env_versions.py warn-if-overrides [--record-dir <dir>]
    python check_env_versions.py show-changes <old file> <new file>
    python check_env_versions.py write-install-constraints --repo <repo> --type <type> --out <file>
    python check_env_versions.py installed-matches-version-file --repo <repo> --type <type>
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ENV_TYPES = ("simulation", "artifact")
OVERRIDES_FILE = "overrides.txt"
RECORD_SUBDIR = Path("etc") / "vivarium_gates_mncnh"  # inside an environment's prefix
MAIN_REF = "origin/main"  # what the shared environment is built from

# Exit codes for ``compare``. They skip 1 and 2, which mean a crash or a usage error.
EXIT_MATCH = 0
EXIT_DIFFERENT = 10
EXIT_NO_RECORD = 11
EXIT_ERROR = 12

# "name==version" or "name @ url", with optional extras like name[cluster].
_NAME = r"([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)(?:\s*\[[^\]]*\])?"
_REQUIREMENT_RE = re.compile(r"^" + _NAME + r"\s*(?:==\s*([^\s,=;][^\s,;]*)|@\s*(\S+))$")


def version_file_name(env_type: str) -> str:
    """Return ``simulation.txt`` or ``artifact.txt``."""
    if env_type not in ENV_TYPES:
        raise ValueError(
            "Unknown environment type {!r}; expected one of {}.".format(
                env_type, ", ".join(ENV_TYPES)
            )
        )
    return env_type + ".txt"


def normalize_name(name: str) -> str:
    """Normalize a package name, so ``Foo_Bar.baz`` and ``foo-bar-baz`` match."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_line(raw_line: str) -> Optional[Tuple[str, str]]:
    """Return ``(name, version or url)`` for a requirement line; None for comments and blanks."""
    line = re.sub(r"\s+#.*$", "", raw_line.strip()).split(";", 1)[0].strip()
    if not line or line.startswith(("#", "-")):
        return None
    match = _REQUIREMENT_RE.match(line)
    if not match:
        raise ValueError(
            "Requirement is neither an '==' pin nor 'name @ <url>': {!r}".format(
                raw_line.strip()
            )
        )
    return normalize_name(match.group(1)), match.group(2) or match.group(3)


def parse_requirements(text: str) -> Dict[str, str]:
    """Read a version or overrides file into ``{name: version or url}``."""
    requirements = {}  # type: Dict[str, str]
    for raw_line in text.splitlines():
        parsed = _parse_line(raw_line)
        if parsed is None:
            continue
        if parsed[0] in requirements:
            raise ValueError("Package {!r} is listed more than once.".format(parsed[0]))
        requirements[parsed[0]] = parsed[1]
    return requirements


def describe_changes(old: Dict[str, str], new: Dict[str, str]) -> List[str]:
    """One sorted line per package that changed, was added, or was removed."""
    lines = []
    for name in sorted(set(old) | set(new)):
        if name not in new:
            lines.append("{}: removed (was {})".format(name, old[name]))
        elif name not in old:
            lines.append("{}: added ({})".format(name, new[name]))
        elif old[name] != new[name]:
            lines.append("{}: {} -> {}".format(name, old[name], new[name]))
    return lines


def _run_git(repo: Path, *args: str) -> Optional[str]:
    """Run git in ``repo``; return its output, or None if it fails."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo)] + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _versions(pins: str, overrides: str) -> Dict[str, str]:
    """Combine a version file and an overrides file; override entries start with "override "."""
    versions = parse_requirements(pins)
    versions.update(
        {"override " + n: ref for n, ref in parse_requirements(overrides).items()}
    )
    return versions


def _read_versions(folder: Path, env_type: str) -> Optional[Dict[str, str]]:
    """Read a checkout's requirements/ or a record folder; None if it has no version file."""
    pins_file = Path(folder) / version_file_name(env_type)
    if not pins_file.is_file():
        return None
    overrides_file = Path(folder) / OVERRIDES_FILE
    overrides = overrides_file.read_text() if overrides_file.is_file() else ""
    return _versions(pins_file.read_text(), overrides)


def _read_versions_at(repo: Path, ref: str, env_type: str) -> Optional[Dict[str, str]]:
    """Read the version files committed at a git ref; None if the ref or file is missing."""
    if not ref or ref.startswith("-"):
        return None
    pins = _run_git(
        repo, "show", "{}:requirements/{}".format(ref, version_file_name(env_type))
    )
    if pins is None:
        return None
    overrides = _run_git(repo, "show", "{}:requirements/{}".format(ref, OVERRIDES_FILE))
    return _versions(pins, overrides or "")


def _overridden(repo: Path) -> set:
    """The names in the checkout's overrides file."""
    path = Path(repo) / "requirements" / OVERRIDES_FILE
    return set(parse_requirements(path.read_text())) if path.is_file() else set()


def _pins_file(repo: Path, env_type: str) -> Path:
    """The checkout's version file, or a clear error if it is missing."""
    path = Path(repo) / "requirements" / version_file_name(env_type)
    if not path.is_file():
        raise FileNotFoundError("No version file at {}".format(path))
    return path


def _record(args: argparse.Namespace) -> int:
    """Copy the checkout's version file and overrides (or remove old ones) into the env."""
    pins_file = _pins_file(args.repo, args.env_type)
    overrides_file = pins_file.parent / OVERRIDES_FILE
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(pins_file), str(dest / pins_file.name))
    if overrides_file.is_file():
        shutil.copyfile(str(overrides_file), str(dest / OVERRIDES_FILE))
    elif (dest / OVERRIDES_FILE).exists():
        (dest / OVERRIDES_FILE).unlink()
    return 0


def _compare(args: argparse.Namespace) -> int:
    """Compare the env's record with the checkout and print any differences."""
    repo = Path(args.repo)
    recorded = _read_versions(args.record_dir, args.env_type)
    if recorded is None:
        return EXIT_NO_RECORD
    checkout = _read_versions(repo / "requirements", args.env_type)
    if checkout is None:
        raise FileNotFoundError("No version file for {} in {}".format(args.env_type, repo))
    changes = describe_changes(recorded, checkout)
    if not changes:
        return EXIT_MATCH
    file_name = version_file_name(args.env_type)
    print(
        "The {} environment's package versions differ from requirements/{}:".format(
            args.env_type, file_name
        )
    )
    for change in changes:
        print("    " + change)
    # The last commit that changed this type's version file or the overrides.
    commit = _run_git(
        repo,
        "log",
        "-1",
        "--date=short",
        "--format=%h %s (%an, %ad)",
        "HEAD",
        "--",
        ":/requirements/" + file_name,
        ":/requirements/" + OVERRIDES_FILE,
    )
    print("Last version change: {}".format((commit or "").strip() or "not committed yet"))
    return EXIT_DIFFERENT


def _explain_shared_mismatch(args: argparse.Namespace) -> int:
    """If the shared env's versions differ from the checkout's, say why and what to do."""
    repo = Path(args.repo)
    checkout = _read_versions(repo / "requirements", args.env_type)
    if checkout is None:
        raise FileNotFoundError("No version file for {} in {}".format(args.env_type, repo))
    recorded = _read_versions(args.record_dir, args.env_type)
    if checkout == recorded:
        return 0
    main = _read_versions_at(repo, MAIN_REF, args.env_type)
    merge_base = (_run_git(repo, "merge-base", "HEAD", MAIN_REF) or "").strip()
    build_own = [
        "Your branch changed its version files (or uses framework overrides),",
        "which the shared environment cannot provide. Build your own",
        "environment instead:",
        "    source environment.sh -t {}".format(args.env_type),
    ]

    # Checked in order; the first that applies explains the mismatch.
    if any(name.startswith("override ") for name in checkout):
        next_steps = build_own  # a shared environment never has overrides
    elif recorded is None or main is None:
        next_steps = [
            "Could not tell why (main's version files or the shared environment's "
            "record are unavailable)."
        ]
    elif checkout == main:
        next_steps = [
            "Your checkout matches main, but the shared environment is behind main:",
            "it has not yet been rebuilt from main's version files. It will catch up",
            "after its next nightly rebuild. Meanwhile, you can build your own",
            "environment with:",
            "    source environment.sh -t {}".format(args.env_type),
        ]
    elif checkout == _read_versions_at(repo, merge_base, args.env_type):
        next_steps = [
            "Your branch has not changed its version files, but main has moved on",
            "and the shared environment follows main. Merge main into your branch",
            "(e.g. 'git merge {}') to pick up main's versions.".format(MAIN_REF),
        ]
    else:
        next_steps = build_own

    if recorded is None:
        changes = [
            "(the shared environment has no record of its versions in {})".format(
                args.record_dir
            )
        ]
    else:
        changes = describe_changes(recorded, checkout)
    print(
        "WARNING: this checkout's {} package versions differ from the shared "
        "environment's:".format(args.env_type)
    )
    print("\n".join(["    " + change for change in changes] + [""] + next_steps))
    return 0


def _warn_if_overrides(args: argparse.Namespace) -> int:
    """Print a warning if the environment was built with framework overrides."""
    overrides_file = Path(args.record_dir) / OVERRIDES_FILE
    if not overrides_file.is_file():
        return 0
    overrides = parse_requirements(overrides_file.read_text())
    if overrides:
        rule = "=" * 78
        print(rule)
        print("WARNING: not a standard environment. It was built with framework overrides")
        print("from requirements/{}:".format(OVERRIDES_FILE))
        for name, ref in sorted(overrides.items()):
            print("    {} @ {}".format(name, ref))
        print(rule)
    return 0


def _show_changes(args: argparse.Namespace) -> int:
    old = parse_requirements(Path(args.old).read_text())
    new = parse_requirements(Path(args.new).read_text())
    for line in describe_changes(old, new):
        print(line)
    return 0


def _write_install_constraints(args: argparse.Namespace) -> int:
    """Write the version file without the overridden packages, for uv.

    uv can't install a git version of a package that is also pinned to a release.
    """
    pins_file = _pins_file(args.repo, args.env_type)
    text = pins_file.read_text()
    parse_requirements(text)  # fail on a bad file before writing anything
    overridden = _overridden(args.repo)
    kept = [
        "# Generated by make install from requirements/{} minus overridden packages; "
        "do not edit.\n".format(pins_file.name)
    ]
    for line in text.splitlines(keepends=True):
        parsed = _parse_line(line)
        if parsed is None or parsed[0] not in overridden:
            kept.append(line)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(kept))
    return 0


def _installed_matches_version_file(args: argparse.Namespace) -> int:
    """List pinned packages installed at a different version; skip overridden ones."""
    import importlib.metadata as metadata  # Python 3.8+

    pins_file = _pins_file(args.repo, args.env_type)
    pins = parse_requirements(pins_file.read_text())
    overridden = _overridden(args.repo)
    installed = {}  # type: Dict[str, str]
    for dist in metadata.distributions():
        if dist.metadata["Name"]:
            # The first one found for a name is the one Python imports.
            installed.setdefault(normalize_name(dist.metadata["Name"]), dist.version)
    mismatches = [
        "{}: pinned {}, installed {}".format(name, pinned, installed[name])
        for name, pinned in sorted(pins.items())
        if name in installed
        and name not in overridden
        and "/" not in pinned  # a git reference, not a version
        and installed[name].lower() != pinned.lower()
    ]
    if not mismatches:
        return 0
    print("Installed packages differ from requirements/{}:".format(pins_file.name))
    for mismatch in mismatches:
        print("    " + mismatch)
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one subcommand and return its exit code (see the module docstring)."""
    default_record_dir = Path(sys.prefix) / RECORD_SUBDIR
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    # Each subcommand: its function, its exit code on error, and whether errors only warn.
    # explain-shared-mismatch and warn-if-overrides run on every activation, so never fail.
    handlers = {}
    for name, function, error_code, warn_only in [
        ("record", _record, 1, False),
        ("compare", _compare, EXIT_ERROR, False),
        ("explain-shared-mismatch", _explain_shared_mismatch, 0, True),
        ("warn-if-overrides", _warn_if_overrides, 0, True),
        ("show-changes", _show_changes, 1, False),
        ("write-install-constraints", _write_install_constraints, 1, False),
        ("installed-matches-version-file", _installed_matches_version_file, 1, False),
    ]:
        handlers[name] = (function, error_code, warn_only)
        command = commands.add_parser(name)
        if name not in ("warn-if-overrides", "show-changes"):
            command.add_argument("--repo", type=Path, required=True)
            # Checked by the command, so a bad value gets its exit code, not argparse's.
            command.add_argument("--type", dest="env_type", required=True)
        if name == "record":
            command.add_argument("--dest", type=Path, default=default_record_dir)
        if name in ("compare", "warn-if-overrides"):
            command.add_argument("--record-dir", type=Path, default=default_record_dir)
        if name == "explain-shared-mismatch":
            command.add_argument("--record-dir", type=Path, required=True)
        if name == "show-changes":
            command.add_argument("old", type=Path)
            command.add_argument("new", type=Path)
        if name == "write-install-constraints":
            command.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    function, error_code, warn_only = handlers[args.command]
    try:
        return function(args)
    except Exception as error:
        if warn_only:
            print("WARNING: {} failed: {}".format(args.command, error))
        else:
            print("ERROR: {} failed: {}".format(args.command, error), file=sys.stderr)
        return error_code


if __name__ == "__main__":
    sys.exit(main())
