"""
Utility functions for PAF simulation workflow.
"""
import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import pyarrow.parquet as pq


def tag_commit(tag: str) -> Optional[str]:
    """Return the commit SHA a tag points to, or ``None`` if it doesn't exist."""
    result = subprocess.run(
        ["git", "rev-list", "-n1", tag],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def head_commit() -> str:
    """Return the SHA of the current HEAD."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def commit_pending_changes(message: str) -> None:
    """Commit any uncommitted tracked-file changes with ``message`` and push to origin.

    No-op if the working tree has nothing to commit. Untracked files are ignored.
    """
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    if not status.stdout.strip():
        print("No pending changes to commit.")
        return

    try:
        subprocess.run(["git", "add", "-u"], check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "commit", "-m", message], check=True, capture_output=True, text=True
        )
        print(f"Committed pending changes: {message!r}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to commit pending changes.\n  stderr: {e.stderr.strip()}")

    try:
        subprocess.run(
            ["git", "push", "origin", "HEAD"], check=True, capture_output=True, text=True
        )
        print("Pushed commit to origin.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to push commit to origin.\n  stderr: {e.stderr.strip()}")


def create_and_push_tag(model_number: str, push: bool = True) -> None:
    """Create a git tag ``v{model_number}`` for the current HEAD, optionally pushing it.

    If the tag already exists on the current commit, it is left as-is. If it
    exists on a *different* commit the user is prompted to force-update;
    when stdin is not a TTY (e.g. running inside a jobmon task) the prompt
    aborts cleanly instead of crashing.

    Parameters
    ----------
    model_number
        The model version number (e.g. "29.0.2"); the tag is ``v{model_number}``.
    push
        If False, the tag is created locally only and never pushed to origin.
        Pushing a tag on an unpushed branch publishes every commit on it, which
        is not always wanted.
    """
    tag = f"v{model_number}"
    force = False

    existing_commit = tag_commit(tag)
    if existing_commit is not None:
        head = head_commit()
        if existing_commit == head:
            print(f"\nGit tag '{tag}' already exists on the current commit. Skipping.")
            return
        print(f"\nWARNING: Git tag '{tag}' already exists (on commit {existing_commit[:8]}).")
        try:
            response = input(
                f"Update tag '{tag}' to the current commit and force-push? [y/N] "
            )
        except EOFError:
            raise RuntimeError(
                f"Git tag '{tag}' already exists on a different commit "
                f"({existing_commit[:8]}). Refusing to force-update non-interactively. "
                "Resolve the tag conflict manually before re-running."
            )
        if response.strip().lower() != "y":
            raise RuntimeError(f"Aborted: tag '{tag}' already exists.")
        force = True

    action = "Updating" if force else "Creating"
    print(
        f"\n{action} git tag '{tag}'{' and pushing to origin' if push else ' (local only)'}..."
    )

    try:
        cmd = ["git", "tag", "-f", tag] if force else ["git", "tag", tag]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  {'Updated' if force else 'Created'} tag '{tag}'")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to create git tag '{tag}'.\n  stderr: {e.stderr.strip()}")

    if not push:
        print(f"  Skipping push of tag '{tag}' to origin (--no-push)")
        return

    try:
        cmd = (
            ["git", "push", "--force", "origin", tag]
            if force
            else ["git", "push", "origin", tag]
        )
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  Pushed tag '{tag}' to origin")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to push git tag '{tag}' to origin.\n  stderr: {e.stderr.strip()}"
        )


# Paths under src/vivarium_gates_mncnh that check_clean_tree() deliberately ignores.
#
# The guard exists to protect the provenance of a run: everything the simulation
# actually executes must be committed, so results can be traced to a git revision.
# It is not a tidiness check, so anything that cannot change simulation behavior is
# excluded -- and anything the run tooling itself is expected to rewrite *while*
# running must be excluded, or the tooling cannot be composed with itself.
#
#   validation/  -- post-hoc V&V analysis code. Imported by notebooks, never by a
#                   component or the model spec, so it cannot alter results.
#   tools/       -- the launch machinery itself (this file included). You have to be
#                   able to edit a launcher while using it.
#   constants/paths.py       -- run_main_sim._update_model_results_dir() rewrites
#                   MODEL_RESULTS_DIR here as part of launching, so the guard would
#                   otherwise reject the tree the launcher just modified. Only the
#                   results V&V notebooks read it; no component does.
#   data/lbwsg_paf/outputs/  -- run_paf_sim writes the calculated PAF parquet files
#                   here (9 tracked files). They are inputs to an *artifact build*,
#                   not to a simulation: a running sim reads PAFs from the artifact,
#                   never from this directory. Excluding them is what lets one script
#                   run the artifact workflow and then launch models.
CLEAN_TREE_EXCLUSIONS = [
    ":!validation",
    ":!tools",
    ":!constants/paths.py",
    ":!data/lbwsg_paf/outputs",
]

# The installed package directory. Git commands that ask about *this code* run
# here rather than in the current working directory: the working directory is
# whatever the operator happened to cd into, and when an environment resolves to
# a different checkout than the one you are standing in, cwd describes a tree
# that contributes nothing to the run.
PACKAGE_DIR = Path(__file__).resolve().parent.parent


def _git_status(paths: List[str]) -> Optional[str]:
    """Return uncommitted changes to *paths*, or ``None`` outside a checkout.

    Always asks about :data:`PACKAGE_DIR`, not the current working directory: the
    question these guards exist to answer is whether *the code that will run* is
    committed, and the directory the operator happens to be standing in has no
    bearing on that. An empty string means a checkout with nothing changed.
    """
    result = subprocess.run(
        [
            "git",
            "-C",
            str(PACKAGE_DIR),
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            *paths,
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def check_clean_tree() -> None:
    """Abort if there are uncommitted changes to tracked files that affect results.

    Scoped to ``src/vivarium_gates_mncnh`` minus :data:`CLEAN_TREE_EXCLUSIONS`; see
    that constant for why each path is excluded.
    """
    changed = _git_status([".", *CLEAN_TREE_EXCLUSIONS])
    if changed is None:
        # Not a checkout at all -- a built install, which cannot carry
        # uncommitted changes. There is nothing for this guard to protect.
        print(f"\n{PACKAGE_DIR} is not a git checkout; skipping the clean-tree check.")
        return
    if changed:
        excluded = ", ".join(e.removeprefix(":!") for e in CLEAN_TREE_EXCLUSIONS)
        raise RuntimeError(
            "There are uncommitted changes to tracked files in src/vivarium_gates_mncnh "
            f"(excluding {excluded}). "
            "Please commit or stash them before running this script.\n"
            f"{changed}"
        )


ENV_TYPES = ("simulation", "artifact")
CANONICAL_ENV_NAMES = {t: f"vivarium_gates_mncnh_{t}" for t in ENV_TYPES}
_SCM_REVISION = re.compile(r"\+g([0-9a-f]+)")


def _lazy_env_resolver():
    """Return vivarium-cluster-tools' env resolvers, or ``(None, None)``.

    Imported lazily and defensively: ``vivarium.cluster_tools.core.jobmon``
    imports the jobmon client at package-import time, and this module is imported
    by ``check_working_tree.py`` (workflow step 0), which must remain importable in
    an environment installed without the ``[cluster]`` extra. Venv support requires
    vivarium-cluster-tools >= 4.5.0.
    """
    try:
        from vivarium.cluster_tools.core.jobmon.env import (
            resolve_env_bin_path,
            resolve_env_prefix,
        )

        return resolve_env_prefix, resolve_env_bin_path
    except Exception:
        return None, None


def resolve_env(env: Optional[str]) -> Optional[str]:
    """Resolve an environment target to an absolute prefix.

    Parameters
    ----------
    env
        A conda environment name, a venv name (matched under ``./.venv/``), or a
        path to either prefix. ``None`` means the currently-active environment.

    Returns
    -------
    str | None
        The absolute environment prefix, or ``None`` for the active environment
        and for names that cannot be resolved (in which case callers fall back to
        ``conda run -n <name>``).
    """
    if env is None:
        return None
    resolve_env_prefix, _ = _lazy_env_resolver()
    if resolve_env_prefix is None:
        return None
    try:
        return resolve_env_prefix(env)
    except Exception:
        return None


def default_env_for_type(env_type: str) -> Optional[str]:
    """Best guess at the environment of *env_type* to use, or ``None`` for active.

    The simulation and artifact environments hold incompatible dependency sets and
    cannot be merged into one. Any script that both builds artifact data and runs
    simulations must therefore dispatch each command to the right environment, and
    can never simply inherit the caller's -- it is guaranteed to be in the wrong one
    for half of what it does. This resolves the *other* environment from the one in
    hand, which is the common case for such a script.

    Keyed off ``sys.prefix`` rather than ``$VIRTUAL_ENV``: dagger runs workflow
    steps by prepending an environment's ``bin/`` to ``PATH`` and sets neither
    ``VIRTUAL_ENV`` nor ``CONDA_DEFAULT_ENV``, but ``sys.prefix`` is always correct.

    Resolution order: the sibling overlay of the active environment (e.g. from
    ``.venv/<name>_simulation`` to ``.venv/<name>_artifact``), then an overlay
    named for this checkout, then the canonical shared conda env name.
    """
    if env_type not in ENV_TYPES:
        raise ValueError(
            f"Unknown environment type '{env_type}'. Expected one of {ENV_TYPES}."
        )

    prefix = Path(sys.prefix)
    if prefix.parent.name == ".venv":
        stem = prefix.name
        for suffix in ENV_TYPES:
            stem = stem.removesuffix(f"_{suffix}")
        sibling = prefix.parent / f"{stem}_{env_type}"
        if sibling == prefix:
            return None
        if sibling.is_dir():
            return str(sibling)

    local = Path.cwd() / ".venv" / f"{Path.cwd().name}_{env_type}"
    if local.is_dir():
        return str(local)

    return CANONICAL_ENV_NAMES[env_type]


def _reference_revision() -> Optional[str]:
    """Return the revision of the code performing this check.

    Resolved from this module's own location rather than the current working
    directory: the question is what revision *this* code is, and cwd has nothing
    to do with that. Running from outside a checkout used to leave the reference
    unknown, which made the revision check abstain -- in exactly the situation
    where an environment name is most likely to resolve somewhere unintended.

    Uses the same rule as the environments being checked -- see
    :func:`_revision_of`.
    """
    try:
        import importlib.metadata as importlib_metadata

        version = importlib_metadata.version("vivarium-gates-mncnh")
    except Exception:
        version = ""
    revision, _, _ = _revision_of(str(PACKAGE_DIR), version)
    return revision


def _checkout_revision(package_dir: str) -> Optional[tuple[str, bool]]:
    """Return ``(HEAD, dirty)`` of the checkout *package_dir* is tracked in.

    Only reports a revision when the package directory is *tracked* by that
    repository, which distinguishes an editable install pointing at a working tree
    (tracked -- the tree is the source of truth) from a built copy that merely
    happens to sit inside one, such as a venv's site-packages (untracked -- the
    install metadata is the source of truth).

    This matters because an editable install's recorded version is frozen at
    install time: the working tree it points at can be switched to another branch,
    or edited, long afterwards, and the metadata will not have moved with it.
    """
    tracked = subprocess.run(
        ["git", "-C", package_dir, "ls-files", "--error-unmatch", "__init__.py"],
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        return None
    head = subprocess.run(
        ["git", "-C", package_dir, "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if head.returncode != 0:
        return None
    status = subprocess.run(
        [
            "git",
            "-C",
            package_dir,
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            ".",
        ],
        capture_output=True,
        text=True,
    )
    return head.stdout.strip(), bool(status.stdout.strip())


def _revision_of(package_dir: str, version: str) -> tuple[Optional[str], str, bool]:
    """Return ``(revision, source, dirty)`` for the package at *package_dir*.

    The one place that decides what an install's revision *is*, used for both the
    environments being checked and the code doing the checking -- they are the same
    question and must not be allowed to answer it differently.

    An editable install runs its working tree, so that tree's current HEAD is
    authoritative and the recorded version may be stale: it is frozen at install
    time while the tree can be rebased or edited afterwards. A built install has no
    tree behind it, so the commit setuptools-scm embedded in its version is all
    there is. Whether the package directory is *tracked* tells the two apart.
    """
    checkout = _checkout_revision(package_dir)
    if checkout is not None:
        revision, dirty = checkout
        return revision, "working tree", dirty
    match = _SCM_REVISION.search(version)
    return (match.group(1) if match else None), "installed build", False


def _package_provenance(prefix: Optional[str]) -> tuple[str, str]:
    """Return ``(version, package_dir)`` for vivarium_gates_mncnh as *prefix* sees it."""
    code = (
        "import importlib.metadata as md, pathlib, vivarium_gates_mncnh as p; "
        "print(md.version('vivarium-gates-mncnh')); "
        "print(pathlib.Path(p.__file__).resolve().parent)"
    )
    if prefix is None:
        executable = sys.executable
    else:
        executable = str(Path(prefix) / "bin" / "python")
        if not Path(executable).exists():
            raise RuntimeError(f"No python interpreter at {executable}.")
    result = subprocess.run([executable, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not import vivarium_gates_mncnh using {executable}.\n"
            f"  {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ''}"
        )
    version, package_dir = result.stdout.strip().splitlines()[:2]
    return version, package_dir


def warn_if_dirty(paths: List[str], written_by: str) -> bool:
    """Print a warning listing any *paths* with uncommitted changes.

    These paths are excluded from :func:`check_clean_tree` so that run tooling can
    write them mid-workflow (see :data:`CLEAN_TREE_EXCLUSIONS`). That exclusion is
    what makes the composition possible, but it also means the changes are easy to
    miss, so say so loudly instead of failing.

    Returns
    -------
    bool
        True if any of *paths* has uncommitted changes.
    """
    changed = _git_status(paths)
    if not changed:
        return False
    print("\n" + "!" * 80)
    print(
        f"WARNING: {written_by} left uncommitted changes to tracked files.\n"
        "         They are excluded from the clean-tree check so this script could\n"
        "         write them, but they are yours to commit -- a run is only\n"
        "         reproducible once they are.\n"
        f"{changed}"
    )
    print("!" * 80 + "\n")
    return True


def check_environment(
    *envs: Optional[str],
    allow_env_mismatch: bool = False,
) -> None:
    """Check that every environment a script will dispatch to runs *this* revision.

    For each target, resolves the environment, confirms ``vivarium_gates_mncnh`` is
    importable there, and compares the commit embedded in its installed version
    (setuptools-scm's ``+g<sha>`` local segment) against ``HEAD``.

    This is a provenance check, not a path-containment check. It is satisfied by an
    editable venv overlay on this checkout, and equally by a non-editable install
    built from this commit (as Jenkins produces), while still catching a shared
    environment that resolves to some other checkout or an older revision.

    Parameters
    ----------
    envs
        The environment targets to validate -- conda env names, venv names, or
        prefix paths. ``None`` means the currently-active environment. Every target
        a script will actually dispatch to should be passed, so that validating one
        environment never implies anything about another.
    allow_env_mismatch
        If True, an unverified revision -- a mismatch, or a revision that cannot be
        determined on either side -- is reported as a warning rather than an error.
        For deliberately running against an environment not built from this commit.
        It does *not* cover an environment that cannot be resolved or cannot import
        the package; those are always fatal, since nothing about what would run is
        known.

    Raises
    ------
    RuntimeError
        Always, if an environment cannot be resolved or cannot import the package.
        Otherwise if its revision cannot be verified against this code's and
        *allow_env_mismatch* is False.
    """
    print("\nChecking environments...")

    head = _reference_revision()
    # Two kinds of finding, because they warrant different answers. A broken
    # environment cannot run anything correctly, so it is always fatal. An
    # unverified revision may be a deliberate choice, so it is what
    # allow_env_mismatch covers -- and nothing else, or the flag becomes a way to
    # wave through an environment nobody can account for.
    broken = []
    unverified = []

    for env in envs or (None,):
        label = env if env is not None else "active environment"
        prefix = resolve_env(env)

        if env is not None and prefix is None:
            # Unresolvable name: it may still be a conda env that `conda run` finds,
            # but we cannot inspect it, so we cannot vouch for what it will run.
            broken.append(
                f"{label}: could not be resolved to an environment. Pass a path to "
                "its directory, or check the name exists."
            )
            continue

        try:
            version, package_dir = _package_provenance(prefix)
        except RuntimeError as e:
            broken.append(f"{label}: {e}")
            continue

        print(f"  {label}")
        print(f"    vivarium_gates_mncnh {version}")
        print(f"    {package_dir}")

        revision, source, dirty = _revision_of(package_dir, version)
        if dirty:
            print("    ! that working tree has uncommitted changes")

        if head is None:
            unverified.append(
                f"{label}: cannot determine the revision of the code running this "
                "check, so there is nothing to compare against."
            )
        elif revision is None:
            unverified.append(
                f"{label}: its {source} carries no revision, so it cannot be compared "
                f"against {head[:9]}."
            )
        elif not (head.startswith(revision) or revision.startswith(head)):
            unverified.append(
                f"{label}: {source} is at revision {revision[:9]}, not {head[:9]}. It "
                "would run different code than this checkout. Rebuild that environment "
                "(e.g. 'source environment.sh -s') or point at the right one."
            )
        else:
            print(f"    \u2713 {source} matches {head[:9]}")

    if broken:
        raise RuntimeError(
            "Environment unusable:\n  - "
            + "\n  - ".join(broken)
            + "\n(--allow-env-mismatch does not cover this: the environment cannot run "
            "this package at all.)"
        )

    if unverified:
        message = "Environment revision not verified:\n  - " + "\n  - ".join(unverified)
        if allow_env_mismatch:
            print(f"\nWARNING: {message}\n")
        else:
            raise RuntimeError(message + "\nPass --allow-env-mismatch to run anyway.")

    print()


def run_command(
    cmd: List[str],
    description: str,
    env: Optional[str] = None,
    auto_confirm: bool = False,
    capture_full_output: bool = False,
    log_prefix: str = "",
) -> str | None:
    """
    Run a shell command and handle errors.

    Parameters
    ----------
    cmd : List[str]
        Command and arguments to execute
    description : str
        Description of what the command does (for error messages)
    env : str, optional
        The environment to run the command in: a conda env name, a venv name, or
        a path to either one's directory (its "prefix" -- the root holding bin/). If None, the command runs in the
        currently-active environment. A target that resolves to a prefix is
        dispatched by prepending its ``bin/`` to ``PATH`` -- the same mechanism
        dagger uses -- which works for venv overlays as well as conda envs. A name
        that cannot be resolved falls back to ``conda run -n <name>``
    auto_confirm : bool, optional
        If True, automatically answer 'y' to any prompts (useful for make_artifacts)
    capture_full_output : bool, optional
        If True, capture and return the full command output as a string
    log_prefix : str, optional
        String to prepend to every line printed by this command. Useful when
        several commands are run concurrently and their output interleaves

    Returns
    -------
    str | None
        The full output if capture_full_output is True, otherwise None
    """

    def emit(line: str, end: str = "\n") -> None:
        print(f"{log_prefix}{line}", end=end)

    emit(f"\n{'='*80}")
    emit(f"Running: {description}")
    emit(f"Environment: {env if env else 'active environment'}")

    # Build the command. Dispatch by PATH prefix where the target resolves to an
    # environment prefix -- this is what dagger does, and unlike `conda run` it
    # works for the venv overlays built by `make build-shared-env`. Fall back to
    # `conda run -n <name>` only for a name we could not resolve.
    # An unspecified target means "the environment this process is running in" --
    # which is sys.prefix, NOT the inherited PATH. Running `.venv/bin/python -m ...`
    # without activating the venv leaves its bin/ off PATH entirely, so relying on
    # inheritance silently resolves entry points from whatever conda base happens to
    # be first on PATH.
    env_vars = None
    prefix = sys.prefix if env is None else resolve_env(env)
    if prefix is not None:
        _, resolve_env_bin_path = _lazy_env_resolver()
        bin_path = (
            resolve_env_bin_path(prefix)
            if resolve_env_bin_path is not None
            else str(Path(prefix) / "bin")
        )
        env_vars = {**os.environ, "PATH": f"{bin_path}:{os.environ.get('PATH', '')}"}
        emit(f"Prefix: {prefix}")
    elif env is not None:
        cmd = ["conda", "run", "--no-capture-output", "-n", env] + cmd

    emit(f"Command: {' '.join(cmd)}")
    if auto_confirm:
        emit("Auto-confirm: y (continuous)")

    emit(f"{'='*80}\n")

    full_output = []

    # Continuous 'y' for prompts, fed from a real `yes` process rather than a
    # shell pipeline: the old `yes y | <joined cmd>` form ran with shell=True on a
    # space-joined string, so any argument or environment path containing a space
    # or shell metacharacter was silently re-split into separate arguments.
    yes_process = (
        subprocess.Popen(["yes", "y"], stdout=subprocess.PIPE) if auto_confirm else None
    )
    stdin = yes_process.stdout if yes_process else None

    try:
        if capture_full_output:
            # Use Popen to capture output in real-time.
            # Start in a new process group so we can kill the entire tree on interrupt.
            process = subprocess.Popen(
                cmd,
                stdin=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                start_new_session=True,
                env=env_vars,
            )

            try:
                # Read output line by line
                for line in process.stdout:
                    # Print the line to maintain visibility
                    emit(line, end="")
                    full_output.append(line)

                # Wait for process to complete
                return_code = process.wait()
            except KeyboardInterrupt:
                emit(f"\nInterrupted. Terminating {description}...")
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                raise

            if return_code != 0:
                raise RuntimeError(f"Failed {description}. Exit code: {return_code}")

            return "".join(full_output)
        else:
            # Print output to screen
            try:
                result = subprocess.run(
                    cmd, stdin=stdin, start_new_session=True, env=env_vars
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Failed {description}. Exit code: {result.returncode}"
                    )
            except KeyboardInterrupt:
                emit(f"\nInterrupted. Terminating {description}...")
                raise
    finally:
        if yes_process is not None:
            yes_process.stdout.close()
            yes_process.terminate()
            yes_process.wait()


def check_psimulate_finished(psimulate_output: str) -> bool:
    """
    Check whether psimulate finished properly by parsing its output.

    Parses the second-to-last line of psimulate output which should be:
    "(M of N total jobs completed successfully overall)"

    Parameters
    ----------
    psimulate_output : str
        The full output from the psimulate command

    Returns
    -------
    bool
        True if M equals N (all jobs completed successfully), False otherwise
    """
    print("\nChecking psimulate completion status...")

    # Split output into lines and get the last non-empty lines
    lines = [line.strip() for line in psimulate_output.strip().split("\n") if line.strip()]

    # Parse the second-to-last line for job completion status
    # Expected format: "(M of N total jobs completed successfully overall)"
    completion_line = lines[-2]
    completion_pattern = re.compile(
        r"\((\d+) of (\d+) total jobs completed successfully overall\)"
    )
    completion_match = completion_pattern.search(completion_line)

    if completion_match:
        completed = int(completion_match.group(1))
        total = int(completion_match.group(2))
        print(f"Job completion: {completed} of {total}")

        if completed == total:
            print("✓ All jobs completed successfully")
            return True
        else:
            print(f"✗ Only {completed} of {total} jobs completed successfully")
            return False
    else:
        print(f"WARNING: Could not parse completion status from: {completion_line}")
        return False


def extract_results_dir(psimulate_output: str) -> Optional[str]:
    """
    Extract the results directory from psimulate output.

    Parses the last line of psimulate output which should be:
    "Results written to: {results_dir}"

    Parameters
    ----------
    psimulate_output : str
        The full output from the psimulate command

    Returns
    -------
    Optional[str]
        The results directory path, or None if not found
    """
    # Split output into lines and get the last non-empty lines
    lines = [line.strip() for line in psimulate_output.strip().split("\n") if line.strip()]

    # Parse the last line for results directory
    # Expected format: "Results written to: {results_dir}"
    results_line = lines[-1]
    results_pattern = re.compile(r"Results written to:\s*(.+)")
    results_match = results_pattern.search(results_line)

    if results_match:
        results_dir = results_match.group(1).strip()
        # Remove ANSI escape codes (color codes) from the path
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        results_dir = ansi_escape.sub("", results_dir)
        print(f"Results directory: {results_dir}")
        return results_dir
    else:
        print(f"WARNING: Could not parse results directory from: {results_line}")
        return None


def move_results(source_pattern: str, dest_dir: str, description: str) -> None:
    """
    Move result files to the destination directory.

    Handles two output formats:
    1. Flat parquet files matching the pattern (e.g., {measure}.parquet)
    2. Directories matching the pattern, each containing parquet file(s)

    In both cases, the destination file is named {measure_name}.parquet.

    Parameters
    ----------
    source_pattern : str
        Source directory pattern (can include wildcards)
    dest_dir : str
        Destination directory
    description : str
        Description of the files being moved
    """
    print(f"\nMoving {description}")
    print(f"From: {source_pattern}")
    print(f"To: {dest_dir}")

    # Create destination directory if it doesn't exist
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    # Expand wildcards using glob
    matching_paths = glob.glob(source_pattern)

    if not matching_paths:
        raise RuntimeError(
            f"Failed to move {description}. No paths matched pattern: {source_pattern}"
        )

    # Move files from matching paths
    moved_count = 0
    for source_path in matching_paths:
        source_path_obj = Path(source_path)

        try:
            if source_path_obj.is_file() and source_path_obj.suffix == ".parquet":
                # Flat parquet file: move directly
                dest_file = Path(dest_dir) / source_path_obj.name
                shutil.move(str(source_path_obj), dest_file)
                moved_count += 1
            elif source_path_obj.is_dir():
                # Directory: find parquet files inside
                parquet_files = sorted(source_path_obj.glob("*.parquet"))
                if not parquet_files:
                    raise RuntimeError(
                        f"No parquet files found in directory {source_path}. "
                        f"Contents: {[f.name for f in source_path_obj.iterdir()]}"
                    )
                elif len(parquet_files) == 1:
                    # Single parquet file: move and rename to {directory_name}.parquet
                    dest_filename = f"{source_path_obj.name}.parquet"
                    dest_file = Path(dest_dir) / dest_filename
                    shutil.move(str(parquet_files[0]), dest_file)
                    moved_count += 1
                else:
                    # Multiple shards: concatenate into single file
                    import pandas as pd

                    dfs = [pd.read_parquet(f) for f in parquet_files]
                    combined = pd.concat(dfs, ignore_index=True)
                    dest_filename = f"{source_path_obj.name}.parquet"
                    dest_file = Path(dest_dir) / dest_filename
                    combined.to_parquet(dest_file, index=False)
                    moved_count += 1
            else:
                raise RuntimeError(
                    f"Unexpected path type for {source_path}: not a .parquet file or directory"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to move from {source_path} to {dest_dir}. Error: {e}")

    print(f"Moved {moved_count} file(s) successfully\n")
