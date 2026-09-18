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


def _env_resolvers():
    """Return vivarium-cluster-tools' ``(prefix, bin_path)`` env resolvers.

    Every environment decision -- which prefix a target names, which ``bin/``
    directories go on PATH -- depends on these, and guessing at either is how a run
    ends up in the wrong environment: ``<prefix>/bin`` alone drops a venv overlay's
    base conda env, where the console scripts live, so psimulate silently resolves
    to whatever conda base is first on PATH. So a missing resolver raises here with
    the real cause rather than being worked around downstream.

    The import is deferred to call time because reaching these costs far more than
    they do. ``env.py`` itself imports nothing heavy, but importing it runs its
    parent package's ``__init__``, which reaches the real jobmon client:
    ``core/jobmon/__init__`` -> ``artifact`` -> ``client`` -> ``jobmon.client.api``.
    This module is imported by ``check_working_tree.py`` (workflow step 0), whose
    entire job is three git calls; a guard that protects the pipeline should not
    pay for jobmon, nor be stopped by one that is merely slow or misconfigured --
    hence also the broad ``except``, which covers an installed-but-broken jobmon
    and not only an absent one.

    Note the deferral is *not* about supporting an environment without the
    ``[cluster]`` extra. Every environment the Makefile builds has it (simulation
    via ``ENV_REQS=dev``, artifact via ``ENV_REQS=data``, and overlays inherit from
    those), so the absent case is a bare ``pip install .`` that nothing here
    produces. Venv support requires vivarium-cluster-tools >= 4.5.0.

    Raises
    ------
    RuntimeError
        If the resolvers cannot be imported. Importing *this* module stays possible
        without the cluster stack; only resolving or dispatching to an environment
        needs them.
    """
    try:
        from vivarium.cluster_tools.core.jobmon.env import (
            resolve_env_bin_path,
            resolve_env_prefix,
        )
    except Exception as e:
        raise RuntimeError(
            "vivarium-cluster-tools >= 4.5.0 could not be imported, and it provides "
            "the environment resolver this tooling dispatches with.\n"
            "This environment is missing the '[cluster]' extra, has too old a "
            "vivarium-cluster-tools, or has a jobmon installation that fails to "
            f"import. psimulate would not be available here either. ({e})\n"
            "Install it with 'pip install -e \".[cluster]\"', or rebuild the "
            "environment with 'source environment.sh -s'."
        )
    return resolve_env_prefix, resolve_env_bin_path


def resolve_env(env: Optional[str]) -> Optional[str]:
    """Resolve an environment target to an absolute prefix.

    Parameters
    ----------
    env
        A conda environment name, a venv name, or a path to either prefix.
        ``None`` means the currently-active environment. Prefer a path: upstream
        matches a venv name under ``Path.cwd()/.venv/<name>``, so a bare name
        depends on where it is resolved from, and a conda name resolves per-user.

    Returns
    -------
    str | None
        The absolute environment prefix, or ``None`` for the active environment.

    Raises
    ------
    RuntimeError
        If the resolvers are unavailable, or *env* names no environment we can
        inspect. An environment nobody can inspect is not one to dispatch to: it
        is exactly the case where a run would silently execute unknown code.
    """
    if env is None:
        return None
    resolve_env_prefix, _ = _env_resolvers()
    try:
        return resolve_env_prefix(env)
    except RuntimeError:
        # Upstream's own failure: it names the target and says precisely what it
        # looked for and where, so pass it through unembellished.
        raise
    except Exception as e:
        # Anything else is incidental rather than a verdict on the target -- it
        # resolves conda names by subprocess, so a missing conda arrives as an
        # OSError naming only `conda`. Label it, and normalise the type so
        # check_environment can collect it alongside the rest.
        raise RuntimeError(f"{env}: {e}")


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
    ``.venv/<name>_simulation`` to ``.venv/<name>_artifact``), then the canonical
    conda env name, which is what a setup not using overlays has. Falling back
    from an overlay to a name prints how to build the missing sibling, since that
    reason is not recoverable downstream. Deliberately nothing keyed off the
    current working directory: cwd says nothing about which environment the caller
    is in, and everything else here is careful not to ask it. Pass ``--env`` for
    anything these two do not cover.
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
        # An overlay user falling back to a conda name is rarely what they meant,
        # and this is the only place that knows why it happened. Do not raise:
        # someone may legitimately have both setups, and the canonical env may
        # well exist. But say the actionable thing now, because by the time
        # check_environment reports it, all that is left is "could not be
        # resolved", which does not mention the missing sibling.
        print(
            f"\nNOTE: running in the venv overlay {prefix},\n"
            f"      but there is no sibling {env_type} overlay at {sibling}.\n"
            f"      Falling back to the conda env '{CANONICAL_ENV_NAMES[env_type]}'.\n"
            f"      To build the overlay instead: make build-shared-env type={env_type}\n"
        )

    return CANONICAL_ENV_NAMES[env_type]


def _revision_of(package_dir: str, installed_version: str) -> tuple[Optional[str], str, bool]:
    """Return ``(revision, revision_source, dirty)`` for the package at *package_dir*.

    The one place that decides what an install's revision *is*, used for both the
    environments being checked and the code doing the checking -- they are the same
    question and must not be allowed to answer it differently.

    An editable install runs its working tree, so that tree's current HEAD is
    authoritative and the recorded version may be stale: it is frozen at install
    time while the tree can be rebased or edited afterwards. A built install has no
    tree behind it, so the commit setuptools-scm embedded in its version is all
    there is.

    Whether the package directory is *tracked* tells the two apart: an editable
    install points at a working tree (tracked), while a built copy merely sits in
    a directory, such as a venv's site-packages, that may happen to be inside some
    checkout (untracked).

    ``dirty`` is reported for display only. It means *any* uncommitted change under
    *package_dir*, including the paths :data:`CLEAN_TREE_EXCLUSIONS` deliberately
    permits, so it must not be treated as a failure -- that would reject runs
    :func:`check_clean_tree` is designed to allow.
    """

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", package_dir, *args], capture_output=True, text=True
        )

    runs_from_working_tree = git("ls-files", "--error-unmatch", "__init__.py").returncode == 0
    tree_head = git("rev-parse", "HEAD") if runs_from_working_tree else None
    if tree_head is not None and tree_head.returncode == 0:
        uncommitted = git("status", "--porcelain", "--untracked-files=no", "--", ".")
        return tree_head.stdout.strip(), "working tree", bool(uncommitted.stdout.strip())

    revision_in_version = _SCM_REVISION.search(installed_version)
    return (
        (revision_in_version.group(1) if revision_in_version else None),
        "installed build",
        False,
    )


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
    installed_version, package_dir = result.stdout.strip().splitlines()[:2]
    return installed_version, package_dir


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

    # Once, not per environment: a missing resolver is a fact about this process,
    # not a finding about any target.
    _env_resolvers()

    # The reference side: what revision is the code doing the checking? Resolved
    # from PACKAGE_DIR, not the cwd, and through the same _revision_of rule as the
    # targets -- it is the same question and must not get a different answer.
    try:
        import importlib.metadata as importlib_metadata

        our_version = importlib_metadata.version("vivarium-gates-mncnh")
    except Exception:
        our_version = ""
    reference_revision, _, _ = _revision_of(str(PACKAGE_DIR), our_version)

    # Two kinds of finding, because they warrant different answers. A broken
    # environment cannot run anything correctly, so it is always fatal. An
    # unverified revision may be a deliberate choice, so it is what
    # allow_env_mismatch covers -- and nothing else, or the flag becomes a way to
    # wave through an environment nobody can account for.
    broken = []
    unverified = []

    for env in envs or (None,):
        label = env if env is not None else "active environment"

        # Collected rather than raised, so every target is reported in one go.
        try:
            prefix = resolve_env(env)
        except RuntimeError as e:
            broken.append(str(e))  # already names the target
            continue
        try:
            installed_version, package_dir = _package_provenance(prefix)
        except RuntimeError as e:
            broken.append(f"{label}: {e}")
            continue

        print(f"  {label}")
        print(f"    vivarium_gates_mncnh {installed_version}")
        print(f"    {package_dir}")

        target_revision, revision_source, dirty = _revision_of(package_dir, installed_version)
        if dirty:
            print("    ! that working tree has uncommitted changes")

        if reference_revision is None:
            unverified.append(
                f"{label}: cannot determine the revision of the code running this "
                "check, so there is nothing to compare against."
            )
        elif target_revision is None:
            unverified.append(
                f"{label}: its {revision_source} carries no revision, so it cannot be "
                f"compared against {reference_revision[:9]}."
            )
        elif not (
            reference_revision.startswith(target_revision)
            or target_revision.startswith(reference_revision)
        ):
            unverified.append(
                f"{label}: {revision_source} is at revision {target_revision[:9]}, not "
                f"{reference_revision[:9]}. It would run different code than this "
                "checkout. Rebuild that environment (e.g. 'source environment.sh -s') "
                "or point at the right one."
            )
        else:
            print(f"    \u2713 {revision_source} matches {reference_revision[:9]}")

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
        currently-active environment. The target is dispatched by prepending its
        ``bin/`` directories to ``PATH`` -- the same mechanism dagger uses -- which
        works for venv overlays as well as conda envs
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

    Raises
    ------
    RuntimeError
        If the environment cannot be resolved -- including because
        vivarium-cluster-tools is not importable -- or if the command exits
        non-zero.
    """

    def emit(line: str, end: str = "\n") -> None:
        print(f"{log_prefix}{line}", end=end)

    emit(f"\n{'='*80}")
    emit(f"Running: {description}")
    emit(f"Environment: {env if env else 'active environment'}")

    # Dispatch by prepending the environment's bin/ directories to PATH -- this is
    # what dagger does, and unlike `conda run` it works for the venv overlays built
    # by `make build-shared-env` as well as for conda envs, so it is the only
    # mechanism here. An unresolvable target raises rather than falling back to
    # something that would run in an environment nobody could inspect.
    # An unspecified target means "the environment this process is running in" --
    # which is sys.prefix, NOT the inherited PATH. Running `.venv/bin/python -m ...`
    # without activating the venv leaves its bin/ off PATH entirely, so relying on
    # inheritance silently resolves entry points from whatever conda base happens to
    # be first on PATH.
    _, resolve_env_bin_path = _env_resolvers()
    prefix = sys.prefix if env is None else resolve_env(env)
    bin_path = resolve_env_bin_path(prefix)
    env_vars = {**os.environ, "PATH": f"{bin_path}:{os.environ.get('PATH', '')}"}

    emit(f"Prefix: {prefix}")
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
