"""
Utility functions for PAF simulation workflow.
"""

import glob
import importlib
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Literal, Optional, Tuple

#: The two environment flavours this repository builds. Their dependency sets
#: conflict and cannot be merged into one environment.
EnvType = Literal["simulation", "artifact"]
ENV_TYPES: Tuple[EnvType, ...] = ("simulation", "artifact")


#: The distribution name. Deliberately *not* derived from the checkout
#: directory: a Jenkins PR workspace is named e.g.
#: ``vivarium_gates_mncnh_PR-327-head@2``, and anything that must match an
#: artifact published outside this checkout has to use the distribution name.
DIST_NAME = "vivarium_gates_mncnh"

#: The installed package's directory, i.e. ``<repo>/src/vivarium_gates_mncnh``.
#: This is how :func:`repo_root` finds the checkout, because every supported
#: environment installs the package editable and so the package directory lives
#: inside it.
PACKAGE_DIR = Path(__file__).resolve().parent.parent


def _environment_sh_hint(env_type: Optional[EnvType] = None) -> str:
    """Return the ``source environment.sh`` invocation that builds *env_type*."""
    flags = "-s" if env_type in (None, "simulation") else f"-s -t {env_type}"
    return f"source environment.sh {flags}"


def repo_root() -> Path:
    """Return the absolute path to the repository root, independent of cwd.

    Every git helper in this module asks git about *this* path rather than the
    current working directory. The directory an operator happens to be standing
    in has no bearing on which code is about to run, and a relative pathspec
    resolved against the wrong cwd makes a guard pass vacuously rather than
    fail -- a silent hole in exactly the check that is supposed to protect a
    run's provenance.

    Resolution order:

    1. The installed package's location. Every supported environment installs
       this package editable, so the package directory lives inside the
       checkout.
    2. ``git rev-parse --show-toplevel`` from the current working directory.

    Returns
    -------
    Path
        The directory containing ``.git``.

    Raises
    ------
    RuntimeError
        If neither strategy yields a directory containing ``.git``. This raises
        rather than returning a best guess on purpose: a wrong root would make
        :func:`check_clean_tree` inspect the wrong repository and pass
        vacuously.
    """
    searched: List[Path] = []

    # 1. The installed package's location. Every supported environment
    #    installs this package editable, so the package directory lives inside
    #    the checkout.
    package_dir = Path(PACKAGE_DIR)
    for candidate in (package_dir, *package_dir.parents):
        searched.append(candidate)
        # ``.exists()``, not ``.is_dir()``: in a git worktree -- which is how
        # this repository is often checked out -- ``.git`` is a file.
        if (candidate / ".git").exists():
            return candidate

    # 2. Ask git about the current working directory.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
        )
    except FileNotFoundError:
        result = None
    if result is not None and result.returncode == 0 and result.stdout.strip():
        root = Path(result.stdout.strip()).resolve()
        searched.append(root)
        if (root / ".git").exists():
            return root

    raise RuntimeError(
        f"Could not locate the repository root. Neither {PACKAGE_DIR} nor "
        f"'git rev-parse --show-toplevel' from "
        f"{Path.cwd()} pointed at a directory containing '.git'.\n"
        "  Searched: " + ", ".join(str(path) for path in searched) + "\n"
        "Install the package editable into the active environment -- "
        f"'{_environment_sh_hint()}' does this -- and re-run from the checkout."
    )


def _env_satisfies(env_type: EnvType) -> bool:
    """Return whether the *active* interpreter can do *env_type* work.

    Probes capability rather than inspecting a name, because a name says
    nothing about what is installed. Reuses the convention already established
    in ``tests/conftest.py``, which distinguishes the two environments by
    whether ``vivarium_inputs`` imports.

    - ``"artifact"`` -- ``vivarium_inputs`` is importable.
    - ``"simulation"`` -- ``psimulate`` is on ``PATH`` *and* ``vivarium_inputs``
      is not importable.

    ``psimulate`` alone does not discriminate: the ``data`` extra that builds
    the artifact environment pulls in ``vivarium_gates_mncnh[cluster,...]``, so
    vivarium-cluster-tools -- and therefore ``psimulate`` -- is installed in
    both. Probing for it alone would report the artifact environment as able to
    do simulation work, and :func:`sibling_env` would then run psimulate there.
    """
    try:
        importlib.import_module("vivarium_inputs")
        artifact_capable = True
    except ImportError:
        artifact_capable = False

    if env_type == "artifact":
        return artifact_capable
    if env_type == "simulation":
        return shutil.which("psimulate") is not None and not artifact_capable
    raise ValueError(f"Unknown environment type '{env_type}'. Expected one of {ENV_TYPES}.")


def require_environment(env_type: EnvType) -> None:
    """Abort unless the *active* environment can do *env_type* work.

    Nothing in this repository dispatches a command into another environment
    any more: every script runs in whatever environment it was started in, and
    the workflow runner is what decides which that is. So the only question
    worth asking is whether the environment we are already in is the right kind
    -- which is exactly what this answers.

    That is also why it is cheap. Resolving an environment by name, prefix or
    overlay, and then constructing a ``PATH`` to dispatch into it, was a
    substantial amount of machinery that existed solely to let one script
    alternate between two environments. Splitting that script into per-step
    workflow tasks removed the need for all of it.

    Parameters
    ----------
    env_type
        The flavour of work about to be done.

    Raises
    ------
    ValueError
        If *env_type* is not one of :data:`ENV_TYPES`.
    RuntimeError
        If the active environment cannot do that work. The message names the
        ``source environment.sh`` invocation that builds the right one, and
        -- when run under a workflow -- the step's ``environment`` key is the
        thing to correct.
    """
    if env_type not in ENV_TYPES:
        raise ValueError(
            f"Unknown environment type '{env_type}'. Expected one of {ENV_TYPES}."
        )

    if _env_satisfies(env_type):
        print(f"  \u2713 active environment ({sys.prefix}) can do {env_type} work")
        return

    other = "artifact" if env_type == "simulation" else "simulation"
    raise RuntimeError(
        f"This is not a {env_type} environment: {sys.prefix}\n"
        f"  It looks like a {other} environment instead.\n"
        f"  Build or activate the right one with '{_environment_sh_hint(env_type)}'.\n"
        "  If this is a workflow step, set its 'environment' key to the "
        f"{env_type} environment."
    )


def warn_if_dirty(paths: List[str], written_by: str) -> bool:
    """Print a warning listing any *paths* with uncommitted changes.

    These paths are excluded from :func:`check_clean_tree` so that run tooling
    can write them mid-workflow. That exclusion is what lets one script build
    artifact data and then launch models, but it also makes the changes easy to
    miss -- so say so loudly rather than failing.

    Returns
    -------
    bool
        True if any of *paths* has uncommitted changes.
    """
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root()),
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            *paths,
        ],
        capture_output=True,
        text=True,
    )
    changed = result.stdout.strip() if result.returncode == 0 else ""
    if not changed:
        return False
    print("\n" + "!" * 80)
    print(
        f"WARNING: {written_by} left uncommitted changes to tracked files.\n"
        "         They are excluded from the clean-tree check so this script could\n"
        "         write them, but they are yours to commit -- a run is only\n"
        f"         reproducible once they are.\n{changed}"
    )
    print("!" * 80 + "\n")
    return True


def tag_commit(tag: str) -> Optional[str]:
    """Return the commit SHA a tag points to, or ``None`` if it doesn't exist."""
    result = subprocess.run(
        ["git", "-C", str(repo_root()), "rev-list", "-n1", tag],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def head_commit() -> str:
    """Return the SHA of the current HEAD."""
    result = subprocess.run(
        ["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def commit_pending_changes(message: str) -> None:
    """Commit any uncommitted tracked-file changes with ``message`` and push to origin.

    No-op if the working tree has nothing to commit. Untracked files are ignored.
    """
    git = ["git", "-C", str(repo_root())]
    status = subprocess.run(
        git + ["status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    if not status.stdout.strip():
        print("No pending changes to commit.")
        return

    try:
        subprocess.run(git + ["add", "-u"], check=True, capture_output=True, text=True)
        subprocess.run(
            git + ["commit", "-m", message], check=True, capture_output=True, text=True
        )
        print(f"Committed pending changes: {message!r}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to commit pending changes.\n  stderr: {e.stderr.strip()}")

    try:
        subprocess.run(
            git + ["push", "origin", "HEAD"], check=True, capture_output=True, text=True
        )
        print("Pushed commit to origin.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to push commit to origin.\n  stderr: {e.stderr.strip()}")


def create_and_push_tag(model_number: str) -> None:
    """Create a git tag ``v{model_number}`` for the current HEAD and push it to origin.

    If the tag already exists on the current commit, it is left as-is. If it
    exists on a *different* commit the user is prompted to force-update;
    when stdin is not a TTY (e.g. running inside a jobmon task) the prompt
    aborts cleanly instead of crashing.
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

    print(f"\n{'Updating' if force else 'Creating'} git tag '{tag}' and pushing to origin...")

    git = ["git", "-C", str(repo_root())]
    try:
        cmd = git + (["tag", "-f", tag] if force else ["tag", tag])
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  {'Updated' if force else 'Created'} tag '{tag}'")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to create git tag '{tag}'.\n  stderr: {e.stderr.strip()}")

    try:
        cmd = git + (["push", "--force", "origin", tag] if force else ["push", "origin", tag])
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  Pushed tag '{tag}' to origin")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to push git tag '{tag}' to origin.\n  stderr: {e.stderr.strip()}"
        )


def check_clean_tree(repo: Optional[Path] = None) -> None:
    """Abort if tracked files that affect results have uncommitted changes.

    Scoped to ``src/vivarium_gates_mncnh``, excluding ``validation/`` (post-hoc
    analysis code, imported only by notebooks) and ``tools/`` (the launch
    machinery itself -- you have to be able to edit a launcher while using it).

    Parameters
    ----------
    repo
        The repository to inspect. Defaults to :func:`repo_root`. Present so
        the guard can be exercised against a scratch repository in tests.

    Raises
    ------
    RuntimeError
        If any in-scope tracked file has uncommitted changes, or if git cannot
        answer the question.

    Notes
    -----
    The pathspec must be resolved against *repo*, not the current working
    directory. Previously it was relative and resolved against cwd, so running
    from any subdirectory of the checkout matched nothing and the guard passed
    silently -- worse than failing, because a pipeline step whose cwd is chosen
    by the runner then had no guard at all.
    """
    repo = Path(repo) if repo is not None else repo_root()

    # ``:(top)`` anchors each pathspec at the repository root, so the guard
    # inspects the same files no matter which directory the runner chose.
    pathspecs = [
        f":(top)src/{DIST_NAME}",
        f":(top,exclude)src/{DIST_NAME}/validation",
        f":(top,exclude)src/{DIST_NAME}/tools",
    ]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no", "--"]
            + pathspecs,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "git is not available, so the working tree could not be checked. "
            "Refusing to run without a provenance guard; pass --skip-tree-check "
            "if that is genuinely what you want."
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to check the working tree in {repo}.\n"
            f"  git exited {result.returncode}\n"
            f"  stderr: {result.stderr.strip()}"
        )

    if result.stdout.strip():
        raise RuntimeError(
            f"There are uncommitted changes to tracked files in src/{DIST_NAME} "
            "(excluding validation/ and tools/). "
            "Please commit or stash them before running this script.\n"
            f"{result.stdout.strip()}"
        )

    print(f"✓ Working tree is clean in src/{DIST_NAME} (excluding validation/ and tools/).")


def _feed_confirmations(stream) -> None:
    """Answer ``y`` to every prompt the child writes, until it closes stdin.

    Answering once is not enough: a command that prompts a second time would
    block forever on an empty pipe. This is the ``yes y |`` behaviour the
    previous ``shell=True`` implementation bought at the cost of re-splitting
    every argument containing a space.
    """
    try:
        while True:
            stream.write("y\n")
            stream.flush()
    except (BrokenPipeError, OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except (BrokenPipeError, OSError, ValueError):
            pass


def run_command(
    cmd: List[str],
    description: str,
    auto_confirm: bool = False,
    capture_full_output: bool = False,
) -> Optional[str]:
    """
    Run a command in the active environment and handle errors.

    The command runs in whatever environment this process is already in. There
    is deliberately no way to target another one: scripts here no longer
    alternate environments, the workflow runner selects one per step, and the
    active environment's editable install is the code under test. Use
    :func:`require_environment` to assert it is the right kind before running
    anything.

    Parameters
    ----------
    cmd : List[str]
        Command and arguments to execute
    description : str
        Description of what the command does (for error messages)
    auto_confirm : bool, optional
        If True, answer 'y' to any prompts (useful for make_artifacts)
    capture_full_output : bool, optional
        If True, capture and return the full command output as a string

    Returns
    -------
    str | None
        The full output if capture_full_output is True, otherwise None

    Raises
    ------
    RuntimeError
        If the executable is not found, or if the command exits non-zero.

    Notes
    -----
    Both the captured and uncaptured paths run the same *argv*. The previous
    implementation built a ``yes y | conda run ... <space-joined cmd>`` string
    for ``auto_confirm`` and ran it with ``shell=True``, which re-split any
    argument containing a space, and -- when combined with
    ``capture_full_output`` -- silently passed the *unwrapped* command to
    ``Popen`` instead, dropping both the environment and the auto-confirm.
    Removing ``conda run`` removed the reason that hack existed.

    ``auto_confirm`` writes ``y\\n`` to the child's stdin repeatedly until the
    pipe closes, not once: a command that prompts more than once would
    otherwise hang.
    """
    argv = list(cmd)

    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Environment: {sys.prefix}")
    print(f"Command: {' '.join(argv)}")
    if auto_confirm:
        print("Auto-confirm: y (continuous)")
    print(f"{'='*80}\n")

    try:
        # Start in a new process group so an interrupt can kill the whole tree.
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE if auto_confirm else None,
            stdout=subprocess.PIPE if capture_full_output else None,
            stderr=subprocess.STDOUT if capture_full_output else None,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Failed {description}: '{argv[0]}' was not found.\n"
            f"Build the environment that provides it with "
            f"'{_environment_sh_hint()}' (add '-t artifact' for artifact tooling)."
        )

    if auto_confirm:
        # Keep answering until the child closes the pipe. A single 'y' would
        # hang a command that prompts more than once.
        confirm_thread = threading.Thread(
            target=_feed_confirmations, args=(process.stdin,), daemon=True
        )
        confirm_thread.start()

    full_output: List[str] = []
    try:
        if capture_full_output:
            for line in process.stdout:
                print(line, end="")
                full_output.append(line)
        return_code = process.wait()
    except KeyboardInterrupt:
        print(f"\nInterrupted. Terminating {description}...")
        os.killpg(process.pid, signal.SIGTERM)
        process.wait()
        raise

    if return_code != 0:
        raise RuntimeError(f"Failed {description}. Exit code: {return_code}")

    return "".join(full_output) if capture_full_output else None


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
    if len(lines) < 2:
        print(
            "WARNING: psimulate produced too little output to contain a completion "
            f"status ({len(lines)} non-empty line(s)). Treating the run as unfinished."
        )
        return False
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
    if not lines:
        print("WARNING: psimulate produced no output to parse a results directory from.")
        return None
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
