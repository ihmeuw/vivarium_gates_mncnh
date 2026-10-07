#!/bin/bash

# This script must be sourced, not executed, because it activates an environment
# in the current shell. Detect non-sourced invocation and fail with a clear message.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo
  echo "ERROR: This script must be sourced, not executed."
  echo "Usage: source environment.sh [options]"
  echo "Run 'source environment.sh -h' for help."
  exit 1
fi

# Define variables
username=$(whoami)
env_type="simulation"
make_new="no"
use_shared="no"
install_git_lfs="no"
env_versions="src/vivarium_gates_mncnh/tools/env_versions.py"
fetch_timeout_seconds=10 # Max wait for fetching origin/main

# Reset OPTIND so help can be invoked multiple times per shell session.
OPTIND=0
Help()
{ 
   # Display Help
   echo
   echo "Script to automatically create and validate conda environments."
   echo
   echo "Syntax: source environment.sh [-h|t|s|f|l]"
   echo "options:"
   echo "h     Print this Help."
   echo "t     Type of conda environment. Either 'simulation' (default) or 'artifact'."
   echo "s     Use shared environment (venv overlay). Recommended for cluster development."
   echo "      Warns, but never rebuilds, when your version files differ from the shared environment's."
   echo "f     Force a rebuild (also picks up conda-level package changes and new commits on an overrides branch)."
   echo "l     Install git lfs (only applies when creating a new conda environment)."
}

# Process input options
while getopts ":hsflt:" option; do
   case $option in
      h) # display help
         Help
         return;;
      t) # Type of conda environment to build
         env_type=$OPTARG;;
      s) # Use shared environment
         use_shared="yes";;
      f) # Force creation of a new environment
         make_new="yes";;
      l) # Install git lfs
         install_git_lfs="yes";;
     \?) # Invalid option
         echo
         echo "ERROR: Invalid option"
         return;;
   esac
done
# Parse environment name
env_name=$(make -s print-dist-name)
if [[ -z "$env_name" ]]; then
  echo "ERROR: could not determine the distribution name from pyproject.toml" >&2
  return 1
fi
env_name+="_$env_type"
branch_name=$(git rev-parse --abbrev-ref HEAD)

# Pull repo to get latest changes from remote if remote exists
git ls-remote --exit-code --heads origin $branch_name >/dev/null 2>&1
exit_code=$?
if [[ "$exit_code" == "0" ]]; then
  git fetch --all
  echo
  echo "Git branch '$branch_name' exists in the remote repository; pulling latest changes"
  git pull origin $branch_name
fi

# Capture error and exit script when sourced
# But also clear the trap when exiting to avoid affecting the parent shell
trap 'trap - ERR && return' ERR
set -E

if [[ "$use_shared" == "yes" ]]; then
  # Deactivate any active conda environments so only the venv is active
  for i in $(seq "${CONDA_SHLVL:-0}"); do
    conda deactivate
  done

  # Shared environment (venv overlay)
  venv_path=".venv/$env_name"
  
  if [[ ! -d "$venv_path" ]] || [[ "$make_new" == "yes" ]]; then
    # Venv doesn't exist or user requested force rebuild
    echo "Creating venv for shared environment in $venv_path"
    make build-shared-env type=$env_type force=yes
  fi
  echo "Activating shared environment venv $env_name"
  source ".venv/$env_name/bin/activate"
  # If the shared env's package versions differ from this checkout's, explain why;
  # we can't rebuild it from here. Nothing below can stop activation.
  # Fetch main (the same ref as MAIN_REF in env_versions.py) without prompting or hanging.
  GIT_TERMINAL_PROMPT=0 timeout "$fetch_timeout_seconds" git fetch --quiet origin main 2>/dev/null || true
  # The venv's base prefix is the shared env.
  shared_prefix="$(python -c 'import sys; print(sys.base_prefix)' 2>/dev/null)" || shared_prefix=""
  if [[ -n "$shared_prefix" ]]; then
    shared_record_dir="$shared_prefix/etc/vivarium_gates_mncnh"  # where the shared env's build recorded its versions
    # Say how this checkout's versions differ from the shared env's, and what to do.
    python "$env_versions" explain-shared-mismatch --repo . --type "$env_type" --record-dir "$shared_record_dir" || true
    # Warn if the shared env was built with framework overrides.
    python "$env_versions" warn-if-overrides --record-dir "$shared_record_dir" || true
  else
    echo "WARNING: could not locate the shared environment; skipping the package version check"
  fi

else
  # Initialize conda if not already initialized
  conda_path=$($SHELL -ic 'conda info --base')
  if [ ! -d "$conda_path" ]; then
    echo
    echo "ERROR: Conda path $conda_path does not exist"
    return
  fi
  if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
    echo
    echo "Initializing conda from $conda_path"
    source "$conda_path/etc/profile.d/conda.sh"
  else
    echo
    echo "ERROR: Unable to find conda in expected locations"
    return
  fi
  # Conda environment
  lfs_flag=""
  if [[ "$install_git_lfs" == "yes" ]]; then
    lfs_flag="lfs=yes"
  fi

  need_to_build="yes"
  env_info=$(conda info --envs | grep $env_name | head -n 1)
  
  if [[ "$env_info" != "" ]]; then
    # Environment exists
    if [[ "$make_new" != "yes" ]]; then
      conda activate $env_name
      # Compare the versions this env was built with against the checkout's version files.
      # Rebuild only if they differ.
      # `&& rc=0 || rc=$?` keeps a nonzero exit from tripping the ERR trap.
      python "$env_versions" compare --repo . --type "$env_type" && rc=0 || rc=$?
      if [[ "$rc" == "0" ]]; then
        need_to_build="no"
      elif [[ "$rc" == "10" ]]; then  # env_versions.EXIT_DIFFERENT
        echo "Package versions changed; rebuilding environment '$env_name'"
      elif [[ "$rc" == "11" ]]; then  # env_versions.EXIT_NO_RECORD
        echo "Environment '$env_name' has no record of its package versions; rebuilding"
      else
        echo "WARNING: could not compare '$env_name' with the version files (exit code $rc); rebuilding"
      fi
    fi
  fi
  
  if [[ "$need_to_build" == "yes" ]]; then
    # Deactivate current environment if it's the one we're about to rebuild
    if [[ "$CONDA_DEFAULT_ENV" == "$env_name" ]]; then
      echo "Deactivating currently active environment..."
      conda deactivate
    fi
    echo "Creating conda environment '$env_name'"
    make build-env type=$env_type name=$env_name force=yes $lfs_flag
  fi
  echo "Activating conda environment '$env_name'"
  conda activate $env_name
  # Warn if this env was built with framework overrides.
  python "$env_versions" warn-if-overrides || true
fi

# Clear the ERR trap to avoid affecting subsequent commands in the parent shell
trap - ERR
