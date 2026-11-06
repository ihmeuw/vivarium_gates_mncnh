#!/bin/bash

# Build this checkout's .venv from uv.lock and activate it.

# This script must be sourced, not executed, because it activates an environment
# in the current shell. Detect non-sourced invocation and fail with a clear message.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo
  echo "ERROR: This script must be sourced, not executed."
  echo "Usage: source environment.sh [options]"
  echo "Run 'source environment.sh -h' for help."
  exit 1
fi

_vgm_environment() {
  # The uv version this repo is tested with; older versions are refused.
  # TODO: read required-version from pyproject.toml instead, once build-utils
  # reads it (MIC-7678) and that release is tested (MIC-7681).
  local uv_version="0.13.6"
  local uv_install="curl -LsSf https://astral.sh/uv/${uv_version}/install.sh | sh"
  # psimulate reads its Slack token from the environment.
  # TODO: remove once psimulate reads the token itself.
  local slack_config="/mnt/team/simulation_science/priv/engineering/config/slack_bot_config.sh"

  local repo_root force="no" option OPTIND=1
  repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

  while getopts ":hf" option; do
    case $option in
      h)
        echo
        echo "Build this checkout's .venv from uv.lock and activate it."
        echo
        echo "Syntax: source environment.sh [-h|f]"
        echo "options:"
        echo "h     Print this Help."
        echo "f     Reinstall every package in .venv."
        return 0;;
      f)
        force="yes";;
      \?)
        echo
        echo "ERROR: Invalid option -$OPTARG. Run 'source environment.sh -h' for help."
        return 1;;
    esac
  done

  # Define the merge/diff drivers referenced in .gitattributes.
  # We set the nbdime keys directly
  # rather than running 'nbdime config-git --enable', because that writes
  # '*.ipynb merge=jupyternotebook' to .git/info/attributes, which overrides
  # .gitattributes and would defeat the merge=ours rule for executed notebooks.
  git -C "$repo_root" config merge.ours.driver true
  git -C "$repo_root" config merge.jupyternotebook.name "jupyter notebook merge driver"
  git -C "$repo_root" config merge.jupyternotebook.driver "git-nbmergedriver merge %O %A %B %L %P"
  git -C "$repo_root" config diff.jupyternotebook.command "git-nbdiffdriver diff"

  # uv must be on PATH and new enough.
  if ! command -v uv >/dev/null 2>&1; then
    echo
    echo "ERROR: uv is not on PATH. Install it with:"
    echo "  $uv_install"
    return 1
  fi
  local found
  found=$(uv --version | awk '{print $2}')
  if [[ "$(printf '%s\n%s\n' "$uv_version" "$found" | sort -V | head -n 1)" != "$uv_version" ]]; then
    echo
    echo "ERROR: uv $found is older than $uv_version. Install the newer one with:"
    echo "  $uv_install"
    return 1
  fi

  # Leftovers from the old setup. uv will not use a .venv that is not a venv,
  # so stop there. We never delete these ourselves.
  if [[ -d "$repo_root/.venv" && ! -f "$repo_root/.venv/pyvenv.cfg" ]]; then
    echo
    echo "ERROR: $repo_root/.venv is not a venv. It holds the old shared-environment overlays:"
    ls -d "$repo_root"/.venv/*/ 2>/dev/null | sed 's/^/  /'
    echo "Remove it, then source this script again:"
    echo "  rm -rf $repo_root/.venv"
    return 1
  fi
  local hooks_dir hook lfs_hooks=()
  # Relative to repo_root; --path-format=absolute needs a newer git than the cluster's.
  hooks_dir=$(git -C "$repo_root" rev-parse --git-path hooks 2>/dev/null)
  [[ "$hooks_dir" == /* ]] || hooks_dir="$repo_root/$hooks_dir"
  for hook in "$hooks_dir"/*; do
    [[ -f "$hook" ]] && grep -q "git lfs\|git-lfs" "$hook" && lfs_hooks+=("$hook")
  done
  if (( ${#lfs_hooks[@]} )); then
    echo
    echo "NOTE: $hooks_dir holds Git LFS hooks left by the old git-lfs option."
    echo "This repo tracks no files in Git LFS. To remove them:"
    echo "  rm ${lfs_hooks[*]}"
  fi

  # Only .venv should be active.
  if declare -F deactivate >/dev/null; then
    deactivate
  fi
  if declare -F conda >/dev/null; then
    local i
    for i in $(seq "${CONDA_SHLVL:-0}"); do
      conda deactivate
    done
  fi

  local sync_args=(--locked --project "$repo_root")
  [[ "$force" == "yes" ]] && sync_args+=(--reinstall)
  if ! uv sync "${sync_args[@]}"; then
    echo
    if ! uv lock --check --project "$repo_root" >/dev/null 2>&1; then
      echo "ERROR: uv.lock does not match pyproject.toml."
      echo "Run 'make lock-versions' and commit uv.lock."
    else
      echo "ERROR: 'uv sync ${sync_args[*]}' failed; see the messages above."
    fi
    return 1
  fi

  echo "Activating $repo_root/.venv"
  source "$repo_root/.venv/bin/activate"

  if [[ -r "$slack_config" ]]; then
    source "$slack_config"
  fi
}

if _vgm_environment "$@"; then
  unset -f _vgm_environment
else
  unset -f _vgm_environment
  return 1
fi
