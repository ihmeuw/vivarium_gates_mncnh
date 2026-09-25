#!/usr/bin/env bash
#
# Run the whole LBWSG PAF workflow by hand, in order.
#
# The workflow alternates between the artifact and simulation environments,
# whose dependencies conflict and cannot be merged. run_paf_sim.py runs ONE
# phase per invocation, wholly inside whichever environment is active, so
# something has to enter the right environment between phases. On the cluster
# that is the workflow runner (see artifact_workflow.yaml). This script is the
# equivalent for running it by hand.
#
# It makes no assumption about how your environments were built. Name them with
# ARTIFACT_ENV and SIMULATION_ENV, as either:
#
#   a conda environment name   e.g. ARTIFACT_ENV=vivarium_gates_mncnh_artifact
#   a path to a venv           e.g. ARTIFACT_ENV=.venv/vivarium_gates_mncnh_artifact
#
# Anything containing a '/' is treated as a venv path; anything else as a conda
# name. Defaults cover both standard setups: the `make build-shared-env` venv
# overlays if present, otherwise the `make build-env` conda environments.
#
# Usage:
#   ./run_paf_workflow.sh -a test_automation -l Ethiopia
#   ARTIFACT_ENV=my_artifact_env ./run_paf_workflow.sh -a test_automation
#
# Every argument is passed through to run_paf_sim.py unchanged.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUN_PAF_SIM="$SCRIPT_DIR/run_paf_sim.py"

DIST_NAME="vivarium_gates_mncnh"

default_env () {  # $1 = artifact | simulation
    local overlay="$REPO_ROOT/.venv/${DIST_NAME}_$1"
    if [[ -d "$overlay" ]]; then
        printf '%s' "$overlay"
    else
        printf '%s' "${DIST_NAME}_$1"
    fi
}

ARTIFACT_ENV="${ARTIFACT_ENV:-$(default_env artifact)}"
SIMULATION_ENV="${SIMULATION_ENV:-$(default_env simulation)}"

# Each phase, and the environment it needs. Must agree with STEP_ENVIRONMENTS
# in run_paf_sim.py and with the per-step `environment` keys in
# artifact_workflow.yaml.
STEPS=(
    "artifact:initial-artifact"
    "simulation:enn-paf"
    "artifact:enn-artifact"
    "simulation:lnn-paf"
    "artifact:final-artifact"
)

run_step () {  # $1 = flavour, $2 = step, rest = pass-through args
    local flavour="$1" step="$2"; shift 2
    local env_spec
    case "$flavour" in
        artifact)   env_spec="$ARTIFACT_ENV" ;;
        simulation) env_spec="$SIMULATION_ENV" ;;
        *) echo "Unknown environment flavour '$flavour'" >&2; return 1 ;;
    esac

    echo
    echo "################################################################################"
    echo "# step '$step'  --  $flavour environment: $env_spec"
    echo "################################################################################"

    # A subshell, so an activation cannot leak into the next step.
    (
        if [[ "$env_spec" == */* ]]; then
            local activate="$env_spec/bin/activate"
            if [[ ! -f "$activate" ]]; then
                echo "No venv at '$env_spec' (expected $activate)." >&2
                echo "Build it with 'source environment.sh -s${flavour:+ -t $flavour}', or set ${flavour^^}_ENV." >&2
                exit 1
            fi
            # Sourcing activate, rather than calling the interpreter directly,
            # is what puts the shared environment's bin/ on PATH -- psimulate
            # lives there, not in the overlay.
            # shellcheck disable=SC1090
            source "$activate"
        else
            if ! command -v conda >/dev/null 2>&1; then
                echo "'$env_spec' looks like a conda environment name, but conda is not on PATH." >&2
                echo "Set ${flavour^^}_ENV to a venv path instead." >&2
                exit 1
            fi
            # conda activate needs the shell hook in a non-interactive shell.
            eval "$(conda shell.bash hook)"
            conda activate "$env_spec"
        fi

        python "$RUN_PAF_SIM" --step "$step" "$@"
    )
}

for entry in "${STEPS[@]}"; do
    run_step "${entry%%:*}" "${entry#*:}" "$@"
done

echo
echo "################################################################################"
echo "# All five phases completed."
echo "################################################################################"
