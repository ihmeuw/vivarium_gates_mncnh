"""
Programmatic equivalent of ``model_specifications/workflow_config.yaml``.

VCT is used only to build step instances via its workflow-config API
(``get_python_step`` / ``get_simulation_step`` / ``get_pytest_step``).
This script owns the Jobmon side of the workflow directly: it creates the
``Tool`` and ``Workflow``, calls ``step.get_tasks(...)`` on each step, wires
sequential dependencies, and binds + runs the workflow.

Run from the repository root so relative step paths resolve::

    python -m vivarium_gates_mncnh.tools.run_workflow

A non-base conda environment must be active so steps that do not declare
their own ``environment`` can inherit it (matches the YAML behavior).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from jobmon.client.api import Tool
from jobmon.client.task import Task
from jobmon.client.workflow import Workflow
from jobmon.core.configuration import JobmonConfig
from jobmon.core.exceptions import ConfigError
from vivarium_cluster_tools.psimulate.workflow_config import ResourceConfig
from vivarium_cluster_tools.psimulate.workflow_config.config import BaseStepConfig
from vivarium_cluster_tools.psimulate.workflow_config.interface import (
    get_pytest_step,
    get_python_step,
    get_simulation_step,
    resolve_step_env_prefix,
)

WORKFLOW_NAME = "simulation_workflow"
PROJECT = "proj_simscience"
QUEUE = "all.q"
OUTPUT_DIRECTORY = Path(
    "/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/results/model33.1"
)

MODEL_SPECIFICATION = Path("src/vivarium_gates_mncnh/model_specifications/model_spec.yaml")
BRANCH_CONFIGURATION = Path(
    "src/vivarium_gates_mncnh/model_specifications/branches/scenarios.yaml"
)
ARTIFACT_ROOT = Path(
    "/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts/test_workflow"
)
LOCATIONS = ["ethiopia", "nigeria", "pakistan"]

PYTEST_ENVIRONMENT = "vivarium_gates_mncnh_artifact"
MAX_ATTEMPTS = 2


def build_steps() -> list[BaseStepConfig]:
    """Build each workflow step via the VCT API."""
    check_clean_working_tree_step = get_python_step(
        name="check_clean_working_tree",
        resources=ResourceConfig(
            memory_gb=1,
            project=PROJECT,
            queue=QUEUE,
            runtime="00:01:00",
        ),
        output_directory=OUTPUT_DIRECTORY,
        path="src/vivarium_gates_mncnh/tools/check_working_tree.py",
    )

    simulation_steps = [
        get_simulation_step(
            name=f"run_simulation_{location}",
            resources=ResourceConfig(
                memory_gb=4,
                project=PROJECT,
                queue=QUEUE,
                runtime="00:20:00",
            ),
            output_directory=OUTPUT_DIRECTORY,
            model_specification=MODEL_SPECIFICATION,
            branch_configuration=BRANCH_CONFIGURATION,
            artifact_path=ARTIFACT_ROOT / f"{location}.hdf",
        )
        for location in LOCATIONS
    ]

    tests_artifact_step = get_pytest_step(
        name="tests_artifact",
        resources=ResourceConfig(
            memory_gb=2,
            project=PROJECT,
            queue=QUEUE,
            runtime="01:00:00",
            cores=4,
        ),
        output_directory=OUTPUT_DIRECTORY,
        environment=PYTEST_ENVIRONMENT,
        path=["tests/"],
        k="results",
        runslow=True,
    )

    return [check_clean_working_tree_step, *simulation_steps, tests_artifact_step]


def build_jobmon_workflow(
    steps: list[BaseStepConfig], workflow_args: str, build_timestamp: str, is_resume: bool
) -> Workflow:
    """Create the Jobmon workflow and populate it with tasks from the steps."""
    tool = Tool(name="vivarium_gates_mncnh_run_workflow")
    workflow = tool.create_workflow(
        workflow_args=workflow_args,
        name=WORKFLOW_NAME,
        default_cluster_name="slurm",
        default_max_attempts=MAX_ATTEMPTS,
    )

    previous_step_tasks: list[Task] = []
    all_tasks: list[Task] = []
    for step in steps:
        step_tasks = step.get_tasks(
            tool,
            env_prefix=resolve_step_env_prefix(step),
            build_timestamp=build_timestamp,
            is_resume=is_resume,
        )
        for task in step_tasks:
            for prev in previous_step_tasks:
                task.add_upstream(prev)
        all_tasks.extend(step_tasks)
        previous_step_tasks = step_tasks

    workflow.add_tasks(all_tasks)
    return workflow


def main(resume: bool) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    workflow_args_path = OUTPUT_DIRECTORY / ".workflow_args"
    build_timestamp_path = OUTPUT_DIRECTORY / ".build_timestamp"

    if resume:
        workflow_args = workflow_args_path.read_text().strip()
        build_timestamp = build_timestamp_path.read_text().strip()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        workflow_args = f"workflow_{WORKFLOW_NAME}_{timestamp}"
        build_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        workflow_args_path.write_text(workflow_args)
        build_timestamp_path.write_text(build_timestamp)

    steps = build_steps()
    workflow = build_jobmon_workflow(
        steps,
        workflow_args=workflow_args,
        build_timestamp=build_timestamp,
        is_resume=resume,
    )

    workflow.bind()
    print(f"Submitting Jobmon workflow {workflow_args!r} (id={workflow.workflow_id}).")
    try:
        gui_url = JobmonConfig().get("http", "gui_url")
    except ConfigError:
        gui_url = ""
    if gui_url:
        print(f"Monitor progress at {gui_url}/#/workflow/{workflow.workflow_id}")
    status = workflow.run(resume=resume)
    print(f"Workflow finished with status {status!r}.")
    if status != "D":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously failed workflow, skipping completed tasks.",
    )
    args = parser.parse_args()
    main(resume=args.resume)
