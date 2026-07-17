import os
from pathlib import Path
from typing import Any, Generator

import pandas as pd
import pytest
import yaml
from layered_config_tree import LayeredConfigTree
from pytest import TempPathFactory
from vivarium import Artifact
from vivarium_testing_utils import FuzzyChecker

from vivarium_gates_mncnh.constants import paths

# Detect environment type by checking for vivarium_inputs package
try:
    import vivarium_inputs

    IS_SIMULATION_ENV = False
except ImportError:
    IS_SIMULATION_ENV = True

# Use collect_ignore to prevent pytest from collecting certain directories
# This is evaluated at module level before collection starts
collect_ignore = []
if IS_SIMULATION_ENV:
    # In simulation env, don't collect tests from automated_v_and_v and model_notebooks
    collect_ignore.extend(
        [
            "automated_v_and_v",
            "model_notebooks",
        ]
    )
else:
    # In artifact env, only collect tests from automated_v_and_v and model_notebooks
    # Ignore all test_*.py files in the tests root directory
    test_dir = Path(__file__).parent
    for test_file in test_dir.glob("test_*.py"):
        collect_ignore.append(test_file.name)


def pytest_collection_modifyitems(config, items):
    skip_jenkins = pytest.mark.skip(reason="skipping tests in jenkins")
    is_on_jenkins = os.environ.get("JENKINS_URL")

    if is_on_jenkins:
        for item in items:
            item.add_marker(skip_jenkins)


@pytest.fixture(scope="session")
def model_spec_path() -> Path:
    repo_path = paths.BASE_DIR
    config_path = repo_path / "model_specifications" / "model_spec.yaml"
    return config_path


@pytest.fixture(scope="session")
def sim_state_step_mapper(model_spec_path: Path) -> dict[str, int]:
    # Derive the step -> take_steps count from the model spec's simulation_events
    # (the single source of truth) so it can never drift from the real ordering.
    events = list(LayeredConfigTree(model_spec_path).configuration.time.simulation_events)
    return {event: i + 1 for i, event in enumerate(events)}


@pytest.fixture(scope="session")
def artifact(model_spec_path) -> Artifact:
    config = LayeredConfigTree(model_spec_path)
    artifact = Artifact(config.configuration.input_data.artifact_path)
    return artifact


@pytest.fixture(scope="session")
def output_directory() -> str:
    v_v_path = Path(os.path.dirname(__file__)) / "v_and_v_output"
    return v_v_path


@pytest.fixture(scope="session")
def fuzzy_checker(output_directory) -> Generator[FuzzyChecker, Any, Any]:
    checker = FuzzyChecker()

    yield checker

    checker.save_diagnostic_output(output_directory)
