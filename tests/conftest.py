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
from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX, SEED_INDEX

from vivarium_gates_mncnh.constants import paths
from vivarium_gates_mncnh.constants.data_values import SIMULATION_EVENT_NAMES

SIMULATION_STEPS = [
    SIMULATION_EVENT_NAMES.FIRST_TRIMESTER_ANC,
    SIMULATION_EVENT_NAMES.LATER_PREGNANCY_SCREENING,
    SIMULATION_EVENT_NAMES.LATER_PREGNANCY_INTERVENTION,
    SIMULATION_EVENT_NAMES.ULTRASOUND,
    SIMULATION_EVENT_NAMES.DELIVERY_FACILITY,
    SIMULATION_EVENT_NAMES.AZITHROMYCIN_ACCESS,
    SIMULATION_EVENT_NAMES.MISOPROSTOL_ACCESS,
    SIMULATION_EVENT_NAMES.CPAP_ACCESS,
    SIMULATION_EVENT_NAMES.ACS_ACCESS,
    SIMULATION_EVENT_NAMES.ANTIBIOTICS_ACCESS,
    SIMULATION_EVENT_NAMES.PROBIOTICS_ACCESS,
    SIMULATION_EVENT_NAMES.MATERNAL_SEPSIS,
    SIMULATION_EVENT_NAMES.MATERNAL_HEMORRHAGE,
    SIMULATION_EVENT_NAMES.OBSTRUCTED_LABOR,
    SIMULATION_EVENT_NAMES.MORTALITY,
    SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY,
    SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY,
    SIMULATION_EVENT_NAMES.POSTPARTUM_DEPRESSION,
]


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_jenkins = pytest.mark.skip(reason="skipping tests in jenkins")
    is_on_jenkins = os.environ.get("JENKINS_URL")

    if is_on_jenkins:
        for item in items:
            item.add_marker(skip_jenkins)

    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return

    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def model_spec_path() -> Path:
    repo_path = paths.BASE_DIR
    config_path = repo_path / "model_specifications" / "model_spec.yaml"
    return config_path


@pytest.fixture(scope="session")
def sim_state_step_mapper() -> dict[str, int]:
    step_mapper = {}
    for i in range(len(SIMULATION_STEPS)):
        step_mapper[SIMULATION_STEPS[i]] = i + 1
    return step_mapper


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


def _create_births_observer_data() -> pd.DataFrame:
    """Create births observer data for testing."""
    return pd.DataFrame(
        {
            "value": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 11.0, 12.0] * 2,
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("births", "", "", "", "Male", "live_birth", "A", 0, 0),
                ("births", "", "", "", "Male", "live_birth", "A", 0, 1),
                ("births", "", "", "", "Male", "live_birth", "A", 1, 0),
                ("births", "", "", "", "Male", "live_birth", "A", 1, 1),
                ("births", "", "", "", "Male", "live_birth", "B", 0, 0),
                ("births", "", "", "", "Male", "live_birth", "B", 0, 1),
                ("births", "", "", "", "Male", "live_birth", "B", 1, 0),
                ("births", "", "", "", "Male", "live_birth", "B", 1, 1),
                ("births", "", "", "", "Female", "live_birth", "A", 0, 0),
                ("births", "", "", "", "Female", "live_birth", "A", 0, 1),
                ("births", "", "", "", "Female", "live_birth", "A", 1, 0),
                ("births", "", "", "", "Female", "live_birth", "A", 1, 1),
                ("births", "", "", "", "Female", "live_birth", "B", 0, 0),
                ("births", "", "", "", "Female", "live_birth", "B", 0, 1),
                ("births", "", "", "", "Female", "live_birth", "B", 1, 0),
                ("births", "", "", "", "Female", "live_birth", "B", 1, 1),
                ("births", "", "", "", "Male", "still_birth", "A", 0, 0),
                ("births", "", "", "", "Male", "still_birth", "A", 0, 1),
                ("births", "", "", "", "Female", "still_birth", "A", 1, 0),
                ("births", "", "", "", "Female", "still_birth", "A", 1, 1),
            ],
            names=[
                "measure",
                "entity_type",
                "entity",
                "sub_entity",
                "child_sex",
                "pregnancy_outcome",
                "common_stratify_column",
                DRAW_INDEX,
                SEED_INDEX,
            ],
        ),
    )


@pytest.fixture
def births_observer_data() -> pd.DataFrame:
    """Get births observer data for testing."""
    return _create_births_observer_data()


def _create_deaths_observer_data() -> pd.DataFrame:
    """Create deaths observer data for testing."""
    return pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0] * 8,
        },
        index=pd.MultiIndex.from_product(
            [
                ["death_counts"],
                [""],
                [""],
                [""],
                ["Male", "Female"],
                ["early_neonatal", "late_neonatal"],
                ["A", "B"],
                [0, 1],
                [0, 1],
            ],
            names=[
                "measure",
                "entity_type",
                "entity",
                "sub_entity",
                "child_sex",
                "child_age_group",
                "common_stratify_column",
                DRAW_INDEX,
                SEED_INDEX,
            ],
        ),
    )


@pytest.fixture
def deaths_observer_data() -> pd.DataFrame:
    """Get deaths observer data for testing."""
    return _create_deaths_observer_data()


def _create_csmrisk_artifact_data() -> dict[str, pd.DataFrame]:
    """Create artifact data for testing CSMRisk measure. Note that the artifact data has already
    been formatted to where the draw columns are melted into a single column and "input_draw" is
    added as an index level. This happens in the data loader which processes the artifact data
    before passing it to the measure class."""
    return pd.DataFrame(
        {
            "value": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
            ],
        },
        index=pd.MultiIndex.from_product(
            [
                ["Male", "Female"],
                [0.0, round(7 / 365.0, 8)],
                [round(7 / 365.0, 8), round(28 / 365.0, 8)],
                [2023],
                [2024],
                [0, 1],
            ],
            names=[
                "child_sex",
                "child_age_group_start",
                "child_age_group_end",
                "year_start",
                "year_end",
                DRAW_INDEX,
            ],
        ),
    )


@pytest.fixture
def csmrisk_artifact_data() -> pd.DataFrame:
    """Get artifact data for testing CSMRisk measure."""
    return _create_csmrisk_artifact_data()


@pytest.fixture(scope="session")
def v_and_v_artifact_keys_mapper() -> dict[str, str | pd.DataFrame]:
    """Create a mapping of artifact keys to DataFrames for testing."""
    csmrisk = _create_csmrisk_artifact_data()
    return {
        "cause.neonatal_testing.cause_specific_mortality_risk": csmrisk,
        "population.location": "Ethiopia",
    }


@pytest.fixture(scope="session")
def mncnh_results_dir(tmp_path_factory: TempPathFactory) -> Path:
    """Create a temporary directory for simulation outputs."""
    # Create the temporary directory at session scope
    tmp_path = tmp_path_factory.mktemp("mncnh_data")

    # Create the directory structure
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)

    # Create data directly within this session-scoped fixture
    # so we don't depend on function-scoped fixtures
    _births = _create_births_observer_data()
    _deaths = _create_deaths_observer_data()

    # Save Sim DataFrames
    _births.reset_index().to_parquet(results_dir / "births.parquet")
    _deaths.reset_index().to_parquet(results_dir / "neonatal_testing_death_counts.parquet")

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    artifact_path = artifact_dir / "artifact.hdf"
    artifact = Artifact(artifact_path)
    for key, data in v_and_v_artifact_keys_mapper().items():
        artifact.write(key, data)

    model_spec = {
        "configuration": {
            "input_data": {
                "artifact_path": str(artifact_path),
            }
        }
    }

    # Save model specification
    with open(tmp_path / "model_specification.yaml", "w") as f:
        yaml.dump(model_spec, f)

    return tmp_path
