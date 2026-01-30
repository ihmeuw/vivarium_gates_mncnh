"""Conftest for automated validation tests that require vivarium_testing_utils imports."""
import pandas as pd
import pytest
from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX, SEED_INDEX


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


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
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
                "sex_of_child",
                "child_age_start",
                "child_age_end",
                "year_start",
                "year_end",
                DRAW_INDEX,
            ],
        ),
    )


@pytest.fixture(scope="session")
def csmrisk_artifact_data() -> pd.DataFrame:
    """Get artifact data for testing CSMRisk measure."""
    return _create_csmrisk_artifact_data()


def _create_adjusted_births_artifact_data() -> pd.DataFrame:
    """Create artifact data for testing AdjustedBirths measure."""
    return pd.DataFrame(
        {
            "value": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0] * 2,
        },
        index=pd.MultiIndex.from_product(
            [
                ["Male", "Female"],
                [0, round(7 / 365.0, 8)],
                [round(7 / 365.0, 8), round(28 / 365.0, 8)],
                [2023],
                [2024],
                [0, 1],
            ],
            names=[
                "sex_of_child",
                "child_age_start",
                "child_age_end",
                "year_start",
                "year_end",
                DRAW_INDEX,
            ],
        ),
    )


@pytest.fixture(scope="session")
def adjusted_births_artifact_data() -> pd.DataFrame:
    """Get artifact data for testing AdjustedBirths measure."""
    return _create_adjusted_births_artifact_data()


@pytest.fixture(scope="session")
def v_and_v_artifact_keys_mapper() -> dict[str, str | pd.DataFrame]:
    """Create a mapping of artifact keys to DataFrames for testing."""
    csmrisk = _create_csmrisk_artifact_data()
    adjusted_births = _create_adjusted_births_artifact_data()
    return {
        "cause.neonatal_testing.mortality_risk": csmrisk,
        "population.location": "Ethiopia",
        "cause.neonatal_testing.adjusted_birth_counts": adjusted_births,
    }
