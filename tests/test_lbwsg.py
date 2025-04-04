from pathlib import Path

import pandas as pd
import pytest
from vivarium import Artifact, InteractiveContext
from vivarium_testing_utils import FuzzyChecker

from vivarium_gates_mncnh.constants.data_keys import LBWSG
from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    PIPELINES,
    SIMULATION_EVENT_NAMES,
)

from .utilities import get_interactive_context_state


@pytest.fixture(scope="module")
def birth_state(
    model_spec_path: Path, sim_state_step_mapper: dict[str, int]
) -> InteractiveContext:
    sim = InteractiveContext(model_spec_path)
    return get_interactive_context_state(
        sim, sim_state_step_mapper, SIMULATION_EVENT_NAMES.PREGNANCY
    )


@pytest.fixture(scope="module")
def population(birth_state: InteractiveContext) -> pd.DataFrame:
    return birth_state.get_population()


def test_birth_exposure_coverage(
    birth_state: InteractiveContext,
    population: pd.DataFrame,
    artifact: Artifact,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Tests that the birth exposure coverage is correct"""
    draw = (
        f"draw_{birth_state.model_specification.configuration.input_data.input_draw_number}"
    )
    birth_exposure = artifact.load(LBWSG.BIRTH_EXPOSURE)[draw].reset_index()
    sim_exposure = population[
        [
            COLUMNS.SEX_OF_CHILD,
            COLUMNS.GESTATIONAL_AGE_EXPOSURE,
            COLUMNS.BIRTH_WEIGHT_EXPOSURE,
        ]
    ].copy()
    category_intervals = birth_state.get_component(
        "risk_factor.low_birth_weight_and_short_gestation"
    ).exposure_distribution.category_intervals

    def map_to_category(value: float, category_dict: dict[str, pd.Interval]) -> list[str]:
        categories = []
        for key, interval in category_dict.items():
            if value in interval:
                categories.append(key)
        return categories

    # Map 'gestational_age' to its categories
    sim_exposure["gestational_age_category"] = sim_exposure["gestational_age_exposure"].apply(
        lambda x: map_to_category(x, category_intervals["gestational_age"])
    )
    # Map 'birth_weight' to its categories
    sim_exposure["birth_weight_category"] = sim_exposure["birth_weight_exposure"].apply(
        lambda x: map_to_category(x, category_intervals["birth_weight"])
    )
    # Get common category
    sim_exposure["exposure_categories"] = sim_exposure.apply(
        lambda row: set(row["gestational_age_category"]) & set(row["birth_weight_category"]),
        axis=1,
    )

    # Get single category from set returned above but throw error if simulant is in two categories
    def extract_set_value(s: set) -> str:
        if len(s) == 1:
            return s.pop()
        else:
            raise ValueError("Simulant is in multiple LBWSG categories")

    sim_exposure["category"] = sim_exposure["exposure_categories"].apply(extract_set_value)

    # Check each combination of sex and category
    for sex in ["Female", "Male"]:
        sex_subset = sim_exposure.loc[sim_exposure[COLUMNS.SEX_OF_CHILD] == sex]
        sex_exposure = birth_exposure.loc[birth_exposure[COLUMNS.SEX_OF_CHILD] == sex]
        for category in sex_exposure.parameter:
            expected_exposure = sex_exposure.loc[sex_exposure["parameter"] == category][
                draw
            ].iloc[0]
            fuzzy_checker.fuzzy_assert_proportion(
                len(sex_subset.loc[sex_subset["category"] == category]),
                len(sex_subset),
                expected_exposure,
                name=f"{sex}_{category}_exposure_proportion",
            )
