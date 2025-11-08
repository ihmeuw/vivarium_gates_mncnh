from pathlib import Path

import pandas as pd
import pytest
from vivarium import Artifact, InteractiveContext
from vivarium_testing_utils import FuzzyChecker

from vivarium_gates_mncnh.constants.data_keys import LBWSG, POPULATION
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
def mortality_state(
    model_spec_path: Path, sim_state_step_mapper: dict[str, int]
) -> InteractiveContext:
    sim = InteractiveContext(model_spec_path)
    return get_interactive_context_state(
        sim, sim_state_step_mapper, SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY
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
    # Use exposure before we apply intervention effects, so it was sampled directly from GBD
    sim_exposure = get_simulation_exposure_categories(
        population, birth_state, intervention_effects=False
    )
    # all simulants should be in a LBWSG category from GBD
    assert (sim_exposure["category"] != "").all()

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


def test_relative_risk(
    mortality_state: InteractiveContext,
    artifact: Artifact,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Tests that the relative risk is correct"""
    draw = f"draw_{mortality_state.model_specification.configuration.input_data.input_draw_number}"
    relative_risk = artifact.load(LBWSG.RELATIVE_RISK)[draw].reset_index()
    population = mortality_state.get_population()
    sim_exposure = get_simulation_exposure_categories(population, mortality_state)

    # Get ACMR source and pipeline
    acmr_source = mortality_state.get_value(PIPELINES.ACMR)(population.index)
    acmr_pipeline = mortality_state.get_value(PIPELINES.DEATH_IN_AGE_GROUP_PROBABILITY)(
        population.index
    )
    rr_pipeline = mortality_state.get_value(PIPELINES.ACMR_RR)(population.index)

    # TODO: Test something?


def get_simulation_exposure_categories(
    pop: pd.DataFrame,
    sim: InteractiveContext,
    intervention_effects: bool = True,
) -> pd.DataFrame:
    sim_exposure = pop[
        [
            COLUMNS.SEX_OF_CHILD,
            COLUMNS.GESTATIONAL_AGE_EXPOSURE,
            COLUMNS.BIRTH_WEIGHT_EXPOSURE,
        ]
    ].copy()
    if intervention_effects:
        sim_exposure = pd.concat(
            [
                pop[COLUMNS.SEX_OF_CHILD],
                sim.get_value(PIPELINES.GESTATIONAL_AGE_EXPOSURE)(pop.index),
                sim.get_value(PIPELINES.BIRTH_WEIGHT_EXPOSURE)(pop.index),
            ],
            axis=1,
        )
    else:
        sim_exposure = pd.concat(
            [
                pop[COLUMNS.SEX_OF_CHILD],
                sim.get_value(PIPELINES.GESTATIONAL_AGE_EXPOSURE)
                .source(pop.index)
                .rename("gestational_age.birth_exposure"),
                sim.get_value(PIPELINES.BIRTH_WEIGHT_EXPOSURE)
                .source(pop.index)
                .rename("birth_weight.birth_exposure"),
            ],
            axis=1,
        )
    category_intervals = sim.get_component(
        "risk_factor.low_birth_weight_and_short_gestation"
    ).exposure_distribution.category_intervals

    def map_to_category(value: float, category_dict: dict[str, pd.Interval]) -> list[str]:
        categories = []
        for key, interval in category_dict.items():
            if value in interval:
                categories.append(key)
        return categories

    # Map 'gestational_age' to its categories
    sim_exposure["gestational_age_category"] = sim_exposure[
        "gestational_age.birth_exposure"
    ].apply(lambda x: map_to_category(x, category_intervals["gestational_age"]))
    # Map 'birth_weight' to its categories
    sim_exposure["birth_weight_category"] = sim_exposure["birth_weight.birth_exposure"].apply(
        lambda x: map_to_category(x, category_intervals["birth_weight"])
    )
    # Get common category
    sim_exposure["exposure_categories"] = sim_exposure.apply(
        lambda row: set(row["gestational_age_category"]) & set(row["birth_weight_category"]),
        axis=1,
    )

    # Get single category from set returned above but throw error if simulant is in two categories
    def extract_set_value(s: set) -> str:
        if len(s) == 0:
            return ""
        elif len(s) == 1:
            return s.pop()
        else:
            raise ValueError("Simulant is in multiple LBWSG categories")

    sim_exposure["category"] = sim_exposure["exposure_categories"].apply(extract_set_value)
    return sim_exposure
