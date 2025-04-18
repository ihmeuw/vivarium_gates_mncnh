from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vivarium import Artifact, InteractiveContext
from vivarium_testing_utils import FuzzyChecker

from vivarium_gates_mncnh.components.mortality import MaternalDisordersBurden
from vivarium_gates_mncnh.constants.data_keys import POPULATION
from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    NEONATAL_CAUSES,
    SIMULATION_EVENT_NAMES,
)

from .utilities import get_births_and_deaths_idx, get_interactive_context_state


@pytest.fixture(scope="module")
def late_neonatal_mortality_state(
    sim_state_step_mapper: dict[str, int], model_spec_path: Path
) -> InteractiveContext:
    sim = InteractiveContext(model_spec_path)
    # Use last time step of mortality to have both mother and neonatal deaths
    return get_interactive_context_state(
        sim, sim_state_step_mapper, SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY
    )


@pytest.fixture(scope="module")
def early_neonatal_mortality_state(
    sim_state_step_mapper: dict[str, int], model_spec_path: Path
) -> InteractiveContext:
    sim = InteractiveContext(model_spec_path)
    return get_interactive_context_state(
        sim, sim_state_step_mapper, SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY
    )


@pytest.fixture(scope="module")
def population(late_neonatal_mortality_state: InteractiveContext) -> pd.DataFrame:
    return late_neonatal_mortality_state.get_population()


def test_get_proportional_case_fatality_rates():
    """Tests that proportional case fatality rates sum to 1."""

    # Make case fatality data
    simulant_idx = pd.Index(list(range(10)))
    data_vals = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    choice_data = pd.DataFrame(index=simulant_idx)

    mortality = MaternalDisordersBurden()
    for disoder in mortality.maternal_disorders:
        choice_data[disoder] = data_vals
    # Get total case fatality rates
    choice_data["mortality_probability"] = choice_data.sum(axis=1)

    proportional_cfr_data = mortality.get_proportional_case_fatality_rates(choice_data)

    proportional_cfr_cols = [
        col for col in proportional_cfr_data.columns if "proportional_cfr" in col
    ]
    assert proportional_cfr_data[proportional_cfr_cols].sum(axis=1).all() == 1.0


@pytest.mark.parametrize(
    "cause_of_death_column, alive_column",
    [
        (COLUMNS.MOTHER_CAUSE_OF_DEATH, COLUMNS.MOTHER_ALIVE),
        (COLUMNS.CHILD_CAUSE_OF_DEATH, COLUMNS.CHILD_ALIVE),
    ],
)
def test_cause_of_death_normalized(
    cause_of_death_column: str, alive_column: str, population: pd.DataFrame
) -> None:
    dead = population.loc[population[alive_column] == "dead"]
    is_normalized = 0.0
    for cause_of_death in dead[cause_of_death_column].unique():
        is_normalized += (dead[cause_of_death_column] == cause_of_death).sum() / len(dead)

    assert np.isclose(is_normalized, 1.0, atol=1e-6)


@pytest.mark.parametrize("sex", ["Male", "Female"])
def test_neonatal_acmr(
    sex: str,
    early_neonatal_mortality_state: InteractiveContext,
    late_neonatal_mortality_state: InteractiveContext,
    artifact: Artifact,
    fuzzy_checker: FuzzyChecker,
) -> None:
    enn = early_neonatal_mortality_state.get_population()
    lnn = late_neonatal_mortality_state.get_population()

    enn_death_idx, enn_live_birth_idx = get_births_and_deaths_idx(
        enn,
        sex,
        "early_neonatal_mortality",
        [
            NEONATAL_CAUSES.PRETERM_BIRTH_WITH_RDS,
            NEONATAL_CAUSES.PRETERM_BIRTH_WITHOUT_RDS,
            NEONATAL_CAUSES.NEONATAL_SEPSIS,
            NEONATAL_CAUSES.NEONATAL_ENCEPHALOPATHY,
            "other_causes",
        ],
    )
    lnn_death_idx, lnn_live_birth_idx = get_births_and_deaths_idx(
        lnn,
        sex,
        "late_neonatal_mortality",
        [
            NEONATAL_CAUSES.PRETERM_BIRTH_WITH_RDS,
            NEONATAL_CAUSES.PRETERM_BIRTH_WITHOUT_RDS,
            NEONATAL_CAUSES.NEONATAL_SEPSIS,
            NEONATAL_CAUSES.NEONATAL_ENCEPHALOPATHY,
            "other_causes",
        ],
    )
    # Subtract out early neonatal deaths from late neonatal deaths to not count deaths twice
    lnn_live_birth_idx = lnn_live_birth_idx.difference(enn_death_idx)

    acmr = artifact.load(POPULATION.ACMR)
    draw = f"draw_{late_neonatal_mortality_state.model_specification.configuration.input_data.input_draw_number}"
    acmr = acmr[draw].reset_index()
    # Query acmr for sex and age depending on simulation state
    anc_filter = {
        "early_neonatal": f'sex == "{sex}" and age_start < 7 / 365',
        "late_neonatal": f'sex == "{sex}" and age_start > 6 / 365 and age_end < 29 / 365',
    }
    # 1 row dataframe - convert to probability
    enn_acmr = acmr.query(anc_filter["early_neonatal"]).iloc[0][draw] * (7 / 365)
    lnn_acmr = acmr.query(anc_filter["late_neonatal"]).iloc[0][draw] * (21 / 365)

    # acmr = neonatal deaths / live_births
    fuzzy_checker.fuzzy_assert_proportion(
        len(enn_death_idx),
        len(enn_live_birth_idx),
        enn_acmr,
        name=f"early_neonatal_{sex}_neonatal_acmr",
    )
    fuzzy_checker.fuzzy_assert_proportion(
        len(lnn_death_idx),
        len(lnn_live_birth_idx),
        lnn_acmr,
        name=f"late_neonatal_{sex}_neonatal_acmr",
    )


@pytest.mark.parametrize("sex", ["Male", "Female"])
@pytest.mark.parametrize(
    "cause_of_death",
    [
        [NEONATAL_CAUSES.PRETERM_BIRTH_WITH_RDS, NEONATAL_CAUSES.PRETERM_BIRTH_WITHOUT_RDS],
        [NEONATAL_CAUSES.NEONATAL_SEPSIS],
        [NEONATAL_CAUSES.NEONATAL_ENCEPHALOPATHY],
    ],
)
def test_neonatal_csmr(
    sex: str,
    cause_of_death: list[str],
    early_neonatal_mortality_state: InteractiveContext,
    late_neonatal_mortality_state: InteractiveContext,
    artifact: Artifact,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Tests the csmr for each neonatal cause in the model. Note that both preterm with and without
    RDS are combined to match the GBD value."""
    enn = early_neonatal_mortality_state.get_population()
    lnn = late_neonatal_mortality_state.get_population()

    enn_death_idx, enn_live_birth_idx = get_births_and_deaths_idx(
        enn, sex, "early_neonatal_mortality", cause_of_death
    )
    lnn_death_idx, lnn_live_birth_idx = get_births_and_deaths_idx(
        lnn, sex, "late_neonatal_mortality", cause_of_death
    )
    # Subtract out early neonatal deaths from late neonatal deaths to not count deaths twice
    lnn_live_birth_idx = lnn_live_birth_idx.difference(enn_death_idx)

    artifact_cause = (
        cause_of_death[0]
        if cause_of_death[0] != NEONATAL_CAUSES.PRETERM_BIRTH_WITH_RDS
        else "neonatal_preterm_birth"
    )
    csmr = artifact.load(f"cause.{artifact_cause}.cause_specific_mortality_rate")
    draw = f"draw_{late_neonatal_mortality_state.model_specification.configuration.input_data.input_draw_number}"
    csmr = csmr[draw].reset_index()
    # Query acmr for sex and age depending on simulation state
    data_filter = {
        "early_neonatal": f'sex_of_child == "{sex}" and child_age_start < 7 / 365',
        "late_neonatal": f'sex_of_child == "{sex}" and child_age_start > 6 / 365 and child_age_end < 29 / 365',
    }
    # 1 row dataframe - convert to probability
    enn_csmr = csmr.query(data_filter["early_neonatal"]).iloc[0][draw] * (7 / 365)
    lnn_csmr = csmr.query(data_filter["late_neonatal"]).iloc[0][draw] * (21 / 365)

    # csmr = neonatal deaths / live_births
    fuzzy_checker.fuzzy_assert_proportion(
        len(enn_death_idx),
        len(enn_live_birth_idx),
        enn_csmr,
        name=f"early_neonatal_{sex}_neonatal_acmr",
    )
    fuzzy_checker.fuzzy_assert_proportion(
        len(lnn_death_idx),
        len(lnn_live_birth_idx),
        lnn_csmr,
        name=f"late_neonatal_{sex}_neonatal_acmr",
    )
