from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext

from vivarium_gates_mncnh.components.mortality import MaternalDisordersBurden
from vivarium_gates_mncnh.constants.data_values import COLUMNS, SIMULATION_EVENT_NAMES

from .utilities import get_interactive_context_state


@pytest.fixture(scope="module")
def mortality_state(
    sim_state_step_mapper: dict[str, int], model_spec_path: Path
) -> InteractiveContext:
    sim = InteractiveContext(model_spec_path)
    # Use last time step of mortality to have both mother and neonatal deaths
    return get_interactive_context_state(
        sim, sim_state_step_mapper, SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY
    )


@pytest.fixture(scope="module")
def population(mortality_state: InteractiveContext) -> pd.DataFrame:
    return mortality_state.get_population()


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
