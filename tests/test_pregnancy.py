from pathlib import Path

import pandas as pd
import pytest
from vivarium import InteractiveContext

from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    DURATIONS,
    PIPELINES,
    SIMULATION_EVENT_NAMES,
)

from .utilities import get_interactive_context_state


@pytest.fixture(scope="module")
def pregnancy_state(
    model_spec_path: Path, sim_state_step_mapper: dict[str, int]
) -> InteractiveContext:
    sim = InteractiveContext(model_spec_path)
    return get_interactive_context_state(
        sim, sim_state_step_mapper, SIMULATION_EVENT_NAMES.PREGNANCY
    )


@pytest.fixture(scope="module")
def population(pregnancy_state: InteractiveContext) -> pd.DataFrame:
    return pregnancy_state.get_population()


def test_partial_term_pregnancy_durations(
    population: pd.DataFrame,
) -> None:
    """Tests that partial term pregnancies are between 6 and 24 weeks"""
    partial_term_idx = population.index[
        population[COLUMNS.PREGNANCY_OUTCOME] == "partial_term"
    ]
    assert all(
        population.loc[partial_term_idx, COLUMNS.PARTIAL_TERM_PREGNANCY_DURATION].between(
            DURATIONS.PARTIAL_TERM_LOWER_WEEKS, DURATIONS.PARTIAL_TERM_UPPER_WEEKS
        )
    )
    non_partial_term_idx = population.index.difference(partial_term_idx)
    assert all(
        population.loc[non_partial_term_idx, COLUMNS.PARTIAL_TERM_PREGNANCY_DURATION].isna()
    )


def test_pregnancy_duration_pipeline(
    pregnancy_state: InteractiveContext, population: pd.DataFrame
) -> None:
    """Tests that the pregnancy duration pipeline is correct"""

    partial_term_idx = population.index[
        population[COLUMNS.PREGNANCY_OUTCOME] == "partial_term"
    ]
    partial_ga = population.loc[partial_term_idx, COLUMNS.PARTIAL_TERM_PREGNANCY_DURATION]
    non_partial_idx = population.index.difference(partial_term_idx)
    non_partial_ga = pregnancy_state.get_value(PIPELINES.GESTATIONAL_AGE_EXPOSURE)(
        non_partial_idx
    )
    gestational_age = pd.concat([partial_ga, non_partial_ga]).sort_index()
    unit_converted_ga = pd.to_timedelta(7 * gestational_age, unit="days")
    pregnancy_duration = pregnancy_state.get_value(PIPELINES.PREGNANCY_DURATION)(
        population.index
    )

    assert all(unit_converted_ga == pregnancy_duration)
