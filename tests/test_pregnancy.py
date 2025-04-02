from pathlib import Path

import pandas as pd
import pytest
from vivarium import Artifact, InteractiveContext
from vivarium_testing_utils import FuzzyChecker

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


def test_pregnancy_durations(
    population: pd.DataFrame,
) -> None:
    """Tests that partial term pregnancies are between 6 and 24 weeks"""
    partial_term_idx = population.index[
        population[COLUMNS.PREGNANCY_OUTCOME] == "partial_term"
    ]
    assert all(
        population.loc[partial_term_idx, COLUMNS.GESTATIONAL_AGE].between(
            DURATIONS.PARTIAL_TERM_LOWER_WEEKS, DURATIONS.PARTIAL_TERM_UPPER_WEEKS
        )
    )


def test_pregnancy_duration_pipeline(
    pregnancy_state: InteractiveContext, population: pd.DataFrame
) -> None:
    """Tests that the pregnancy duration pipeline is correct"""
    gestational_age = population[COLUMNS.GESTATIONAL_AGE]
    unit_converted_ga = pd.to_timedelta(7 * gestational_age, unit="days")
    pregnancy_duration = pregnancy_state.get_value(PIPELINES.PREGNANCY_DURATION)(
        population.index
    )

    assert all(unit_converted_ga == pregnancy_duration)
