from pathlib import Path

import pandas as pd
import pytest
from vivarium import Artifact, InteractiveContext
from vivarium_testing_utils import FuzzyChecker

from vivarium_gates_mncnh.constants.data_keys import FACILITY_CHOICE, NO_CPAP_RISK
from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    DELIVERY_FACILITY_TYPES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.constants.metadata import PRETERM_AGE_CUTOFF

from .utilities import get_interactive_context_state


@pytest.fixture(scope="module")
def intrapartum_state(
    model_spec_path: Path, sim_state_step_mapper: dict[str, int]
) -> InteractiveContext:
    sim = InteractiveContext(model_spec_path)
    return get_interactive_context_state(
        sim, sim_state_step_mapper, SIMULATION_EVENT_NAMES.PROBIOTICS_ACCESS
    )


@pytest.fixture(scope="module")
def population(intrapartum_state: InteractiveContext) -> pd.DataFrame:
    return intrapartum_state.get_population()


@pytest.mark.parametrize(
    "facility_type",
    [
        DELIVERY_FACILITY_TYPES.HOME,
        DELIVERY_FACILITY_TYPES.CEmONC,
        DELIVERY_FACILITY_TYPES.BEmONC,
    ],
)
def test_delivery_facility_proportions(
    facility_type: str,
    artifact: Artifact,
    fuzzy_checker: FuzzyChecker,
    population: pd.DataFrame,
) -> None:
    location = population[COLUMNS.LOCATION].unique()[0]
    facility_type_mapper = {
        DELIVERY_FACILITY_TYPES.HOME: artifact.load(FACILITY_CHOICE.P_HOME),
        DELIVERY_FACILITY_TYPES.CEmONC: artifact.load(FACILITY_CHOICE.P_CEmONC),
        DELIVERY_FACILITY_TYPES.BEmONC: artifact.load(FACILITY_CHOICE.P_BEmONC),
    }
    birth_idx = population.index[population[COLUMNS.PREGNANCY_OUTCOME] != "partial_term"]
    fuzzy_checker.fuzzy_assert_proportion(
        (population.loc[birth_idx, COLUMNS.DELIVERY_FACILITY_TYPE] == facility_type).sum(),
        len(birth_idx),
        facility_type_mapper[facility_type],
        name=f"facility_type_{facility_type}_proportion",
    )


@pytest.mark.parametrize(
    "intervention",
    [
        "antibiotics",
        "cpap",
        "probiotics",
    ],
)
@pytest.mark.parametrize(
    "facility_type",
    [
        DELIVERY_FACILITY_TYPES.HOME,
        DELIVERY_FACILITY_TYPES.CEmONC,
        DELIVERY_FACILITY_TYPES.BEmONC,
    ],
)
def test_intervention_availability(
    intervention: str,
    facility_type: str,
    artifact: Artifact,
    intrapartum_state: InteractiveContext,
    fuzzy_checker: FuzzyChecker,
    population: pd.DataFrame,
) -> None:
    draw = f"draw_{intrapartum_state.model_specification.configuration.input_data.input_draw_number}"
    facility_idx = population.index[
        population[COLUMNS.DELIVERY_FACILITY_TYPE] == facility_type
    ]
    intervention_access_idx = population.index[
        population[f"{intervention}_available"] == True
    ]
    facility_type_probability_mapper = {
        DELIVERY_FACILITY_TYPES.HOME: artifact.load(
            f"intervention.no_{intervention}_risk.probability_{intervention}_home"
        ),
        DELIVERY_FACILITY_TYPES.CEmONC: artifact.load(
            f"intervention.no_{intervention}_risk.probability_{intervention}_cemonc"
        ),
        DELIVERY_FACILITY_TYPES.BEmONC: artifact.load(
            f"intervention.no_{intervention}_risk.probability_{intervention}_bemonc"
        ),
    }
    facility_access_probability = facility_type_probability_mapper[facility_type]

    # Antibiotics is a dataframe of draws that have the same value for the neonatal age
    # groups. This is a simple workaround to get the value instead of having to set up
    # a bunch of code to get the lookup table from the antibiotics component
    if isinstance(facility_access_probability, pd.DataFrame):
        # THe first value will be the early neonatal age group for females
        facility_access_probability = facility_access_probability[draw].iloc[0]

    fuzzy_checker.fuzzy_assert_proportion(
        len(intervention_access_idx.intersection(facility_idx)),
        len(facility_idx),
        facility_access_probability,
        name=f"{intervention}_availability_{facility_type}_proportion",
    )


def test_probiotics_access(population: pd.DataFrame) -> None:
    # Probiotics is only available for preterm births
    not_preterm_idx = population.index[
        population[COLUMNS.GESTATIONAL_AGE_EXPOSURE] >= PRETERM_AGE_CUTOFF
    ]
    assert (population.loc[not_preterm_idx, COLUMNS.PROBIOTICS_AVAILABLE] == False).all()
