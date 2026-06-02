"""Smoke test for SepsisEffectsOnHemoglobin component.

V&V criteria: Hemoglobin concentration stratified by puerperal sepsis incidence
should differ by approximately the magnitude of the puerperal sepsis hemoglobin
effect in each post-pregnancy time step.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vivarium import InteractiveContext

from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    PIPELINES,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
)

from .utilities import get_interactive_context_state


@pytest.fixture(scope="module")
def early_neonatal_state(
    sim_state_step_mapper: dict[str, int], model_spec_path: Path
) -> InteractiveContext:
    sim = InteractiveContext(model_spec_path)
    return get_interactive_context_state(
        sim, sim_state_step_mapper, SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY
    )


@pytest.fixture(scope="module")
def late_neonatal_state(
    sim_state_step_mapper: dict[str, int], model_spec_path: Path
) -> InteractiveContext:
    sim = InteractiveContext(model_spec_path)
    return get_interactive_context_state(
        sim, sim_state_step_mapper, SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY
    )


@pytest.mark.parametrize(
    "state_fixture,step_name",
    [
        ("early_neonatal_state", "early_neonatal_mortality"),
        ("late_neonatal_state", "late_neonatal_mortality"),
    ],
)
def test_sepsis_hemoglobin_shift_applied(
    state_fixture: str, step_name: str, request: pytest.FixtureRequest
) -> None:
    """Check that simulants with sepsis have lower hemoglobin than those without."""
    sim: InteractiveContext = request.getfixturevalue(state_fixture)

    pop = sim.get_population(
        [COLUMNS.MATERNAL_SEPSIS, COLUMNS.PREGNANCY_OUTCOME]
    )
    # Only look at live births (avoid confounding from stillbirths/other outcomes)
    live_births = pop.loc[
        pop[COLUMNS.PREGNANCY_OUTCOME] == PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME
    ]

    has_sepsis = live_births[COLUMNS.MATERNAL_SEPSIS]
    sepsis_idx = live_births.index[has_sepsis]
    no_sepsis_idx = live_births.index[~has_sepsis]

    if len(sepsis_idx) == 0:
        pytest.skip(f"No simulants with sepsis at {step_name}")

    # Get hemoglobin exposure values from the pipeline
    hgb = sim.get_value(PIPELINES.HEMOGLOBIN_EXPOSURE)(live_births.index)

    mean_hgb_sepsis = hgb.loc[sepsis_idx].mean()
    mean_hgb_no_sepsis = hgb.loc[no_sepsis_idx].mean()

    # Sepsis shift is negative, so sepsis group should have lower hemoglobin
    observed_diff = mean_hgb_sepsis - mean_hgb_no_sepsis
    assert observed_diff < 0, (
        f"Expected sepsis group to have lower hemoglobin at {step_name}. "
        f"Sepsis mean: {mean_hgb_sepsis:.2f}, No sepsis mean: {mean_hgb_no_sepsis:.2f}"
    )
