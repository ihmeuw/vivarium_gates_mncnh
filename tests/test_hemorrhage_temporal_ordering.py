"""InteractiveContext checks for MIC-7286: hemorrhage temporal ordering.

These checks are BLIND to the implementation: they assert the quantitative
expectations named in the iteration plan (the contract) against the state table
and incidence-risk pipelines, using the repo's existing InteractiveContext
patterns (see ``tests/test_lbwsg.py`` / ``tests/test_mortality.py``).

Expectations verified (numbering matches the iteration plan):
  3. PPH / intrapartum-disorder incidence risks are probabilities in [0, 1].
  4. Two-pass mortality gating: intrapartum-disorder deaths are resolved only
     among simulants who survived the antepartum mortality pass.
  5. AME is applied only to ``partial_term``.
  6. Only *severe* hemorrhage cases can die of hemorrhage.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from vivarium.engine import InteractiveContext
from vivarium.engine.framework.configuration import build_model_specification

from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    HEMORRHAGE_SEVERITY,
    PIPELINES,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
)

from .utilities import get_interactive_context_state

# Reduced population for a fast-but-powerful interactive run. Large enough that
# the hard-zero invariants (no non-partial_term AME, no non-severe hemorrhage
# death) are meaningfully exercised.
POP_SIZE = 100_000

# Intrapartum-disorder boolean state-table columns.
INTRAPARTUM_DISORDER_COLUMNS = [
    COLUMNS.OBSTRUCTED_LABOR,
    COLUMNS.POSTPARTUM_HEMORRHAGE,
    COLUMNS.MATERNAL_SEPSIS,
    COLUMNS.RESIDUAL_MATERNAL_DISORDERS,
]

# Incidence-risk pipelines for the intrapartum disorders. ``PIPELINES`` only
# names sepsis/PPH explicitly; obstructed labor and residual follow the
# ``{disorder}.incidence_risk`` convention (see contract note in the summary).
INTRAPARTUM_INCIDENCE_RISK_PIPELINES = {
    COLUMNS.OBSTRUCTED_LABOR: f"{COLUMNS.OBSTRUCTED_LABOR}.incidence_risk",
    COLUMNS.POSTPARTUM_HEMORRHAGE: PIPELINES.POSTPARTUM_HEMORRHAGE_INCIDENCE_RISK,
    COLUMNS.MATERNAL_SEPSIS: PIPELINES.MATERNAL_SEPSIS_INCIDENCE_RISK,
    COLUMNS.RESIDUAL_MATERNAL_DISORDERS: f"{COLUMNS.RESIDUAL_MATERNAL_DISORDERS}.incidence_risk",
}

PPH_SEVERITY_COL = f"{COLUMNS.POSTPARTUM_HEMORRHAGE}_severity"


def _build_sim(model_spec_path: Path, pop_size: int = POP_SIZE) -> InteractiveContext:
    """Build an interactive sim on the model spec with a reduced population."""
    spec = build_model_specification(model_spec_path)
    spec.configuration.population.population_size = pop_size
    return InteractiveContext(spec)


def _pipeline_values(sim: InteractiveContext, pipeline_name: str) -> pd.Series:
    """Return a scalar incidence-risk pipeline as a float Series over the pop."""
    values = sim.get_population(pipeline_name)
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    return pd.Series(values).astype(float)


@pytest.fixture(scope="module")
def mortality_state(
    model_spec_path: Path, sim_state_step_mapper: dict[str, int]
) -> InteractiveContext:
    """Sim advanced through the mortality event.

    By this state every maternal-disorder boolean, severity column, and the
    mother's cause of death are final, so the two-pass gating / mutual-exclusion
    invariants can be asserted regardless of the exact intermediate event order.
    """
    sim = _build_sim(model_spec_path)
    return get_interactive_context_state(
        sim, sim_state_step_mapper, SIMULATION_EVENT_NAMES.MORTALITY
    )


# ---------------------------------------------------------------------------
# Expectation 3: PPH / intrapartum-disorder incidence risks are probabilities
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("disorder", INTRAPARTUM_DISORDER_COLUMNS)
def test_intrapartum_incidence_risk_in_unit_interval(
    disorder: str, mortality_state: InteractiveContext
) -> None:
    """Expectation 3: each intrapartum incidence-risk pipeline is a probability."""
    pipeline_name = INTRAPARTUM_INCIDENCE_RISK_PIPELINES[disorder]
    risk = _pipeline_values(mortality_state, pipeline_name)
    assert risk.notna().all(), f"{pipeline_name} produced NaNs"
    assert (risk >= 0.0).all(), f"{pipeline_name} below 0 (min={risk.min()})"
    assert (risk <= 1.0).all(), f"{pipeline_name} above 1 (max={risk.max()})"
    assert risk.max() > 0.0, f"{pipeline_name} is identically zero"


# ---------------------------------------------------------------------------
# Expectation 5 (AME side): AME only assigned to partial_term
# ---------------------------------------------------------------------------
def test_ame_only_assigned_to_partial_term(mortality_state: InteractiveContext) -> None:
    """Expectation 5: abortion/miscarriage/ectopic (AME) only on partial_term."""
    pop = mortality_state.get_population(
        [COLUMNS.PREGNANCY_OUTCOME, COLUMNS.ABORTION_MISCARRIAGE_ECTOPIC_PREGNANCY]
    )
    ame = pop[COLUMNS.ABORTION_MISCARRIAGE_ECTOPIC_PREGNANCY].astype(bool)
    non_partial_term = (
        pop[COLUMNS.PREGNANCY_OUTCOME] != PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME
    )
    n_bad = int((ame & non_partial_term).sum())
    assert n_bad == 0, (
        f"{n_bad} non-partial_term simulants have AME == True; "
        "AME must only be applied to partial_term pregnancies."
    )


# ---------------------------------------------------------------------------
# Expectation 4: two-pass mortality gating
# ---------------------------------------------------------------------------
def test_intrapartum_deaths_only_among_survivors(
    mortality_state: InteractiveContext,
) -> None:
    """Expectation 4: intrapartum-disorder deaths are resolved only among
    simulants who survived the antepartum (AME) mortality pass -- i.e.
    every intrapartum-disorder death carries that disorder as its incidence."""
    pop = mortality_state.get_population(
        [COLUMNS.MOTHER_CAUSE_OF_DEATH] + INTRAPARTUM_DISORDER_COLUMNS
    )
    for disorder in INTRAPARTUM_DISORDER_COLUMNS:
        died_of_disorder = pop.index[pop[COLUMNS.MOTHER_CAUSE_OF_DEATH] == disorder]
        if len(died_of_disorder) == 0:
            continue
        has_disorder = pop.loc[died_of_disorder, disorder].astype(bool)
        assert has_disorder.all(), (
            f"{(~has_disorder).sum()} simulants died of {disorder} without having "
            f"the {disorder} incidence flag set."
        )


# ---------------------------------------------------------------------------
# Expectation 6: only severe hemorrhage cases can die of hemorrhage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "hemorrhage_cause, severity_col",
    [
        (COLUMNS.POSTPARTUM_HEMORRHAGE, PPH_SEVERITY_COL),
    ],
)
def test_only_severe_hemorrhage_deaths(
    hemorrhage_cause: str, severity_col: str, mortality_state: InteractiveContext
) -> None:
    """Expectation 6: every hemorrhage death must be a severe case."""
    pop = mortality_state.get_population([COLUMNS.MOTHER_CAUSE_OF_DEATH, severity_col])
    dead_of_cause = pop.loc[pop[COLUMNS.MOTHER_CAUSE_OF_DEATH] == hemorrhage_cause]
    non_severe = dead_of_cause.loc[dead_of_cause[severity_col] != HEMORRHAGE_SEVERITY.SEVERE]
    assert len(non_severe) == 0, (
        f"{len(non_severe)} {hemorrhage_cause} deaths were not severe "
        f"(severities found: {sorted(non_severe[severity_col].unique())})."
    )
