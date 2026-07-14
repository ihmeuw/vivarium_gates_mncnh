"""Regression test for SepsisEffectsOnHemoglobin postpartum-period timing.

V&V criterion (MNCNH concept model): the maternal-sepsis hemoglobin shift is
applied during the mother's postpartum period, using the early-postpartum
(0-6 week) magnitude during the ``early_postpartum`` step and the
late-postpartum (6-39 week) magnitude during the ``late_postpartum`` step.

Regression guard: an earlier version keyed the shift on the neonatal-mortality
steps (early/late neonatal), so the late-postpartum shift was never reproduced
and the early shift landed on the wrong timeline. See
tests/model_notebooks/interactive/interactive_simulation_sepsis_on_hemoglobin.ipynb.

These tests read the event ordering straight from the model spec (rather than a
hardcoded step mapper) so they stay correct if the event list changes.
"""

from pathlib import Path

import pytest
from layered_config_tree import LayeredConfigTree
from vivarium import InteractiveContext

from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    PIPELINES,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
)

SEPSIS_COMPONENT_NAME = "sepsis_effects_on_hemoglobin"


def _state_at(model_spec_path: Path, event_name: str) -> InteractiveContext:
    """Return a sim stepped to ``event_name`` using the real simulation_events order."""
    config = LayeredConfigTree(model_spec_path)
    events = list(config.configuration.time.simulation_events)
    sim = InteractiveContext(model_spec_path)
    sim.take_steps(events.index(event_name) + 1)
    return sim


@pytest.fixture(scope="module")
def reference_state(model_spec_path: Path) -> InteractiveContext:
    # Intrapartum mortality step: the postpartum shift is not yet applied, so this
    # is the unshifted baseline for each sepsis simulant.
    return _state_at(model_spec_path, SIMULATION_EVENT_NAMES.MORTALITY)


@pytest.fixture(scope="module")
def early_postpartum_state(model_spec_path: Path) -> InteractiveContext:
    return _state_at(model_spec_path, SIMULATION_EVENT_NAMES.EARLY_POSTPARTUM)


@pytest.fixture(scope="module")
def late_postpartum_state(model_spec_path: Path) -> InteractiveContext:
    return _state_at(model_spec_path, SIMULATION_EVENT_NAMES.LATE_POSTPARTUM)


def _sepsis_live_birth_idx(sim: InteractiveContext):
    pop = sim.get_population([COLUMNS.MATERNAL_SEPSIS, COLUMNS.PREGNANCY_OUTCOME])
    live_birth = pop[COLUMNS.PREGNANCY_OUTCOME] == PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME
    return pop.index[live_birth & pop[COLUMNS.MATERNAL_SEPSIS]]


def _applied_shift(
    reference_state: InteractiveContext, postpartum_state: InteractiveContext, idx
) -> float:
    """Mean per-simulant hemoglobin change from the reference step to a postpartum step.

    A deterministic seed means the two sims share an identical cohort and base
    hemoglobin, so the per-simulant difference isolates exactly the sepsis shift
    the component applied at the postpartum step.
    """
    hgb_ref = reference_state.get_population(PIPELINES.HEMOGLOBIN_EXPOSURE).loc[idx]
    hgb_pp = postpartum_state.get_population(PIPELINES.HEMOGLOBIN_EXPOSURE).loc[idx]
    return (hgb_pp - hgb_ref).mean()


@pytest.mark.parametrize(
    "postpartum_fixture, shift_attr",
    [
        ("early_postpartum_state", "early_postpartum_shift"),
        ("late_postpartum_state", "late_postpartum_shift"),
    ],
)
def test_sepsis_shift_matches_period_target(
    postpartum_fixture: str,
    shift_attr: str,
    reference_state: InteractiveContext,
    request: pytest.FixtureRequest,
) -> None:
    """The shift applied to sepsis simulants in each postpartum period equals that
    period's target (the value the component loaded from the artifact)."""
    postpartum_state: InteractiveContext = request.getfixturevalue(postpartum_fixture)

    sepsis_idx = _sepsis_live_birth_idx(reference_state)
    if len(sepsis_idx) == 0:
        pytest.skip("No live-birth simulants with maternal sepsis in this draw")

    applied_shift = _applied_shift(reference_state, postpartum_state, sepsis_idx)
    target = getattr(postpartum_state.get_component(SEPSIS_COMPONENT_NAME), shift_attr)

    assert applied_shift == pytest.approx(target, abs=1e-6), (
        f"Sepsis hemoglobin shift at {postpartum_fixture} was {applied_shift:.4f}, "
        f"expected the {shift_attr} target {target:.4f}."
    )


def test_early_and_late_postpartum_shifts_are_distinct(
    reference_state: InteractiveContext,
    early_postpartum_state: InteractiveContext,
    late_postpartum_state: InteractiveContext,
) -> None:
    """Core regression guard: the late-postpartum shift is genuinely applied and
    differs from the early one (the pre-fix model never reproduced the late shift)."""
    sepsis_idx = _sepsis_live_birth_idx(reference_state)
    if len(sepsis_idx) == 0:
        pytest.skip("No live-birth simulants with maternal sepsis in this draw")

    early = _applied_shift(reference_state, early_postpartum_state, sepsis_idx)
    late = _applied_shift(reference_state, late_postpartum_state, sepsis_idx)

    assert early != pytest.approx(late, abs=1e-6), (
        f"Early ({early:.4f}) and late ({late:.4f}) postpartum shifts should differ; "
        "the late-postpartum period is not being applied distinctly."
    )
