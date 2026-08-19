from pathlib import Path

import pandas as pd
import pytest
from vivarium.engine import InteractiveContext

from vivarium_gates_mncnh.constants.data_values import PIPELINES, SIMULATION_EVENT_NAMES
from vivarium_gates_mncnh.plugins.time import EventClock

SIMULATION_EVENTS = [
    SIMULATION_EVENT_NAMES.EARLY_POSTPARTUM,
    SIMULATION_EVENT_NAMES.LATE_POSTPARTUM,
]


def make_clock(step_index: int) -> EventClock:
    """An EventClock parked on a given step index, without a full sim setup."""
    clock = EventClock()
    clock.simulation_events = SIMULATION_EVENTS
    clock.step_index = step_index
    return clock


@pytest.mark.parametrize(
    "step_index, expected",
    [
        (0, SIMULATION_EVENT_NAMES.EARLY_POSTPARTUM),
        (1, SIMULATION_EVENT_NAMES.LATE_POSTPARTUM),
    ],
)
def test_step_name_on_an_event(step_index: int, expected: str) -> None:
    assert make_clock(step_index).step_name == expected


@pytest.mark.parametrize("step_index", [2, 3, -1])
def test_step_name_off_the_end_of_the_events(step_index: int) -> None:
    """Off-list step indices are None rather than an IndexError or a wrong event.

    step_forward increments past the final event when a sim runs to completion,
    and step_backward can take the index below zero -- where a negative index
    would otherwise wrap around and silently report an event from the end of
    the list.
    """
    assert make_clock(step_index).step_name is None


def test_stepping_past_the_final_event_raises() -> None:
    """A finished sim refuses to step rather than stepping on as a no-op.

    Code that hunts for an event with `while step_name != target: sim.step()`
    used to die on the IndexError once it ran off the end. Now that step_name
    is None there, such a loop would spin forever if the clock let it keep
    stepping, so the refusal has to come from step_forward instead.
    """
    clock = make_clock(len(SIMULATION_EVENTS))
    with pytest.raises(IndexError, match="already run through all"):
        clock.step_forward(pd.Index([]))


def test_hemoglobin_exposure_is_queryable_at_the_end_of_the_sim(
    model_spec_path: Path,
) -> None:
    """Pipelines can still be queried once the sim has run to completion.

    The hemoglobin exposure pipeline is mutated by step-gated components that
    ask the clock for the current event name, so querying it in this state used
    to raise an IndexError out of EventClock.step_name.

    The clock is advanced by hand rather than by stepping the sim: one step per
    simulation event is what the run would do, which the first assert pins down,
    and skipping the steps keeps this off the slow path. The end-to-end version
    is the last cell of tests/model_notebooks/interactive/
    interactive_simulation_sepsis_on_hemoglobin.ipynb.
    """
    sim = InteractiveContext(model_spec_path)
    clock = sim._clock

    assert sim.get_number_of_steps_remaining() == len(clock.simulation_events)
    clock.step_index = len(clock.simulation_events)
    assert clock.step_name is None

    exposure = sim.get_population(PIPELINES.HEMOGLOBIN_EXPOSURE)
    assert not exposure.empty
    assert exposure.notna().all()

    with pytest.raises(IndexError, match="already run through all"):
        sim.step()
