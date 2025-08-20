from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.state_machine import (
    Machine,
    State,
    TransientState,
    Transition,
    TransitionSet,
)
from vivarium.types import ClockTime

from vivarium_gates_mncnh.constants.data_values import SIMULATION_EVENT_NAMES

if TYPE_CHECKING:
    from vivarium.types import DataInput


class TreeMachine(Machine):
    def __init__(
        self,
        state_column: str,
        states: list[State],
        initial_state=None,
        time_step_name: str = "",
    ):
        super().__init__(state_column, states, initial_state)
        # Time step name where the simulants will go through the decision tree
        self._time_step_trigger = time_step_name

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self._sim_step_name = builder.time.simulation_event_name()

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() == self._time_step_trigger:
            super().on_time_step(event)


class DecisionTreeState(TransientState):
    def __init__(
        self,
        state_id: str,
        update_col: str,
        update_value: str | bool,
    ) -> None:
        super().__init__(state_id)
        self.update_column = update_col
        self.update_value = update_value

    @property
    def columns_required(self) -> list[str]:
        return [self.update_column]

    def transition_side_effect(self, index: pd.Index, _event_time: ClockTime) -> None:
        pop = self.population_view.get(index)
        pop[self.update_column] = self.update_value
        self.population_view.update(pop)


class ANCInitialState(State):
    def __init__(
        self,
        state_id: str,
        allow_self_transition: bool = False,
        initialization_weights: DataInput = 0.0,
    ) -> None:
        super().__init__(state_id)
        self.transition_set = ANCTransitionSet(
            self.state_id, allow_self_transition=allow_self_transition
        )
        self._sub_components = [self.transition_set]


class ANCTransitionSet(TransitionSet):
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.propensity = builder.value.get_value(f"antenatal_care.correlated_propensity")
        self.transitions = self.get_ordered_transitions(builder, self.transitions)

    def choose_new_state(
        self, index: pd.Index[int]
    ) -> tuple[list[State | str], pd.Series[Any]]:
        """Use propensities to choose new state. All changes are mentioned in comments, ie
        code without an associated comment is unchanged from the original method."""
        outputs, probabilities = zip(
            *[
                (transition.output_state, np.array(transition.probability(index)))
                for transition in self.transitions
            ]
        )
        probabilities = np.transpose(probabilities)
        outputs, probabilities = self._normalize_probabilities(outputs, probabilities)
        cdf_probabilities = [np.cumsum(probs) for probs in probabilities]
        propensities = self.propensity(index)
        # find index of first cumulative probability that is greater than propensity
        output_indexes = [
            np.searchsorted(cdf, propensity, side="right")
            for cdf, propensity in zip(cdf_probabilities, propensities)
        ]
        return outputs, pd.Series([outputs[i].state_id for i in output_indexes])

    def get_ordered_transitions(
        self, builder: Builder, transitions: list[Transition]
    ) -> list[Transition]:
        transition_ordering = builder.configuration.propensity_ordering.antenatal_care
        order_map = {state_id: i for i, state_id in enumerate(transition_ordering)}
        return sorted(
            self.transitions,
            key=lambda transition: order_map[transition.output_state.state_id],
        )
