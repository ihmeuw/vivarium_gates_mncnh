from __future__ import annotations

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.state_machine import Machine, State, TransientState
from vivarium.types import ClockTime

from vivarium_gates_mncnh.constants.data_values import SIMULATION_EVENT_NAMES


class TreeMachine(Machine):
    def __init__(
        self, 
        state_column: str, 
        states: list[State], 
        initial_state = None,
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
    def __init__(self, 
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

