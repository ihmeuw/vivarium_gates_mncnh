from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.state_machine import Machine, State, Transient, Transition
from vivarium.types import ClockTime

from vivarium_gates_mncnh.constants.data_values import (
    ANC_RATES,
    COLUMNS,
    SIMULATION_EVENT_NAMES,
    ULTRASOUND_TYPES,
)
from vivarium_gates_mncnh.utilities import get_location


class TreeMachine(Machine):
    @property
    def columns_created(self) -> list[str]:
        return [self.state_column]

    @property
    def columns_required(self) -> list[str] | None:
        return None

    def __init__(self, state_column: str, states: list[State], initial_state: State) -> None:
        super().__init__(state_column, states)
        self.initial_state = initial_state

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self._sim_step_name = builder.time.simulation_event_name()

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.update(
            pd.Series(
                self.initial_state.state_id, index=pop_data.index, name=self.state_column
            ),
        )

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() == SIMULATION_EVENT_NAMES.PREGNANCY:
            self.transition(event.index, event.time)


class DecisionTreeState(State):
    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.location = get_location(builder)

    def add_decision(
        self,
        output_state: State,
        decision_function: Callable[[pd.Index], pd.Series],
    ) -> None:
        transition = Transition(
            self, output_state, self.get_probability_function(decision_function)
        )
        self.add_transition(transition)

    def get_probability_function(
        self, decision_function: Callable[[pd.Index], pd.Series]
    ) -> Callable[[pd.Index], pd.Series]:
        """We need to check the simulation step name within the probability function, because it will change"""

        def pf(index: pd.Index) -> pd.Series:
            if self._sim_step_name() != SIMULATION_EVENT_NAMES.PREGNANCY:
                return lambda index: pd.Series(0.0, index=index)
            return decision_function(index)

        return pf


class TransientDecisionTreeState(DecisionTreeState, Transient):
    pass


class ANCState(TransientDecisionTreeState):
    def __init__(self) -> None:
        super().__init__("attended_antental_care")

    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.ATTENDED_CARE_FACILITY]

    def transition_side_effect(self, index: pd.Index, _event_time: ClockTime) -> None:
        pop = self.population_view.get(index)
        pop[COLUMNS.ATTENDED_CARE_FACILITY] = True
        self.population_view.update(pop)


class StandardUltrasound(TransientDecisionTreeState):
    def __init__(self) -> None:
        super().__init__("standard_ultasound")

    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.ULTRASOUND_TYPE]

    def transition_side_effect(self, index: pd.Index, _event_time: ClockTime) -> None:
        pop = self.population_view.get(index)
        pop[COLUMNS.ULTRASOUND_TYPE] = ULTRASOUND_TYPES.STANDARD
        self.population_view.update(pop)


class AIAssistedUltrasound(TransientDecisionTreeState):
    def __init__(self) -> None:
        super().__init__("ai_assisted_ultrasound")

    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.ULTRASOUND_TYPE]

    def transition_side_effect(self, index: pd.Index, _event_time: ClockTime) -> None:
        pop = self.population_view.get(index)
        pop[COLUMNS.ULTRASOUND_TYPE] = ULTRASOUND_TYPES.AI_ASSISTED
        self.population_view.update(pop)


def ANC() -> Machine:
    initial_state = DecisionTreeState("initial")
    attended_antental_care = ANCState()
    gets_ultrasound = TransientDecisionTreeState("gets_ultrasound")
    standard_ultasound = StandardUltrasound()
    ai_assisted_ultrasound = AIAssistedUltrasound()
    end_state = DecisionTreeState("end")

    # Decisions
    initial_state.add_decision(
        attended_antental_care,
        lambda index: pd.Series(
            ANC_RATES.ATTENDED_CARE_FACILITY[initial_state.location], index=index
        ),
    )
    initial_state.add_decision(
        end_state,
        lambda index: pd.Series(
            1 - ANC_RATES.ATTENDED_CARE_FACILITY[initial_state.location], index=index
        ),
    )
    attended_antental_care.add_decision(
        gets_ultrasound,
        lambda index: pd.Series(
            ANC_RATES.RECEIVED_ULTRASOUND[attended_antental_care.location], index=index
        ),
    )
    attended_antental_care.add_decision(
        end_state,
        lambda index: pd.Series(
            1 - ANC_RATES.RECEIVED_ULTRASOUND[attended_antental_care.location], index=index
        ),
    )
    gets_ultrasound.add_decision(
        standard_ultasound,
        lambda index: pd.Series(
            ANC_RATES.ULTRASOUND_TYPE[ULTRASOUND_TYPES.STANDARD], index=index
        ),
    )
    gets_ultrasound.add_decision(
        ai_assisted_ultrasound,
        lambda index: pd.Series(
            1 - ANC_RATES.ULTRASOUND_TYPE[ULTRASOUND_TYPES.AI_ASSISTED], index=index
        ),
    )
    standard_ultasound.add_decision(end_state, lambda index: pd.Series(1.0, index=index))
    ai_assisted_ultrasound.add_decision(end_state, lambda index: pd.Series(1.0, index=index))

    return TreeMachine(
        "anc_state",
        [
            initial_state,
            attended_antental_care,
            gets_ultrasound,
            standard_ultasound,
            ai_assisted_ultrasound,
            end_state,
        ],
        initial_state,
    )


class AntenatalCare(Component):
    @property
    def columns_created(self):
        return [
            COLUMNS.ATTENDED_CARE_FACILITY,
            COLUMNS.ULTRASOUND_TYPE,
            COLUMNS.STATED_GESTATIONAL_AGE,
            COLUMNS.SUCCESSFUL_LBW_IDENTIFICATION,
        ]

    @property
    def columns_required(self):
        return [
            COLUMNS.GESTATIONAL_AGE,
            COLUMNS.BIRTH_WEIGHT,
            COLUMNS.SEX_OF_CHILD,
        ]

    @property
    def sub_components(self) -> list[Component]:
        return [self.machine]

    def __init__(self) -> None:
        super().__init__()
        self.machine = ANC()

    def setup(self, builder: Builder):
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = get_location(builder)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        anc_data = pd.DataFrame(
            {
                COLUMNS.ATTENDED_CARE_FACILITY: False,
                COLUMNS.ULTRASOUND_TYPE: ULTRASOUND_TYPES.NO_ULTRASOUND,
                COLUMNS.STATED_GESTATIONAL_AGE: np.nan,
                COLUMNS.SUCCESSFUL_LBW_IDENTIFICATION: np.nan,
            },
            index=pop_data.index,
        )
        self.population_view.update(anc_data)

    def on_time_step_cleanup(self, event: Event) -> None:
        # TODO: Add columns of stated gestational age and successful lbw identification
        pass
