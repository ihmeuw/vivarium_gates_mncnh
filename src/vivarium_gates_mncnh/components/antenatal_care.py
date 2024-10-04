from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import scipy.stats as stats
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.state_machine import Machine, State, Transient, Transition
from vivarium.types import ClockTime

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import (
    ANC_RATES,
    COLUMNS,
    LOW_BIRTH_WEIGHT_THRESHOLD,
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
    def add_decision(
        self,
        output_state: State,
        decision_function: Callable[[pd.Index], pd.Series] = lambda index: pd.Series(
            1.0, index=index
        ),
    ) -> None:
        transition = Transition(self, output_state, decision_function)
        self.add_transition(transition)


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


class UltrasoundState(TransientDecisionTreeState):
    def __init__(self, ultrasound_type: str) -> None:
        super().__init__(f"{ultrasound_type}_ultrasound")
        self.ultrasound_type = ultrasound_type

    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.ULTRASOUND_TYPE]

    def transition_side_effect(self, index: pd.Index, _event_time: ClockTime) -> None:
        pop = self.population_view.get(index)
        pop[COLUMNS.ULTRASOUND_TYPE] = self.ultrasound_type
        self.population_view.update(pop)


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
        return [self.decision_tree]

    def __init__(self) -> None:
        super().__init__()
        self.decision_tree = self.create_anc_decision_tree()

    def setup(self, builder: Builder):
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = get_location(builder)

    def build_all_lookup_tables(self, builder: Builder) -> None:
        # TODO: update data key to constant
        anc_attendance_probability = builder.data.load(
            "covariate.antenatal_care_1_visit_coverage_proportion.estimate"
        )
        self.lookup_tables["anc_attendance_probability"] = self.build_lookup_table(
            builder=builder,
            data_source=anc_attendance_probability,
            value_columns=["value"],
        )
        stated_ga_standard_deviation = self.format_dict_for_lookup_table(
            ANC_RATES.STATED_GESTATIONAL_AGE_STANDARD_DEVIATION,
            COLUMNS.ULTRASOUND_TYPE,
        )
        self.lookup_tables[
            "stated_gestational_age_standa_deviation"
        ] = self.build_lookup_table(
            builder=builder,
            data_source=stated_ga_standard_deviation,
            value_columns=["value"],
        )
        lbw_identification_rates = self.format_dict_for_lookup_table(
            ANC_RATES.SUCCESSFUL_LBW_IDENTIFICATION,
            COLUMNS.ULTRASOUND_TYPE,
        )
        self.lookup_tables["low_birth_weight_identification_rates"] = self.build_lookup_table(
            builder=builder,
            data_source=lbw_identification_rates,
            value_columns=["value"],
        )

    def get_anc_attendance_rate(self, index: pd.Index) -> pd.Series:
        return self.lookup_tables["anc_attendance_probability"](index)

    def format_dict_for_lookup_table(self, data: dict, column: str) -> pd.DataFrame:
        series = pd.Series(data)
        return series.reset_index().rename(columns={"index": column, 0: "value"})

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        anc_data = pd.DataFrame(
            {
                COLUMNS.ATTENDED_CARE_FACILITY: False,
                COLUMNS.ULTRASOUND_TYPE: ULTRASOUND_TYPES.NO_ULTRASOUND,
                COLUMNS.STATED_GESTATIONAL_AGE: np.nan,
                COLUMNS.SUCCESSFUL_LBW_IDENTIFICATION: False,
            },
            index=pop_data.index,
        )

        self.population_view.update(anc_data)

    def on_time_step_cleanup(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.PREGNANCY:
            return
        pop = self.population_view.get(event.index)
        pop[COLUMNS.STATED_GESTATIONAL_AGE] = self._calculate_stated_gestational_age(pop)
        pop[COLUMNS.SUCCESSFUL_LBW_IDENTIFICATION] = self._determine_lbw_identification(pop)

        self.population_view.update(pop)

    def _calculate_stated_gestational_age(self, pop: pd.DataFrame) -> pd.Series:
        # Apply standard deviation based on ultrasound type
        gestational_age = pop[COLUMNS.GESTATIONAL_AGE]
        measurement_errors = self.lookup_tables["stated_gestational_age_standa_deviation"](
            pop.index
        )
        measurement_error_draws = self.randomness.get_draw(
            pop.index, additional_key="measurement_error"
        )
        return stats.norm.ppf(
            measurement_error_draws, loc=gestational_age, scale=measurement_errors
        )

    def _determine_lbw_identification(self, pop: pd.DataFrame) -> pd.Series:
        identification = pd.Series(False, index=pop.index)
        lbw_index = pop.index[pop[COLUMNS.BIRTH_WEIGHT] < LOW_BIRTH_WEIGHT_THRESHOLD]
        identification_rates = self.lookup_tables["low_birth_weight_identification_rates"](
            lbw_index
        )
        draws = self.randomness.get_draw(lbw_index, additional_key="lbw_identification")
        identification[lbw_index] = draws < identification_rates
        return identification

    def create_anc_decision_tree(self) -> Machine:
        initial_state = DecisionTreeState("initial")
        attended_antental_care = ANCState()
        gets_ultrasound = TransientDecisionTreeState("gets_ultrasound")
        standard_ultasound = UltrasoundState(ULTRASOUND_TYPES.STANDARD)
        ai_assisted_ultrasound = UltrasoundState(ULTRASOUND_TYPES.AI_ASSISTED)
        end_state = DecisionTreeState("end")

        # Decisions
        initial_state.add_decision(
            attended_antental_care,
            # TODO: this data will need to be updated when the ANC artifact is finished
            self.get_anc_attendance_rate,
        )
        initial_state.add_decision(
            end_state, lambda index: 1 - self.get_anc_attendance_rate(index)
        )
        attended_antental_care.add_decision(
            gets_ultrasound,
            lambda index: pd.Series(
                ANC_RATES.RECEIVED_ULTRASOUND[self.location], index=index
            ),
        )
        attended_antental_care.add_decision(
            end_state,
            lambda index: pd.Series(
                1 - ANC_RATES.RECEIVED_ULTRASOUND[self.location],
                index=index,
            ),
        )
        gets_ultrasound.add_decision(
            standard_ultasound,
            lambda index: pd.Series(
                ANC_RATES.ULTRASOUND_TYPE[self.location][ULTRASOUND_TYPES.STANDARD],
                index=index,
            ),
        )
        gets_ultrasound.add_decision(
            ai_assisted_ultrasound,
            lambda index: pd.Series(
                ANC_RATES.ULTRASOUND_TYPE[self.location][ULTRASOUND_TYPES.AI_ASSISTED],
                index=index,
            ),
        )
        standard_ultasound.add_decision(end_state)
        ai_assisted_ultrasound.add_decision(end_state)

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