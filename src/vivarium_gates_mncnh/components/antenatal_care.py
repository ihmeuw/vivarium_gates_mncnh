from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData, PopulationView
from vivarium.framework.randomness.stream import _choice
from vivarium.framework.state_machine import (
    State,
    Transient,
    TransientState,
    Transition,
    TransitionSet,
)
from vivarium.types import ClockTime, DataInput

from vivarium_gates_mncnh.components.tree import DecisionTreeState, TreeMachine
from vivarium_gates_mncnh.constants.data_keys import ANC
from vivarium_gates_mncnh.constants.data_values import (
    ANC_ATTENDANCE_TYPES,
    ANC_RATES,
    COLUMNS,
    LOW_BIRTH_WEIGHT_THRESHOLD,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
    ULTRASOUND_TYPES,
)
from vivarium_gates_mncnh.utilities import get_location


class UltrasoundState(TransientState):
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


    def _next_state(
        self,
        index: pd.Index[int],
        event_time: ClockTime,
        transition_set: TransitionSet,
        population_view: PopulationView,
    ) -> None:
        if len(transition_set) == 0 or index.empty:
            return

        outputs, decisions = transition_set.choose_new_state(index)
        # these lines are the only change from vivarium because we 
        # want to use string categories instead of State objects for grouping
        str_outputs = [output.state_id for output in outputs]
        groups = pd.Series(index).groupby(
            pd.CategoricalIndex(decisions.values, categories=str_outputs), observed=False
        )
        
        if groups:
            for state_id, affected_index in sorted(groups, key=lambda x: str(x[0])):
                if state_id == "null_transition":
                    continue
                
                # Find the corresponding State object
                output = next(o for o in outputs if o.state_id == state_id)
                affected_index = pd.Index(affected_index.values)
                
                if isinstance(output, Transient):
                    if not isinstance(output, State):
                        raise ValueError(f"Invalid transition output: {output}")
                    output.transition_effect(affected_index, event_time, population_view)
                    output.next_state(affected_index, event_time, population_view)
                elif isinstance(output, State):
                    output.transition_effect(affected_index, event_time, population_view)
                else:
                    raise ValueError(f"Invalid transition output: {output}")


class ANCTransitionSet(TransitionSet):
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.propensity = builder.value.get_value(f"antenatal_care.correlated_propensity")
        self.transitions = self.get_ordered_transitions(builder, self.transitions)

    def choose_new_state(
        self, index: pd.Index[int]
    ) -> tuple[list[State | str], pd.Series[Any]]:
        """Use propensities to choose new state."""
        outputs, probabilities = zip(
            *[
                (transition.output_state, np.array(transition.probability(index)))
                for transition in self.transitions
            ]
        )
        probabilities = np.transpose(probabilities)
        outputs, probabilities = self._normalize_probabilities(outputs, probabilities)
        propensities = self.propensity(index)
        output_indexes = _choice(propensities, np.arange(4), probabilities)
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


class AntenatalCare(Component):
    @property
    def columns_created(self):
        return [
            COLUMNS.ANC_ATTENDANCE,
            COLUMNS.ULTRASOUND_TYPE,
            COLUMNS.STATED_GESTATIONAL_AGE,
            COLUMNS.SUCCESSFUL_LBW_IDENTIFICATION,
            COLUMNS.FIRST_TRIMESTER_ANC,
            COLUMNS.LATER_PREGNANCY_ANC,
        ]

    @property
    def columns_required(self):
        return [
            COLUMNS.GESTATIONAL_AGE_EXPOSURE,
            COLUMNS.BIRTH_WEIGHT_EXPOSURE,
            COLUMNS.SEX_OF_CHILD,
            COLUMNS.PREGNANCY_OUTCOME,
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
        self.lookup_tables["ANCfirst"] = self.build_lookup_table(
            builder=builder,
            data_source=builder.data.load(ANC.ANCfirst),
            value_columns=["value"],
        )
        self.lookup_tables["ANC1"] = self.build_lookup_table(
            builder=builder,
            data_source=builder.data.load(ANC.ANC1),
            value_columns=["value"],
        )
        self.lookup_tables["ANC4"] = self.build_lookup_table(
            builder=builder,
            data_source=builder.data.load(ANC.ANC4),
            value_columns=["value"],
        )
        stated_ga_standard_deviation = self.format_dict_for_lookup_table(
            ANC_RATES.STATED_GESTATIONAL_AGE_STANDARD_DEVIATION,
            COLUMNS.ULTRASOUND_TYPE,
        )
        self.lookup_tables[
            "stated_gestational_age_standard_deviation"
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

    def format_dict_for_lookup_table(self, data: dict, column: str) -> pd.DataFrame:
        series = pd.Series(data)
        return series.reset_index().rename(columns={"index": column, 0: "value"})

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        anc_data = pd.DataFrame(
            {
                COLUMNS.ANC_ATTENDANCE: ANC_ATTENDANCE_TYPES.NONE,
                COLUMNS.ULTRASOUND_TYPE: ULTRASOUND_TYPES.NO_ULTRASOUND,
                COLUMNS.STATED_GESTATIONAL_AGE: np.nan,
                COLUMNS.SUCCESSFUL_LBW_IDENTIFICATION: False,
                COLUMNS.FIRST_TRIMESTER_ANC: False,
                COLUMNS.LATER_PREGNANCY_ANC: False,
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
        pop[COLUMNS.FIRST_TRIMESTER_ANC] = pop[COLUMNS.ANC_ATTENDANCE].isin(
            [
                ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_ONLY,
                ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_AND_LATER_PREGNANCY,
            ]
        )
        pop[COLUMNS.LATER_PREGNANCY_ANC] = pop[COLUMNS.ANC_ATTENDANCE].isin(
            [
                ANC_ATTENDANCE_TYPES.LATER_PREGNANCY_ONLY,
                ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_AND_LATER_PREGNANCY,
            ]
        )

        self.population_view.update(pop)

    def _calculate_stated_gestational_age(self, pop: pd.DataFrame) -> pd.Series:
        # Apply standard deviation based on ultrasound type
        gestational_age = pop[COLUMNS.GESTATIONAL_AGE_EXPOSURE]
        measurement_errors = self.lookup_tables["stated_gestational_age_standard_deviation"](
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
        lbw_index = pop.index[pop[COLUMNS.BIRTH_WEIGHT_EXPOSURE] < LOW_BIRTH_WEIGHT_THRESHOLD]
        identification_rates = self.lookup_tables["low_birth_weight_identification_rates"](
            lbw_index
        )
        draws = self.randomness.get_draw(lbw_index, additional_key="lbw_identification")
        identification[lbw_index] = draws < identification_rates
        return identification

    def is_full_term(self, index: pd.Index) -> pd.Series:
        """Returns a boolean Series indicating if the pregnancy is full term."""
        pregnancy_outcome = self.population_view.get(index)[COLUMNS.PREGNANCY_OUTCOME]
        return pregnancy_outcome.isin(
            [
                PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME,
                PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME,
            ]
        )

    def create_anc_decision_tree(self) -> TreeMachine:
        initial_state = ANCInitialState("initial")
        # DecisionTreeStates are TransientStates that update
        # a population column value upon transition
        state_A = DecisionTreeState(
            ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_AND_LATER_PREGNANCY,
            COLUMNS.ANC_ATTENDANCE,
            ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_AND_LATER_PREGNANCY,
        )
        state_B = DecisionTreeState(
            ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_ONLY,
            COLUMNS.ANC_ATTENDANCE,
            ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_ONLY,
        )
        state_C = DecisionTreeState(
            ANC_ATTENDANCE_TYPES.LATER_PREGNANCY_ONLY,
            COLUMNS.ANC_ATTENDANCE,
            ANC_ATTENDANCE_TYPES.LATER_PREGNANCY_ONLY,
        )
        state_D = DecisionTreeState(
            ANC_ATTENDANCE_TYPES.NONE,
            COLUMNS.ANC_ATTENDANCE,
            ANC_ATTENDANCE_TYPES.NONE,
        )
        gets_ultrasound = TransientState("gets_ultrasound")
        standard_ultasound = UltrasoundState(ULTRASOUND_TYPES.STANDARD)
        ai_assisted_ultrasound = UltrasoundState(ULTRASOUND_TYPES.AI_ASSISTED)
        end_state = State("end")

        def get_a_transition_probability(index: pd.Index) -> pd.Series:
            is_full_term = self.is_full_term(index)
            result = pd.Series(0.0, index=index)
            ancfirst = self.lookup_tables["ANCfirst"](index)
            anc4 = self.lookup_tables["ANC4"](index)
            # Keep as 0 if not full term
            result[is_full_term] = np.minimum(ancfirst[is_full_term], anc4[is_full_term])
            return result

        def get_b_transition_probability(index: pd.Index) -> pd.Series:
            is_full_term = self.is_full_term(index)
            ancfirst = self.lookup_tables["ANCfirst"](index)
            anc4 = self.lookup_tables["ANC4"](index)
            result = pd.Series(ancfirst, index=index)
            # Return ANCfirst if not full term
            result[is_full_term] = ancfirst[is_full_term] - np.minimum(
                ancfirst[is_full_term], anc4[is_full_term]
            )
            return result

        def get_c_transition_probability(index: pd.Index) -> pd.Series:
            is_full_term = self.is_full_term(index)
            result = pd.Series(0.0, index=index)
            anc1 = self.lookup_tables["ANC1"](index)
            ancfirst = self.lookup_tables["ANCfirst"](index)
            # Keep as 0 if not full term
            result[is_full_term] = anc1[is_full_term] - ancfirst[is_full_term]
            return result

        def get_d_transition_probability(index: pd.Index) -> pd.Series:
            is_full_term = self.is_full_term(index)
            anc1 = self.lookup_tables["ANC1"](index)
            ancfirst = self.lookup_tables["ANCfirst"](index)
            result = pd.Series(0.0, index=index)
            result[is_full_term] = 1 - anc1[is_full_term]
            result[~is_full_term] = 1 - ancfirst[~is_full_term]
            return result

        # Decisions
        # Initial state transitions to A, B, C, or D
        initial_state.add_transition(
            output_state=state_A,
            probability_function=get_a_transition_probability,
        )
        initial_state.add_transition(
            output_state=state_B,
            probability_function=get_b_transition_probability,
        )
        initial_state.add_transition(
            output_state=state_C,
            probability_function=get_c_transition_probability,
        )
        initial_state.add_transition(
            output_state=state_D,
            probability_function=get_d_transition_probability,
        )

        # State A, B, C, D transitions to gets_ultrasound or end_state
        state_A.add_transition(
            output_state=gets_ultrasound,
            probability_function=lambda index: pd.Series(
                ANC_RATES.RECEIVED_ULTRASOUND[self.location], index=index
            ),
        )
        state_A.add_transition(
            output_state=end_state,
            probability_function=lambda index: pd.Series(
                1 - ANC_RATES.RECEIVED_ULTRASOUND[self.location],
                index=index,
            ),
        )
        state_B.add_transition(
            output_state=gets_ultrasound,
            probability_function=lambda index: pd.Series(
                ANC_RATES.RECEIVED_ULTRASOUND[self.location], index=index
            ),
        )
        state_B.add_transition(
            output_state=end_state,
            probability_function=lambda index: pd.Series(
                1 - ANC_RATES.RECEIVED_ULTRASOUND[self.location],
                index=index,
            ),
        )
        state_C.add_transition(
            output_state=gets_ultrasound,
            probability_function=lambda index: pd.Series(
                ANC_RATES.RECEIVED_ULTRASOUND[self.location], index=index
            ),
        )
        state_C.add_transition(
            output_state=end_state,
            probability_function=lambda index: pd.Series(
                1 - ANC_RATES.RECEIVED_ULTRASOUND[self.location],
                index=index,
            ),
        )
        state_D.add_transition(output_state=end_state)

        # Determine ultrasound type
        gets_ultrasound.add_transition(
            output_state=standard_ultasound,
            probability_function=lambda index: pd.Series(
                ANC_RATES.ULTRASOUND_TYPE[self.location][ULTRASOUND_TYPES.STANDARD],
                index=index,
            ),
        )
        gets_ultrasound.add_transition(
            output_state=ai_assisted_ultrasound,
            probability_function=lambda index: pd.Series(
                ANC_RATES.ULTRASOUND_TYPE[self.location][ULTRASOUND_TYPES.AI_ASSISTED],
                index=index,
            ),
        )
        standard_ultasound.add_transition(output_state=end_state)
        ai_assisted_ultrasound.add_transition(output_state=end_state)

        return TreeMachine(
            COLUMNS.ANC_STATE,
            [
                initial_state,
                state_A,
                state_B,
                state_C,
                state_D,
                gets_ultrasound,
                standard_ultasound,
                ai_assisted_ultrasound,
                end_state,
            ],
            initial_state,
            time_step_name=SIMULATION_EVENT_NAMES.PREGNANCY,
        )
