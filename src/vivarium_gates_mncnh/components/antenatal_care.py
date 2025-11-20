from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness.stream import _choice
from vivarium.framework.state_machine import (
    State,
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
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
    ULTRASOUND_TYPES,
)
from vivarium_gates_mncnh.constants.scenarios import INTERVENTION_SCENARIOS
from vivarium_gates_mncnh.utilities import get_location


class ANCAttendance(Component):
    @property
    def columns_created(self):
        return [COLUMNS.ANC_ATTENDANCE]

    @property
    def columns_required(self):
        return [COLUMNS.PREGNANCY_OUTCOME]

    def setup(self, builder: Builder):
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.propensity = builder.value.get_value(f"antenatal_care.correlated_propensity")

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        anc_data = pd.DataFrame(
            {
                COLUMNS.ANC_ATTENDANCE: pd.NA,
            },
            index=pop_data.index,
        )
        self.population_view.update(anc_data)

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

    def is_full_term(self, index: pd.Index) -> pd.Series:
        """Returns a boolean Series indicating if the pregnancy is full term."""
        pregnancy_outcome = self.population_view.get(index)[COLUMNS.PREGNANCY_OUTCOME]
        return pregnancy_outcome.isin(
            [
                PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME,
                PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME,
            ]
        )

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.FIRST_TRIMESTER_ANC:
            return

        def get_both_visits_probability(index: pd.Index) -> pd.Series:
            is_full_term = self.is_full_term(index)
            result = pd.Series(0.0, index=index)
            ancfirst = self.lookup_tables["ANCfirst"](index)
            anc4 = self.lookup_tables["ANC4"](index)
            # Keep as 0 if not full term
            result[is_full_term] = np.minimum(ancfirst[is_full_term], anc4[is_full_term])
            return result

        def get_early_visit_only_probability(index: pd.Index) -> pd.Series:
            is_full_term = self.is_full_term(index)
            ancfirst = self.lookup_tables["ANCfirst"](index)
            anc4 = self.lookup_tables["ANC4"](index)
            result = pd.Series(ancfirst, index=index)
            # Return ANCfirst if not full term
            result[is_full_term] = ancfirst[is_full_term] - np.minimum(
                ancfirst[is_full_term], anc4[is_full_term]
            )
            return result

        def get_later_visit_only_probability(index: pd.Index) -> pd.Series:
            is_full_term = self.is_full_term(index)
            result = pd.Series(0.0, index=index)
            anc1 = self.lookup_tables["ANC1"](index)
            ancfirst = self.lookup_tables["ANCfirst"](index)
            # Keep as 0 if not full term
            result[is_full_term] = anc1[is_full_term] - ancfirst[is_full_term]
            return result

        def get_no_visit_probability(index: pd.Index) -> pd.Series:
            is_full_term = self.is_full_term(index)
            anc1 = self.lookup_tables["ANC1"](index)
            ancfirst = self.lookup_tables["ANCfirst"](index)
            result = pd.Series(0.0, index=index)
            result[is_full_term] = 1 - anc1[is_full_term]
            result[~is_full_term] = 1 - ancfirst[~is_full_term]
            return result

        # this ordering is important when using propensities
        # to preserve correlations in facility choice model
        probabilities = pd.concat(
            [
                get_no_visit_probability(event.index),
                get_later_visit_only_probability(event.index),
                get_early_visit_only_probability(event.index),
                get_both_visits_probability(event.index),
            ],
            axis=1,
        )
        probabilities.columns = [
            ANC_ATTENDANCE_TYPES.NONE,
            ANC_ATTENDANCE_TYPES.LATER_PREGNANCY_ONLY,
            ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_ONLY,
            ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_AND_LATER_PREGNANCY,
        ]

        # use correlated propensity to decide ANC attendance
        propensities = self.propensity(event.index)
        anc_choices = probabilities.columns
        anc_attendance = _choice(propensities, anc_choices, probabilities.values)
        anc_attendance.name = COLUMNS.ANC_ATTENDANCE

        self.population_view.update(anc_attendance)


class Ultrasound(Component):
    @property
    def columns_created(self):
        return [
            COLUMNS.ULTRASOUND_TYPE,
            COLUMNS.STATED_GESTATIONAL_AGE,
        ]

    @property
    def columns_required(self):
        return [COLUMNS.ANC_ATTENDANCE, COLUMNS.GESTATIONAL_AGE_EXPOSURE]

    def setup(self, builder: Builder):
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = get_location(builder)
        self.scenario = INTERVENTION_SCENARIOS[builder.configuration.intervention.scenario]

    def build_all_lookup_tables(self, builder: Builder) -> None:
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

    def format_dict_for_lookup_table(self, data: dict, column: str) -> pd.DataFrame:
        series = pd.Series(data)
        return series.reset_index().rename(columns={"index": column, 0: "value"})

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        initial_data = pd.DataFrame(
            {
                COLUMNS.ULTRASOUND_TYPE: pd.NA,
                COLUMNS.STATED_GESTATIONAL_AGE: np.nan,
            },
            index=pop_data.index,
        )
        self.population_view.update(initial_data)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.ULTRASOUND:
            return

        def get_ultrasound_probability() -> float:
            # ultrasound coverage is either full or baseline
            if self.scenario.ultrasound_coverage == "full":
                ultrasound_prob = 1.0
            else:
                ultrasound_prob = ANC_RATES.RECEIVED_ULTRASOUND[self.location]
            return ultrasound_prob

        def get_standard_ultrasound_probability() -> float:
            if self.scenario.standard_ultrasound_coverage == "none":
                standard_probability = 0.0
            elif self.scenario.standard_ultrasound_coverage == "half":
                standard_probability = 0.5
            elif self.scenario.standard_ultrasound_coverage == "full":
                standard_probability = 1.0
            else:
                standard_probability = ANC_RATES.ULTRASOUND_TYPE[self.location][
                    ULTRASOUND_TYPES.STANDARD
                ]
            return standard_probability

        pop = self.population_view.get(event.index)
        anc_pop = pop.loc[pop[COLUMNS.ANC_ATTENDANCE] != ANC_ATTENDANCE_TYPES.NONE]

        # determine who gets ultrasound
        ultrasound_probability = get_ultrasound_probability()
        gets_ultrasound = self.randomness.choice(
            anc_pop.index,
            choices=[True, False],
            p=[ultrasound_probability, 1 - ultrasound_probability],
            additional_key="gets_ultrasound",
        )

        # determine type of ultrasound for those who get it
        standard_ultrasound_probability = get_standard_ultrasound_probability()
        ultrasound_pop = anc_pop.loc[gets_ultrasound]
        ultrasound_type = self.randomness.choice(
            ultrasound_pop.index,
            choices=[ULTRASOUND_TYPES.STANDARD, ULTRASOUND_TYPES.AI_ASSISTED],
            p=[standard_ultrasound_probability, 1 - standard_ultrasound_probability],
            additional_key="ultrasound_type",
        )

        pop[COLUMNS.ULTRASOUND_TYPE] = ULTRASOUND_TYPES.NO_ULTRASOUND
        pop.loc[ultrasound_pop.index, COLUMNS.ULTRASOUND_TYPE] = ultrasound_type

        def calculate_stated_gestational_age(pop: pd.DataFrame) -> pd.Series:
            # Apply standard deviation based on ultrasound type
            gestational_age = pop[COLUMNS.GESTATIONAL_AGE_EXPOSURE]
            measurement_errors = self.lookup_tables[
                "stated_gestational_age_standard_deviation"
            ](pop.index)
            measurement_error_draws = self.randomness.get_draw(
                pop.index, additional_key="measurement_error"
            )
            return stats.norm.ppf(
                measurement_error_draws, loc=gestational_age, scale=measurement_errors
            )

        pop[COLUMNS.STATED_GESTATIONAL_AGE] = calculate_stated_gestational_age(pop)

        self.population_view.update(pop)
