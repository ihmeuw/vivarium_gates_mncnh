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
    PIPELINES,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
    ULTRASOUND_TYPES,
)
from vivarium_gates_mncnh.constants.scenarios import INTERVENTION_SCENARIOS
from vivarium_gates_mncnh.utilities import get_location


class ANCAttendance(Component):
    @property
    def columns_created(self):
        return [
            COLUMNS.ANC_ATTENDANCE,
            COLUMNS.TIME_OF_FIRST_ANC_VISIT,
            COLUMNS.TIME_OF_LATER_ANC_VISIT,
        ]

    @property
    def columns_required(self):
        return [COLUMNS.PREGNANCY_OUTCOME]

    def setup(self, builder: Builder):
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.propensity = builder.value.get_value(f"antenatal_care.correlated_propensity")
        self.pregnancy_duration = builder.value.get_value(PIPELINES.PREGNANCY_DURATION)
        self.gestational_age = builder.value.get_value(PIPELINES.GESTATIONAL_AGE_EXPOSURE)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        anc_data = pd.DataFrame(
            {
                COLUMNS.ANC_ATTENDANCE: pd.NA,
                COLUMNS.TIME_OF_FIRST_ANC_VISIT: pd.Series(
                    pd.NaT, index=pop_data.index, dtype="timedelta64[ns]"
                ),
                COLUMNS.TIME_OF_LATER_ANC_VISIT: pd.Series(
                    pd.NaT, index=pop_data.index, dtype="timedelta64[ns]"
                ),
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
        return pregnancy_outcome == PREGNANCY_OUTCOMES.FULL_TERM_OUTCOME

    def on_time_step(self, event: Event) -> None:
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

        if self._sim_step_name() == SIMULATION_EVENT_NAMES.FIRST_TRIMESTER_ANC:
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

            # determine timing of first ANC visits for those who attend
            time_of_first_visit = self._calculate_first_visit_timing(event, anc_attendance)
            self.population_view.update(time_of_first_visit)

        # determine timing of later visits
        if self._sim_step_name() == SIMULATION_EVENT_NAMES.LATER_PREGNANCY_SCREENING:
            pop = self.population_view.get(event.index)
            time_of_later_visit = self._calculate_later_visit_timing(event, pop)
            self.population_view.update(time_of_later_visit)

    ##################
    # Helper methods #
    ##################

    def _calculate_first_visit_timing(
        self, event: Event, anc_attendance: pd.Series
    ) -> pd.Series:
        """Calculate timing of first trimester ANC visits."""
        pregnancy_duration_in_weeks = self.pregnancy_duration(event.index) / pd.Timedelta(
            days=7
        )
        attends_first_trimester_anc = anc_attendance.isin(
            [
                ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_ONLY,
                ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_AND_LATER_PREGNANCY,
            ]
        )

        # define lower and upper bounds for visit timing
        has_short_pregnancy = pregnancy_duration_in_weeks < 8
        has_medium_pregnancy = pregnancy_duration_in_weeks.between(8, 12)

        # https://vivarium-research.readthedocs.io/en/latest/models/concept_models/vivarium_mncnh_portfolio/anemia_component/module_document.html#id6
        low = pd.Series(8.0, index=event.index)
        high = pd.Series(12.0, index=event.index)
        low.loc[has_short_pregnancy] = 6.0
        high.loc[has_short_pregnancy] = pregnancy_duration_in_weeks.loc[has_short_pregnancy]
        high.loc[has_medium_pregnancy] = pregnancy_duration_in_weeks.loc[has_medium_pregnancy]
        # calculate visit timing
        draw = self.randomness.get_draw(event.index, additional_key="anc_first_visit_timing")
        time_of_first_visit = pd.Series(
            (low + (high - low) * draw), name=COLUMNS.TIME_OF_FIRST_ANC_VISIT
        ) * pd.Timedelta(days=7)
        time_of_first_visit.loc[~attends_first_trimester_anc] = pd.NaT
        return time_of_first_visit

    def _calculate_later_visit_timing(self, event: Event, pop: pd.DataFrame) -> pd.Series:
        """Calculate timing of later pregnancy ANC visits."""
        # use first visit modified gestational age exposure for pregnancy duration
        pregnancy_duration_in_weeks = self.gestational_age(event.index)
        attends_later_anc = pop[COLUMNS.ANC_ATTENDANCE].isin(
            [
                ANC_ATTENDANCE_TYPES.LATER_PREGNANCY_ONLY,
                ANC_ATTENDANCE_TYPES.FIRST_TRIMESTER_AND_LATER_PREGNANCY,
            ]
        )

        # calculate visit timing
        draw = self.randomness.get_draw(event.index, additional_key="anc_later_visit_timing")
        # https://vivarium-research.readthedocs.io/en/latest/models/concept_models/vivarium_mncnh_portfolio/anemia_component/module_document.html#id6
        low = pd.Series(12, index=event.index)
        high = pd.Series(pregnancy_duration_in_weeks - 2, index=event.index)
        time_of_later_visit = pd.Series(
            (low + (high - low) * draw), name=COLUMNS.TIME_OF_LATER_ANC_VISIT
        ) * pd.Timedelta(days=7)
        time_of_later_visit.loc[~attends_later_anc] = pd.NaT
        return time_of_later_visit


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

        self.population_view.update(pop)

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
