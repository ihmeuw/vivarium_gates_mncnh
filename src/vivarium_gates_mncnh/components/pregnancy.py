import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium_public_health.utilities import get_lookup_columns

from vivarium_gates_mncnh.components.children import NewChildren
from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import (
    CHILD_INITIALIZATION_AGE,
    COLUMNS,
    DURATIONS,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.constants.metadata import ARTIFACT_INDEX_COLUMNS


class Pregnancy(Component):

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self):
        return [
            COLUMNS.PREGNANCY_OUTCOME,
            COLUMNS.PREGNANCY_DURATION,
            COLUMNS.GESTATIONAL_AGE,
            COLUMNS.BIRTH_WEIGHT,
            COLUMNS.SEX_OF_CHILD,
            COLUMNS.CHILD_AGE,
            COLUMNS.CHILD_ALIVE,
        ]

    @property
    def sub_components(self):
        return super().sub_components + [self.new_children]

    @property
    def initialization_requirements(self) -> dict[str, list[str]]:
        return {
            "requires_columns": [],
            "requires_values": ["birth_outcome_probabilities"],
            "requires_streams": [],
        }

    @property
    def time_step_priority(self) -> int:
        # This is to age the children before mortality happens
        return 0

    def __init__(self):
        super().__init__()
        self.new_children = NewChildren()

    def setup(self, builder: Builder):
        """Performs this component's simulation setup.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools.
        """
        self.time_step = builder.time.step_size()
        self.randomness = builder.randomness.get_stream(self.name)
        self._sim_step_name = builder.time.simulation_event_name()
        self.birth_outcome_probabilities = builder.value.register_value_producer(
            "birth_outcome_probabilities",
            source=self.lookup_tables["birth_outcome_probabilities"],
            requires_columns=get_lookup_columns(
                [self.lookup_tables["birth_outcome_probabilities"]]
            ),
        )

    def build_all_lookup_tables(self, builder: Builder) -> None:
        super().build_all_lookup_tables(builder)
        # I am not making birth outcome probabilities configurable because the
        # method is so complicated - albrja
        birth_outcome_probabilities = self.get_birth_outcome_probabilities(builder)
        self.lookup_tables["birth_outcome_probabilities"] = self.build_lookup_table(
            builder,
            birth_outcome_probabilities,
            value_columns=["live_birth", "partial_term", "stillbirth"],
        )

    #####################
    # Lifecycle Methods #
    #####################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pregnancy_outcomes_and_durations = self.sample_pregnancy_outcomes_and_durations(
            pop_data
        )
        pregnancy_outcomes_and_durations[COLUMNS.CHILD_ALIVE] = "dead"
        pregnancy_outcomes_and_durations[COLUMNS.CHILD_AGE] = CHILD_INITIALIZATION_AGE / 2
        live_birth_index = pregnancy_outcomes_and_durations.index[
            pregnancy_outcomes_and_durations[COLUMNS.PREGNANCY_OUTCOME]
            == PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME
        ]
        pregnancy_outcomes_and_durations.loc[live_birth_index, COLUMNS.CHILD_ALIVE] = "alive"

        self.population_view.update(pregnancy_outcomes_and_durations)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() not in [
            SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY,
            SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY,
        ]:
            return

        pop = self.population_view.get(event.index)
        alive_children = pop.loc[pop[COLUMNS.CHILD_ALIVE] == "alive"]
        # Update age of children to get correctlookup values - use midpoint of age groups
        if self._sim_step_name() == SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY:
            pop.loc[alive_children.index, COLUMNS.CHILD_AGE] = (7 / 2) / 365.0
        else:
            pop.loc[alive_children.index, COLUMNS.CHILD_AGE] = ((28 - 7) / 2) / 365.0

        self.population_view.update(pop)

    ##################
    # Helper methods #
    ##################

    def get_birth_outcome_probabilities(self, builder: Builder) -> pd.DataFrame:
        asfr = builder.data.load(data_keys.PREGNANCY.ASFR).set_index(ARTIFACT_INDEX_COLUMNS)
        sbr = (
            builder.data.load(data_keys.PREGNANCY.SBR)
            .set_index("year_start")
            .drop(columns=["year_end"])
            .reindex(asfr.index, level="year_start")
        )

        raw_incidence_miscarriage = builder.data.load(
            data_keys.PREGNANCY.RAW_INCIDENCE_RATE_MISCARRIAGE
        ).set_index(ARTIFACT_INDEX_COLUMNS)
        raw_incidence_ectopic = builder.data.load(
            data_keys.PREGNANCY.RAW_INCIDENCE_RATE_ECTOPIC
        ).set_index(ARTIFACT_INDEX_COLUMNS)

        total_incidence = (
            asfr
            + asfr.multiply(sbr["value"], axis=0)
            + raw_incidence_ectopic
            + raw_incidence_miscarriage
        )

        partial_term = (raw_incidence_ectopic + raw_incidence_miscarriage) / total_incidence
        partial_term[COLUMNS.PREGNANCY_OUTCOME] = PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME
        live_births = asfr / total_incidence
        live_births[COLUMNS.PREGNANCY_OUTCOME] = PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME
        stillbirths = asfr.multiply(sbr["value"], axis=0) / total_incidence
        stillbirths[COLUMNS.PREGNANCY_OUTCOME] = PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME
        probabilities = pd.concat([partial_term, live_births, stillbirths])
        probabilities = probabilities.pivot(
            columns=COLUMNS.PREGNANCY_OUTCOME, values="value"
        ).reset_index()
        return probabilities

    def sample_pregnancy_outcomes_and_durations(self, pop_data: SimulantData) -> pd.DataFrame:
        # Order the columns so that partial_term isn't in the middle!
        outcome_probabilities = self.birth_outcome_probabilities(pop_data.index)[
            ["partial_term", "stillbirth", "live_birth"]
        ]
        pregnancy_outcomes = pd.DataFrame(
            {
                "pregnancy_outcome": self.randomness.choice(
                    pop_data.index,
                    choices=outcome_probabilities.columns.to_list(),
                    p=outcome_probabilities,
                    additional_key="pregnancy_outcome",
                )
            }
        )

        term_child_map = {
            PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME: self.sample_full_term_durations,
            PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME: self.sample_full_term_durations,
            PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME: self.sample_partial_term_durations,
        }

        for term_length, sampling_function in term_child_map.items():
            term_pop = pregnancy_outcomes[
                pregnancy_outcomes["pregnancy_outcome"] == term_length
            ].index
            pregnancy_outcomes.loc[
                term_pop,
                ["sex_of_child", "birth_weight", "gestational_age", "pregnancy_duration"],
            ] = sampling_function(term_pop)

        return pregnancy_outcomes

    def sample_partial_term_durations(self, partial_term_pop: pd.Index) -> pd.DataFrame:
        child_status = self.new_children.get_child_data_structure(partial_term_pop)
        low, high = DURATIONS.DETECTION_DAYS, DURATIONS.PARTIAL_TERM_DAYS
        draw = self.randomness.get_draw(
            partial_term_pop, additional_key="partial_term_pregnancy_duration"
        )
        child_status["pregnancy_duration"] = pd.to_timedelta(
            (low + (high - low) * draw), unit="days"
        )
        return child_status

    def sample_full_term_durations(self, full_term_pop: pd.Index) -> pd.DataFrame:
        child_status = self.new_children.generate_children(full_term_pop)
        child_status["pregnancy_duration"] = pd.to_timedelta(
            7 * child_status["gestational_age"], unit="days"
        )
        return child_status
