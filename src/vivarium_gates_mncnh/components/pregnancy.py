import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.components.children import NewChildren
from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    DURATIONS,
    PIPELINES,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.constants.metadata import ARTIFACT_INDEX_COLUMNS


class Pregnancy(Component):

    ##############
    # Properties #
    ##############

    @property
    def sub_components(self):
        return super().sub_components + [self.new_children]

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
        self._sim_step_name = builder.time.simulation_event_name()
        self.time_step = builder.time.step_size()
        self.randomness = builder.randomness.get_stream(self.name)

        self.birth_outcome_probabilities_table = self.build_lookup_table(
            builder,
            "birth_outcome_probabilities",
            data_source=self.get_birth_outcome_probabilities(builder),
            value_columns=["live_birth", "partial_term", "stillbirth"],
        )
        builder.value.register_attribute_producer(
            PIPELINES.BIRTH_OUTCOME_PROBABILITIES,
            source=self.birth_outcome_probabilities_table,
        )
        builder.value.register_attribute_producer(
            PIPELINES.PREGNANCY_DURATION,
            source=self.get_pregnancy_durations,
        )
        builder.population.register_initializer(
            self.initialize_pregnancy,
            columns=[
                COLUMNS.PREGNANCY_OUTCOME,
                COLUMNS.PARTIAL_TERM_PREGNANCY_DURATION,
            ],
            required_resources=[
                self.randomness,
                PIPELINES.BIRTH_OUTCOME_PROBABILITIES,
            ],
        )

    #####################
    # Lifecycle Methods #
    #####################

    def initialize_pregnancy(self, pop_data: SimulantData) -> None:
        pregnancy_outcomes = self.sample_pregnancy_outcomes(pop_data)
        partial_term_idx = pregnancy_outcomes.index[
            pregnancy_outcomes[COLUMNS.PREGNANCY_OUTCOME]
            == PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME
        ]
        # Get partial term pregnancy gestational ages (duration)
        pregnancy_outcomes[COLUMNS.PARTIAL_TERM_PREGNANCY_DURATION] = np.nan
        pregnancy_outcomes.loc[
            partial_term_idx, COLUMNS.PARTIAL_TERM_PREGNANCY_DURATION
        ] = self.get_partial_term_gestational_age(partial_term_idx)

        self.population_view.update(pregnancy_outcomes)

    def on_time_step_cleanup(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.LATER_PREGNANCY_INTERVENTION:
            return

        outcome_probabilities = self.population_view.get_attribute_frame(
            event.index, PIPELINES.BIRTH_OUTCOME_PROBABILITIES
        )[[PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME, PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME]]
        pregnancy_outcome = self.population_view.get_attributes(
            event.index, COLUMNS.PREGNANCY_OUTCOME
        )
        is_full_term = pregnancy_outcome == PREGNANCY_OUTCOMES.FULL_TERM_OUTCOME
        full_term_idx = is_full_term.index[is_full_term]
        full_term_outcomes = self.randomness.choice(
            full_term_idx,
            choices=[
                PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME,
                PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME,
            ],
            p=outcome_probabilities.loc[full_term_idx],
            additional_key="full_term_outcome",
        )
        pregnancy_outcome.loc[full_term_idx] = full_term_outcomes

        self.population_view.update(pregnancy_outcome)

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

    def sample_pregnancy_outcomes(self, pop_data: SimulantData) -> pd.DataFrame:
        # Order the columns so that partial_term isn't in the middle!
        partial_term_probabilities = self.population_view.get_attribute_frame(
            pop_data.index, PIPELINES.BIRTH_OUTCOME_PROBABILITIES
        )[PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME]
        pregnancy_outcomes = pd.DataFrame(
            {
                "pregnancy_outcome": self.randomness.choice(
                    pop_data.index,
                    choices=[
                        PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME,
                        PREGNANCY_OUTCOMES.FULL_TERM_OUTCOME,
                    ],
                    p=pd.concat(
                        [partial_term_probabilities, 1 - partial_term_probabilities], axis=1
                    ),
                    additional_key="pregnancy_outcome",
                )
            }
        )
        return pregnancy_outcomes

    def get_partial_term_gestational_age(self, index: pd.Index) -> pd.Series:
        """Get the gestational age for partial term pregnancies."""
        low, high = DURATIONS.PARTIAL_TERM_LOWER_WEEKS, DURATIONS.PARTIAL_TERM_UPPER_WEEKS
        draw = self.randomness.get_draw(
            index, additional_key="partial_term_pregnancy_duration"
        )
        durations = pd.Series((low + (high - low) * draw))
        return durations

    def get_pregnancy_durations(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get_attributes(
            index,
            [
                COLUMNS.PREGNANCY_OUTCOME,
                COLUMNS.PARTIAL_TERM_PREGNANCY_DURATION,
                COLUMNS.GESTATIONAL_AGE_EXPOSURE,
            ],
        )
        partial_term_idx = pop.index[
            pop[COLUMNS.PREGNANCY_OUTCOME] == PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME
        ]
        partial_ga = pop.loc[partial_term_idx, COLUMNS.PARTIAL_TERM_PREGNANCY_DURATION]
        non_partial_idx = index.difference(partial_term_idx)
        non_partial_ga = pop.loc[non_partial_idx, COLUMNS.GESTATIONAL_AGE_EXPOSURE]

        gestational_ages = pd.concat([partial_ga, non_partial_ga]).sort_index()
        durations = pd.to_timedelta(7 * gestational_ages, unit="days")
        return durations
