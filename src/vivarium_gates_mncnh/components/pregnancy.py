import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.metadata import ARTIFACT_INDEX_COLUMNS
from vivarium_gates_mncnh.constants.data_values import PREGNANCY_OUTCOMES


class Pregnancy(Component):

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self):
        return [
            "pregnancy_outcome",
            "pregnancy_duration",
            "sex_of_child",
            "birth_weight",
            "gestational_age",
        ]

    def setup(self, builder: Builder):
        """Performs this component's simulation setup.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools.
        """
        super().setup(builder)
        self.time_step = builder.time.step_size()
        self.randomness = builder.randomness.get_stream(self.name)
        # TODO: need randomness stream for initialization
        # TODO: add attribute to record outputs

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

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        # TODO: Assign partial or full term duration according to table in `Pregnancy term lengths`_ section
        pregnancy_outcomes_and_durations = self.sample_pregnancy_outcomes_and_durations(
            pop_data
        )
        # TODO: Assign sex of infant if pregnancy is full term (stillbirth or live birth)
        # TODO: Assign birthweight of simulant child
        # TODO: Assign propensity values for ANC and ultrasound
        self.population_view.update(pop_update)

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
        partial_term["pregnancy_outcome"] = PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME
        live_births = asfr / total_incidence
        live_births["pregnancy_outcome"] = PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME
        stillbirths = asfr.multiply(sbr["value"], axis=0) / total_incidence
        stillbirths["pregnancy_outcome"] = PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME
        probabilities = pd.concat([partial_term, live_births, stillbirths])
        probabilities = probabilities.pivot(
            columns="pregnancy_outcome", values="value"
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
