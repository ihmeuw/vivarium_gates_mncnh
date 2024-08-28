import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.metadata import ARTIFACT_INDEX_COLUMNS


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
        pop_update = pop_data["pregnancy_duration"] = self.determine_pregnancy_duration(
            pop_data
        )
        # TODO: Assign sex of infant if pregnancy is full term (stillbirth or live birth)
        # TODO: Assign birthweight of simulant child
        # TODO: Assign propensity values for ANC and ultrasound
        self.population_view.update(pop_update)

    def get_birth_outcome_probabilities(self, builder: Builder) -> pd.DataFrame:
        # TODO: get birth outcome probabilities
        return pd.DataFrame()

    def determine_pregnancy_duration(self, pop_data: SimulantData) -> pd.DataFrame:
        duration_df = pop_data.copy()
        # TODO: get pregnancy duration

        return duration_df
