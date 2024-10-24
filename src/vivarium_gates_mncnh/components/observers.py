from typing import Any

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.results import Observer
from vivarium_public_health.results import COLUMNS, PublicHealthObserver
from vivarium_public_health.results import ResultsStratifier as ResultsStratifier_

from vivarium_gates_mncnh.constants.data_values import COLUMNS, PREGNANCY_OUTCOMES


class ResultsStratifier(ResultsStratifier_):
    def setup(self, builder: Builder) -> None:
        self.age_bins = self.get_age_bins(builder)
        self.register_stratifications(builder)

    def register_stratifications(self, builder: Builder) -> None:
        builder.results.register_stratification(
            "age_group",
            self.age_bins["age_group_name"].to_list(),
            mapper=self.map_age_groups,
            is_vectorized=True,
            requires_columns=["age"],
        )

        builder.results.register_stratification(
            "pregnancy_outcome",
            list(
                [
                    PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME,
                    PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME,
                    PREGNANCY_OUTCOMES.PARTIAL_TERM_OUTCOME,
                ]
            ),
            requires_columns=["pregnancy_outcome"],
        )

        builder.results.register_stratification(
            "sex", ["Female", "Male"], requires_columns=["sex"]
        )


class BirthObserver(Observer):

    COL_MAPPING = {
        "sex_of_child": "sex",
        "birth_weight": "birth_weight",
        "gestational_age": "gestational_age",
        "pregnancy_outcome": "pregnancy_outcome",
    }

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_concatenating_observation(
            name="births",
            pop_filter=(
                "("
                f"pregnancy_outcome == '{PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME}' "
                f"or pregnancy_outcome == '{PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME}'"
                ") "
            ),
            requires_columns=list(self.COL_MAPPING),
            results_formatter=self.format,
        )

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        new_births = results[list(self.COL_MAPPING)].rename(columns=self.COL_MAPPING)
        return new_births


class ANCObserver(Observer):
    def register_observations(self, builder: Builder) -> None:
        builder.results.register_concatenating_observation(
            name="anc",
            requires_columns=[
                COLUMNS.AGE,
                COLUMNS.ATTENDED_CARE_FACILITY,
                COLUMNS.ULTRASOUND_TYPE,
                COLUMNS.GESTATIONAL_AGE,
                COLUMNS.STATED_GESTATIONAL_AGE,
                COLUMNS.PREGNANCY_OUTCOME,
            ],
        )
