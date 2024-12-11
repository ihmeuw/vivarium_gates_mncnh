from functools import partial
from typing import Any

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.results import Observer
from vivarium_public_health.results import COLUMNS
from vivarium_public_health.results import ResultsStratifier as ResultsStratifier_

from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    MATERNAL_DISORDERS,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.constants.metadata import ARTIFACT_INDEX_COLUMNS


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


class MaternalDisordersBurdenObserver(Observer):
    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "stratification": {
                self.get_configuration_name(): {
                    "exclude": [],
                    "include": [],
                    "data_sources": {
                        f"{cause}_ylds": partial(self.load_ylds_per_case, cause=cause)
                        for cause in self.maternal_disorders
                    },
                },
            },
        }

    def __init__(self):
        super().__init__()
        self.maternal_disorders = MATERNAL_DISORDERS

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def get_configuration(self, builder):
        return builder.configuration["stratification"][self.get_configuration_name()]

    def register_observations(self, builder: Builder) -> None:
        dead_pop_filter = f"{COLUMNS.ALIVE} == 'dead'"
        builder.results.register_stratification(
            "cause_of_maternal_death",
            MATERNAL_DISORDERS + ["not_dead"],
            excluded_categories=["not_dead"],
            requires_columns=[COLUMNS.CAUSE_OF_DEATH],
        )

        builder.results.register_adding_observation(
            name="maternal_disorder_deaths",
            pop_filter=dead_pop_filter,
            requires_columns=[COLUMNS.ALIVE],
            additional_stratifications=self.configuration.include
            + ["cause_of_maternal_death"],
            excluded_stratifications=self.configuration.exclude,
            to_observe=self.to_observe,
        )
        builder.results.register_adding_observation(
            name="maternal_disorder_ylls",
            pop_filter=dead_pop_filter,
            requires_columns=[COLUMNS.ALIVE, COLUMNS.YEARS_OF_LIFE_LOST],
            additional_stratifications=self.configuration.include
            + ["cause_of_maternal_death"],
            excluded_stratifications=self.configuration.exclude,
            to_observe=self.to_observe,
            aggregator=self.calculate_ylls,
        )
        for cause in self.maternal_disorders:
            builder.results.register_adding_observation(
                name=f"{cause}_counts",
                pop_filter=f"{cause} == True",
                requires_columns=[cause],
                additional_stratifications=self.configuration.include,
                excluded_stratifications=self.configuration.exclude,
                to_observe=self.to_observe,
            )
            builder.results.register_adding_observation(
                name=f"{cause}_ylds",
                pop_filter=f"{cause} == True",
                requires_columns=[cause],
                additional_stratifications=self.configuration.include,
                excluded_stratifications=self.configuration.exclude,
                to_observe=self.to_observe,
                aggregator=partial(self.calculate_ylds, cause=cause),
            )

    def to_observe(self, event: Event) -> bool:
        return self._sim_step_name() == SIMULATION_EVENT_NAMES.MORTALITY

    def calculate_ylls(self, data: pd.DataFrame) -> float:
        return data[COLUMNS.YEARS_OF_LIFE_LOST].sum()

    def calculate_ylds(self, data: pd.DataFrame, cause: str) -> float:
        yld_per_case = self.lookup_tables[f"{cause}_ylds"](data.index)
        return yld_per_case.sum()

    ##################
    # Helper methods #
    ##################

    def load_ylds_per_case(self, builder: Builder, cause: str) -> pd.DataFrame:
        yld_rate = builder.data.load(f"cause.{cause}.yld_rate").set_index(
            ARTIFACT_INDEX_COLUMNS
        )
        incidence_rate = builder.data.load(f"cause.{cause}.incidence_rate").set_index(
            ARTIFACT_INDEX_COLUMNS
        )
        ylds = (yld_rate / incidence_rate).fillna(0).reset_index()

        return ylds
