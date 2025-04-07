from abc import abstractmethod
from functools import partial
from typing import Any

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.results import Observer
from vivarium_public_health.results import COLUMNS
from vivarium_public_health.results import ResultsStratifier as ResultsStratifier_

from vivarium_gates_mncnh.constants.data_values import (
    CAUSES_OF_NEONATAL_MORTALITY,
    CHILD_INITIALIZATION_AGE,
    COLUMNS,
    DELIVERY_FACILITY_TYPES,
    MATERNAL_DISORDERS,
    PIPELINES,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.constants.metadata import ARTIFACT_INDEX_COLUMNS


class ResultsStratifier(ResultsStratifier_):
    def setup(self, builder: Builder) -> None:
        self.age_bins = self.get_age_bins(builder)
        self.child_age_bins = self.get_child_age_bins(builder)
        self.delivery_facility_types = [
            DELIVERY_FACILITY_TYPES.HOME,
            DELIVERY_FACILITY_TYPES.BEmONC,
            DELIVERY_FACILITY_TYPES.CEmONC,
            DELIVERY_FACILITY_TYPES.NONE,
        ]
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
        builder.results.register_stratification(
            "child_sex",
            ["Female", "Male", "invalid"],
            ["invalid"],
            requires_columns=[COLUMNS.SEX_OF_CHILD],
        )
        builder.results.register_stratification(
            "child_age_group",
            self.child_age_bins["age_group_name"].to_list(),
            mapper=self.map_child_age_groups,
            is_vectorized=True,
            requires_columns=[COLUMNS.CHILD_AGE],
        )
        builder.results.register_stratification(
            "delivery_facility_type",
            self.delivery_facility_types,
            excluded_categories=[DELIVERY_FACILITY_TYPES.NONE],
            is_vectorized=True,
            requires_columns=[COLUMNS.DELIVERY_FACILITY_TYPE],
        )

    def get_child_age_bins(self, builder: Builder) -> pd.DataFrame:
        age_bins_data = {
            "child_age_start": [
                0.0,
                CHILD_INITIALIZATION_AGE,
                7 / 365.0,
            ],
            "child_age_end": [
                CHILD_INITIALIZATION_AGE,
                7 / 365.0,
                28 / 365.0,
            ],
            "age_group_name": [
                "stillbirth",
                "early_neonatal",
                "late_neonatal",
            ],
        }
        return pd.DataFrame(age_bins_data)

    def map_child_age_groups(self, pop: pd.DataFrame) -> pd.Series:
        # Overwriting to use child_age_bins
        bins = self.child_age_bins["child_age_start"].to_list() + [
            self.child_age_bins["child_age_end"].iloc[-1]
        ]
        labels = self.child_age_bins["age_group_name"].to_list()
        age_group = pd.cut(pop.squeeze(axis=1), bins, labels=labels).rename("child_age_group")

        return age_group


class BirthObserver(Observer):

    COL_MAPPING = {
        COLUMNS.SEX_OF_CHILD: "sex",
        COLUMNS.BIRTH_WEIGHT_EXPOSURE: "birth_weight",
        COLUMNS.GESTATIONAL_AGE_EXPOSURE: "gestational_age",
        COLUMNS.PREGNANCY_OUTCOME: "pregnancy_outcome",
    }

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def register_observations(self, builder: Builder) -> None:
        # TODO: update this to adding observation when docs are ready
        builder.results.register_concatenating_observation(
            name="births",
            pop_filter=(
                "("
                f"pregnancy_outcome == '{PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME}' "
                f"or pregnancy_outcome == '{PREGNANCY_OUTCOMES.STILLBIRTH_OUTCOME}'"
                ") "
            ),
            requires_columns=list(self.COL_MAPPING) + [COLUMNS.DELIVERY_FACILITY_TYPE],
            results_formatter=self.format,
            to_observe=self.to_observe,
        )

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        new_births = results[
            list(self.COL_MAPPING) + [COLUMNS.DELIVERY_FACILITY_TYPE]
        ].rename(columns=self.COL_MAPPING)
        return new_births

    def to_observe(self, event: Event) -> bool:
        return self._sim_step_name() == SIMULATION_EVENT_NAMES.CPAP_ACCESS


class ANCObserver(Observer):
    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def register_observations(self, builder: Builder) -> None:
        # TODO: update this to adding observation when docs are ready
        builder.results.register_concatenating_observation(
            name="anc",
            requires_columns=[
                COLUMNS.MOTHER_AGE,
                COLUMNS.ATTENDED_CARE_FACILITY,
                COLUMNS.ULTRASOUND_TYPE,
                COLUMNS.STATED_GESTATIONAL_AGE,
                COLUMNS.PREGNANCY_OUTCOME,
            ],
            requires_values=[PIPELINES.PREGNANCY_DURATION],
            to_observe=self.to_observe,
        )

    def to_observe(self, event: Event) -> bool:
        return self._sim_step_name() == SIMULATION_EVENT_NAMES.PREGNANCY


class BurdenObserver(Observer):
    def __init__(
        self,
        burden_disorders: list[str],
        alive_column: str,
        ylls_column: str,
        cause_of_death_column: str,
    ):
        super().__init__()
        self.burden_disorders = burden_disorders
        self.alive_column = alive_column
        self.ylls_column = ylls_column
        self.cause_of_death_column = cause_of_death_column

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def get_configuration(self, builder: Builder) -> dict[str, Any]:
        return builder.configuration["stratification"][self.get_configuration_name()]

    def register_observations(self, builder: Builder) -> None:
        dead_pop_filter = f"{self.alive_column} == 'dead'"
        builder.results.register_stratification(
            name=f"{self.name}_cause_of_death",
            categories=self.burden_disorders + ["not_dead"],
            excluded_categories=["not_dead"],
            requires_columns=[self.cause_of_death_column],
        )

        builder.results.register_adding_observation(
            name=f"{self.name}_disorder_deaths",
            pop_filter=dead_pop_filter,
            requires_columns=[self.alive_column],
            additional_stratifications=self.configuration.include
            + [f"{self.name}_cause_of_death"],
            excluded_stratifications=self.configuration.exclude,
            to_observe=self.to_observe,
        )
        builder.results.register_adding_observation(
            name=f"{self.name}_disorder_ylls",
            pop_filter=dead_pop_filter,
            requires_columns=[self.alive_column, self.ylls_column],
            additional_stratifications=self.configuration.include
            + [f"{self.name}_cause_of_death"],
            excluded_stratifications=self.configuration.exclude,
            to_observe=self.to_observe,
            aggregator=self.calculate_ylls,
        )

    @abstractmethod
    def to_observe(self, event: Event) -> bool:
        pass

    def calculate_ylls(self, data: pd.DataFrame) -> float:
        return data[self.ylls_column].sum()


class MaternalDisordersBurdenObserver(BurdenObserver):
    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "stratification": {
                self.get_configuration_name(): {
                    "exclude": [],
                    "include": [],
                    "data_sources": {
                        f"{cause}_ylds": partial(self.load_ylds_per_case, cause=cause)
                        for cause in self.burden_disorders
                    },
                },
            },
        }

    def __init__(self):
        super().__init__(
            burden_disorders=MATERNAL_DISORDERS,
            alive_column=COLUMNS.MOTHER_ALIVE,
            ylls_column=COLUMNS.MOTHER_YEARS_OF_LIFE_LOST,
            cause_of_death_column=COLUMNS.MOTHER_CAUSE_OF_DEATH,
        )

    def register_observations(self, builder: Builder) -> None:
        super().register_observations(builder)
        for cause in self.burden_disorders:
            builder.results.register_adding_observation(
                name=f"{cause}_death_counts",
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


class NeonatalBurdenObserver(BurdenObserver):
    """Observer to capture death counts and ylls for neonatal sub causes."""

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "stratification": {
                self.get_configuration_name(): {
                    "exclude": ["age_group"],
                    "include": [
                        "child_age_group",
                        "child_sex",
                        "cpap_availability",
                        "antibiotics_availability",
                        "delivery_facility_type",
                    ],
                },
            },
        }

    def __init__(self):
        super().__init__(
            burden_disorders=CAUSES_OF_NEONATAL_MORTALITY + ["other_causes"],
            alive_column=COLUMNS.CHILD_ALIVE,
            ylls_column=COLUMNS.CHILD_YEARS_OF_LIFE_LOST,
            cause_of_death_column=COLUMNS.CHILD_CAUSE_OF_DEATH,
        )

    def register_observations(self, builder: Builder) -> None:
        super().register_observations(builder)
        builder.results.register_stratification(
            "cpap_availability",
            [True, False],
            requires_columns=[COLUMNS.CPAP_AVAILABLE],
        )
        builder.results.register_stratification(
            "antibiotics_availability",
            [True, False],
            requires_columns=[COLUMNS.ANTIBIOTICS_AVAILABLE],
        )
        for cause in self.burden_disorders:
            builder.results.register_adding_observation(
                name=f"{cause}_death_counts",
                pop_filter=f"{self.cause_of_death_column} == '{cause}'",
                requires_columns=[self.cause_of_death_column],
                additional_stratifications=self.configuration.include,
                excluded_stratifications=self.configuration.exclude,
                to_observe=self.to_observe,
            )

    def to_observe(self, event: Event) -> bool:
        # Need to make single observeration of deaths after all time steps where neonates die.
        return self._sim_step_name() == SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY


class NeonatalCauseRelativeRiskObserver(Observer):
    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "stratification": {
                self.get_configuration_name(): {
                    "exclude": ["age_group"],
                    "include": ["child_age_group", "child_sex"],
                },
            },
        }

    def __init__(self):
        super().__init__()
        self.neonatal_causes = CAUSES_OF_NEONATAL_MORTALITY + ["all_causes"]

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def get_configuration(self, builder: Builder) -> dict[str, Any]:
        return builder.configuration["stratification"][self.get_configuration_name()]

    def register_observations(self, builder: Builder) -> None:
        for cause in self.neonatal_causes:
            builder.results.register_adding_observation(
                name=f"{cause}_relative_risk",
                requires_values=[
                    f"effect_of_low_birth_weight_and_short_gestation_on_{cause}.relative_risk"
                ],
                additional_stratifications=self.configuration.include,
                excluded_stratifications=self.configuration.exclude,
                to_observe=self.to_observe,
            )

    def to_observe(self, event: Event) -> bool:
        return (self._sim_step_name() == SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY) or (
            self._sim_step_name() == SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY
        )


class NeonatalInterventionObserver(Observer):
    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "stratification": {
                f"{self.get_configuration_name()}_{self.intervention}": {
                    "exclude": ["age_group"],
                    "include": ["delivery_facility_type"],
                },
            },
        }

    def __init__(self, intervention: str) -> None:
        super().__init__()
        self.intervention = intervention

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()

    def get_configuration(self, builder: Builder) -> dict[str, Any]:
        return builder.configuration["stratification"][
            f"{self.get_configuration_name()}_{self.intervention}"
        ]

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name=self.intervention,
            pop_filter=f"{self.intervention}_available == True",
            requires_columns=[f"{self.intervention}_available"],
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            to_observe=self.to_observe,
        )

    def to_observe(self, event: Event) -> bool:
        # Last time step
        return self._sim_step_name() == SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY
