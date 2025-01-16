from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium_public_health.utilities import get_lookup_columns

from vivarium_gates_mncnh.constants.data_values import (
    CHILD_LOOKUP_COLUMN_MAPPER,
    COLUMNS,
    MATERNAL_DISORDERS,
    PIPELINES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.constants.metadata import ARTIFACT_INDEX_COLUMNS
from vivarium_gates_mncnh.utilities import get_location, rate_to_probability


class MaternalDisordersBurden(Component):
    """A component to handle morbidity and mortality caused by the modeled maternal disorders."""

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            self.name: {
                "data_sources": {
                    **{
                        "life_expectancy": "population.theoretical_minimum_risk_life_expectancy"
                    },
                    **{
                        f"{cause}_case_fatality_rate": partial(
                            self.load_cfr_data, cause=cause
                        )
                        for cause in self.maternal_disorders
                    },
                },
            },
        }

    @property
    def columns_created(self) -> list[str]:
        return [COLUMNS.MOTHER_CAUSE_OF_DEATH, COLUMNS.MOTHER_YEARS_OF_LIFE_LOST]

    @property
    def columns_required(self) -> list[str]:
        return [
            COLUMNS.MOTHER_ALIVE,
            COLUMNS.EXIT_TIME,
            COLUMNS.MOTHER_AGE,
            COLUMNS.MOTHER_SEX,
        ] + self.maternal_disorders

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self) -> None:
        super().__init__()
        self.maternal_disorders = MATERNAL_DISORDERS

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = get_location(builder)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame(
            {
                COLUMNS.MOTHER_CAUSE_OF_DEATH: "not_dead",
                COLUMNS.MOTHER_YEARS_OF_LIFE_LOST: 0.0,
            },
            index=pop_data.index,
        )
        self.population_view.update(pop_update)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() != SIMULATION_EVENT_NAMES.MORTALITY:
            return

        pop = self.population_view.get(event.index)
        has_maternal_disorders = pop[self.maternal_disorders]
        has_maternal_disorders = has_maternal_disorders.loc[
            has_maternal_disorders.any(axis=1)
        ]

        # Get raw and conditional case fatality rates for each simulant
        choice_data = has_maternal_disorders.copy()
        choice_data = self.calculate_case_fatality_rates(choice_data)

        # Decide what simulants die from what maternal disorders
        dead_idx = self.randomness.filter_for_probability(
            choice_data.index,
            choice_data["mortality_probability"],
            "mortality_choice",
        )

        # Update metadata for simulants that died
        if not dead_idx.empty:
            pop.loc[dead_idx, COLUMNS.MOTHER_ALIVE] = "dead"

            # Get maternal disorders each simulant is affect by
            cause_of_death = self.randomness.choice(
                index=dead_idx,
                choices=self.maternal_disorders,
                p=choice_data.loc[
                    dead_idx,
                    [f"{disorder}_proportional_cfr" for disorder in self.maternal_disorders],
                ],
                additional_key="cause_of_death",
            )
            pop.loc[dead_idx, COLUMNS.MOTHER_CAUSE_OF_DEATH] = cause_of_death
            # TODO: should this be age + gestational_age
            pop.loc[dead_idx, COLUMNS.MOTHER_YEARS_OF_LIFE_LOST] = self.lookup_tables[
                "life_expectancy"
            ](dead_idx)

        self.population_view.update(pop)

    ##################
    # Helper methods #
    ##################

    def load_cfr_data(self, builder: Builder, cause: str) -> pd.DataFrame:
        """Load case fatality rate data for maternal disorders."""
        incidence_rate = builder.data.load(f"cause.{cause}.incidence_rate").set_index(
            ARTIFACT_INDEX_COLUMNS
        )
        csmr = builder.data.load(f"cause.{cause}.cause_specific_mortality_rate").set_index(
            ARTIFACT_INDEX_COLUMNS
        )
        cfr = (csmr / incidence_rate).fillna(0).reset_index()

        return cfr

    def calculate_case_fatality_rates(self, simulants: pd.DataFrame) -> pd.DataFrame:
        """Calculate the total and proportional case fatality rate for each simulant."""

        # Simulants is a boolean dataframe of whether or not a simulant has each maternal disorder.
        for cause in self.maternal_disorders:
            simulants[cause] = simulants[cause] * self.lookup_tables[
                f"{cause}_case_fatality_rate"
            ](simulants.index)
        simulants["mortality_probability"] = simulants[self.maternal_disorders].sum(axis=1)
        cfr_data = self.get_proportional_case_fatality_rates(simulants)

        return cfr_data

    def get_proportional_case_fatality_rates(self, simulants: pd.DataFrame) -> pd.DataFrame:
        """Calculate the proportional case fatality rates for each maternal disorder."""

        for cause in self.maternal_disorders:
            simulants[f"{cause}_proportional_cfr"] = (
                simulants[cause] / simulants["mortality_probability"]
            )

        return simulants


class NeonatalMortality(Component):
    """A component to handle neonatal mortality."""

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            self.name: {
                "data_sources": {
                    "all_cause_mortality_rate": self.load_acmr,
                    "life_expectancy": self.load_life_expectancy_data,
                }
            }
        }

    @property
    def columns_created(self) -> list[str]:
        return [COLUMNS.CHILD_CAUSE_OF_DEATH, COLUMNS.CHILD_YEARS_OF_LIFE_LOST]

    @property
    def columns_required(self) -> list[str]:
        return [
            COLUMNS.CHILD_AGE,
            COLUMNS.CHILD_ALIVE,
        ]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = get_location(builder)

        self.all_cause_mortality_rate = builder.value.register_value_producer(
            PIPELINES.ACMR,
            source=self.lookup_tables["all_cause_mortality_rate"],
            component=self,
            required_resources=get_lookup_columns(
                [self.lookup_tables["all_cause_mortality_rate"]]
            ),
        )
        # Modify ACMR pipeline with CSMR for neonatal causes
        self.death_in_age_group = builder.value.register_value_producer(
            PIPELINES.DEATH_IN_AGE_GROUP_PROBABILITY,
            source=self.all_cause_mortality_rate,
            component=self,
            required_resources=[self.all_cause_mortality_rate],
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame(
            {
                COLUMNS.CHILD_CAUSE_OF_DEATH: "not_dead",
                COLUMNS.CHILD_YEARS_OF_LIFE_LOST: 0.0,
            },
            index=pop_data.index,
        )
        self.population_view.update(pop_update)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() not in [
            SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY,
            SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY,
        ]:
            return

        pop = self.population_view.get(event.index)
        alive_children = pop.loc[pop[COLUMNS.CHILD_ALIVE] == "alive"]
        mortality_rates = self.death_in_age_group(alive_children.index)
        # Convert to rates to probability
        if self._sim_step_name() == SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY:
            duration = 7 / 365.0
        else:
            duration = 21 / 365.0
        mortality_risk = rate_to_probability(mortality_rates, duration)

        # Determine which neonates die and update metadata
        dead_idx = self.randomness.filter_for_probability(
            alive_children.index,
            mortality_risk,
            f"{self._sim_step_name}_neonatal_mortality_choice",
        )
        if not dead_idx.empty:
            pop.loc[dead_idx, COLUMNS.CHILD_ALIVE] = "dead"
            pop.loc[dead_idx, COLUMNS.CHILD_CAUSE_OF_DEATH] = "other_causes"
            pop.loc[dead_idx, COLUMNS.CHILD_YEARS_OF_LIFE_LOST] = self.lookup_tables[
                "life_expectancy"
            ](dead_idx)

        self.population_view.update(pop)

    ##################
    # Helper methods #
    ##################

    def load_acmr(self, builder: Builder) -> pd.DataFrame:
        """Load all-cause mortality rate data."""
        acmr = builder.data.load("cause.all_causes.cause_specific_mortality_rate")
        child_acmr = acmr.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
        return child_acmr

    def load_life_expectancy_data(self, builder: Builder) -> pd.DataFrame:
        """Load life expectancy data."""
        life_expectancy = builder.data.load(
            "population.theoretical_minimum_risk_life_expectancy"
        )
        child_life_expectancy = life_expectancy.rename(columns=CHILD_LOOKUP_COLUMN_MAPPER)
        return child_life_expectancy
