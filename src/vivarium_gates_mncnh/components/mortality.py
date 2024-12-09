from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.constants.data_values import COLUMNS, SIMULATION_EVENT_NAMES
from vivarium_gates_mncnh.constants.metadata import ARTIFACT_INDEX_COLUMNS
from vivarium_gates_mncnh.utilities import get_location


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
                            self.load_cfr_data, key_name=cause
                        )
                        for cause in self.maternal_disorders
                    },
                    **{
                        f"{cause}_yld_rate": f"cause.{cause}.yld_rate"
                        for cause in self.maternal_disorders
                    },
                },
            },
        }

    @property
    def columns_created(self) -> list[str]:
        return [COLUMNS.CAUSE_OF_DEATH, COLUMNS.YEARS_OF_LIFE_LOST] + [
            f"{cause}_ylds" for cause in self.maternal_disorders
        ]

    @property
    def columns_required(self) -> list[str]:
        return [
            COLUMNS.ALIVE,
            COLUMNS.EXIT_TIME,
            COLUMNS.AGE,
            COLUMNS.SEX,
        ] + self.maternal_disorders

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self) -> None:
        super().__init__()
        # TODO: update list of maternal disorders when implemented
        self.maternal_disorders = [
            COLUMNS.OBSTRUCTED_LABOR,
            COLUMNS.MATERNAL_HEMORRHAGE,
            COLUMNS.MATERNAL_SEPSIS,
        ]

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
                **{
                    COLUMNS.CAUSE_OF_DEATH: "not_dead",
                    COLUMNS.YEARS_OF_LIFE_LOST: 0.0,
                },
                **{f"{cause}_ylds": 0.0 for cause in self.maternal_disorders},
            },
            index=pop_data.index,
        )
        self.population_view.update(pop_update)

    def on_time_step(self, event) -> None:
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
            pop.loc[dead_idx, COLUMNS.ALIVE] = "dead"

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
            pop.loc[dead_idx, COLUMNS.CAUSE_OF_DEATH] = cause_of_death
            pop.loc[dead_idx, COLUMNS.YEARS_OF_LIFE_LOST] = self.lookup_tables[
                "life_expectancy"
            ](dead_idx)

        # Update YLDs for each maternal disorder
        yld_idx = has_maternal_disorders.index.difference(dead_idx)
        for cause in self.maternal_disorders:
            pop.loc[yld_idx, f"{cause}_ylds"] = self.lookup_tables[f"{cause}_yld_rate"](
                yld_idx
            )

        self.population_view.update(pop)

    ##################
    # Helper methods #
    ##################

    def load_cfr_data(self, builder: Builder, key_name: str) -> pd.DataFrame:
        """Load case fatality rate data for maternal disorders."""
        maternal_disorder = key_name.split("_case_fatality_rate")[0]
        incidence_rate = builder.data.load(
            f"cause.{maternal_disorder}.incidence_rate"
        ).set_index(ARTIFACT_INDEX_COLUMNS)
        csmr = builder.data.load(
            f"cause.{maternal_disorder}.cause_specific_mortality_rate"
        ).set_index(ARTIFACT_INDEX_COLUMNS)
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
