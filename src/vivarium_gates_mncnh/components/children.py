from typing import List, Tuple

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import (
    CHILD_INITIALIZATION_AGE,
    COLUMNS,
    INFANT_MALE_PERCENTAGES,
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
)


class NewChildren(Component):
    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> list[str]:
        return [
            COLUMNS.SEX_OF_CHILD,
            COLUMNS.CHILD_AGE,
            COLUMNS.CHILD_ALIVE,
        ]

    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.PREGNANCY_OUTCOME]

    @property
    def initialization_requirements(self):
        return [
            COLUMNS.PREGNANCY_OUTCOME,
        ]

    @property
    def time_step_priority(self) -> int:
        # This is to age the children before mortality happens
        return 0

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)
        self._sim_step_name = builder.time.simulation_event_name()
        self.male_sex_percentage = INFANT_MALE_PERCENTAGES[
            builder.data.load(data_keys.POPULATION.LOCATION)
        ]

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        index = pop_data.index
        sex_of_child = self.randomness.choice(
            index,
            choices=["Male", "Female"],
            p=[self.male_sex_percentage, 1 - self.male_sex_percentage],
            additional_key="sex_of_child",
        )
        new_children = pd.DataFrame(
            {
                COLUMNS.SEX_OF_CHILD: sex_of_child,
                COLUMNS.CHILD_AGE: CHILD_INITIALIZATION_AGE / 2,
                COLUMNS.CHILD_ALIVE: "dead",
            },
            index=index,
        )
        # Find live births and set child status to alive
        outcomes = self.population_view.subview([COLUMNS.PREGNANCY_OUTCOME]).get(index)
        live_birth_index = outcomes.index[
            outcomes[COLUMNS.PREGNANCY_OUTCOME] == PREGNANCY_OUTCOMES.LIVE_BIRTH_OUTCOME
        ]
        new_children.loc[live_birth_index, COLUMNS.CHILD_ALIVE] = "alive"
        self.population_view.update(new_children)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() not in [
            SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY,
            SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY,
        ]:
            return

        pop = self.population_view.get(event.index)
        alive_children = pop.loc[pop[COLUMNS.CHILD_ALIVE] == "alive"]
        # Update age of children to get correctlookup values - use midpoint of age groups
        age_group_midpoints = {
            SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY: (7 / 2) / 365.0,
            SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY: (7 + (28 - 7) / 2) / 365.0,
        }
        pop.loc[alive_children.index, COLUMNS.CHILD_AGE] = age_group_midpoints[
            self._sim_step_name()
        ]

        self.population_view.update(pop)
