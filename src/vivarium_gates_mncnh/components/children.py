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
    PREGNANCY_OUTCOMES,
    SIMULATION_EVENT_NAMES,
)


class NewChildren(Component):
    ##############
    # Properties #
    ##############

    @property
    def time_step_priority(self) -> int:
        # This is to age the children before mortality happens
        return 0

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)
        self._sim_step_name = builder.time.simulation_event_name()
        self.male_sex_percentage = builder.data.load(
            data_keys.POPULATION.INFANT_MALE_PERCENTAGE
        )
        builder.population.register_initializer(
            self.initialize_children,
            columns=[
                COLUMNS.SEX_OF_CHILD,
                COLUMNS.CHILD_AGE,
            ],
            required_resources=[COLUMNS.PREGNANCY_OUTCOME, self.randomness],
        )

    def initialize_children(self, pop_data: SimulantData) -> None:
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
            },
            index=index,
        )
        self.population_view.initialize(new_children)

    def on_time_step(self, event: Event) -> None:
        if self._sim_step_name() not in [
            SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY,
            SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY,
        ]:
            return

        child_alive = self.population_view.get(event.index, COLUMNS.CHILD_ALIVE)
        alive_children_idx = child_alive.index[child_alive == "alive"]
        # Update age of children to get correct lookup values - use midpoint of age groups
        age_group_midpoints = {
            SIMULATION_EVENT_NAMES.EARLY_NEONATAL_MORTALITY: (7 / 2) / 365.0,
            SIMULATION_EVENT_NAMES.LATE_NEONATAL_MORTALITY: (7 + (28 - 7) / 2) / 365.0,
        }
        new_age = age_group_midpoints[self._sim_step_name()]
        self.population_view.update(
            COLUMNS.CHILD_AGE,
            lambda age: pd.Series(new_age, index=alive_children_idx, name=COLUMNS.CHILD_AGE),
        )
