from asyncio import Event

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium_public_health.population import BasePopulation, ScaledPopulation
from vivarium_public_health.utilities import to_years

from vivarium_gates_mncnh.constants import data_keys


class AgelessPopulation(ScaledPopulation):
    """A component to handle the population of the model. Simulants will not have their age incremented"""

    def on_time_step(self, event: Event) -> None:
        pass


class EvenlyDistributedPopulation(BasePopulation):
    """
    Component for producing and aging simulants which are initialized with ages
    evenly distributed between age start and age end, and evenly split between
    male and female.
    """

    @property
    def columns_created(self) -> list[str]:
        return [
            "child_age",
            "sex_of_child",
            "child_alive",
            "location",
            "entrance_time",
            "exit_time",
        ]

    def __init__(self):
        super().__init__()
        self._sub_components = []

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.location = builder.data.load(data_keys.POPULATION.LOCATION)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        age_start = pop_data.user_data.get("age_start", self.config.initialization_age_min)
        age_end = pop_data.user_data.get("age_end", self.config.initialization_age_max)

        population = pd.DataFrame(index=pop_data.index)
        population["entrance_time"] = pop_data.creation_time
        population["exit_time"] = pd.NaT
        population["child_alive"] = "alive"
        population["location"] = self.location
        population["child_age"] = np.linspace(
            age_start, age_end, num=len(population) + 1, endpoint=False
        )[1:]
        population["sex_of_child"] = "Female"
        population.loc[population.index % 2 == 1, "sex_of_child"] = "Male"
        self.register_simulants(population[list(self.key_columns)])
        self.population_view.update(population)

    def on_time_step(self, event: Event) -> None:
        """Ages simulants each time step."""
        # This is overwriting for columns
        population = self.population_view.get(event.index, query="child_alive == 'alive'")
        population["child_age"] += to_years(event.step_size)
        self.population_view.update(population)
