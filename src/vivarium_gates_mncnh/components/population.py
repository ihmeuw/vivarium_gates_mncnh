from asyncio import Event

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium_public_health.population import BasePopulation, ScaledPopulation
from vivarium_public_health.utilities import to_years

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.data_values import CHILD_INITIALIZATION_AGE


class AgelessPopulation(ScaledPopulation):
    """A component to handle the population of the model. Simulants will not
    have their age incremented. Removes standard Mortality, Disability, and
    AgeOutSimulants sub-components because this model has its own mortality
    system (MaternalDisordersBurden and NeonatalMortality)."""

    def __init__(self, scaling_factor: str | pd.DataFrame):
        super().__init__(scaling_factor)
        self._sub_components = []

    def setup(self, builder: Builder) -> None:
        super().setup(builder)

    def on_time_step(self, event: Event) -> None:
        pass


class EvenlyDistributedPopulation(BasePopulation):
    """
    Component for producing and aging simulants which are initialized with ages
    evenly distributed between age start and age end, and evenly split between
    male and female.
    """

    def __init__(self):
        super().__init__()
        self._sub_components = []

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.location = builder.data.load(data_keys.POPULATION.LOCATION)
        builder.population.register_initializer(
            self.initialize_evenly_distributed_population,
            columns=[
                "child_age",
                "sex_of_child",
                "child_alive",
                "location",
                "entrance_time",
                "exit_time",
            ],
        )

    def initialize_evenly_distributed_population(self, pop_data: SimulantData) -> None:
        population = pd.DataFrame(index=pop_data.index)
        population["entrance_time"] = pop_data.creation_time
        population["exit_time"] = pd.NaT
        population["child_alive"] = "alive"
        population["location"] = self.location
        # NOTE: If ages are initialized less than or equal to CHILD_INITIALIZATION_AGE,
        # those simulants will be mapped to the stillbirth age group, so we must start at that value!
        population["child_age"] = np.linspace(
            CHILD_INITIALIZATION_AGE, 0.005, num=len(population) + 1, endpoint=False
        )[1:]
        population["sex_of_child"] = "Female"
        population.loc[population.index % 2 == 1, "sex_of_child"] = "Male"
        self.register_simulants(population[list(self.key_columns)])
        self.population_view.initialize(population)

    def on_time_step(self, event: Event) -> None:
        """Ages simulants each time step."""
        population = self.population_view.get(
            event.index, ["child_age"], query="child_alive == 'alive'"
        )
        self.population_view.update(
            "child_age",
            lambda age: age + to_years(event.step_size),
        )
