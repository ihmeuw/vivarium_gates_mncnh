from asyncio import Event

from vivarium_public_health.population import ScaledPopulation


class AgelessPopulation(ScaledPopulation):
    """A component to handle the population of the model. Simulants will not have their age incremented"""

    def on_time_step(self, event: Event) -> None:
        pass
