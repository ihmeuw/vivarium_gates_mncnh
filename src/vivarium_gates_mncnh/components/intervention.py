import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.constants.data_values import (
    COLUMNS,
    CPAP_ACCESS_PROBABILITIES,
    DELIVERY_FACILITY_TYPE_PROBABILITIES,
    DELIVERY_FACILITY_TYPES,
    PIPELINES,
    SIMULATION_EVENT_NAMES,
)
from vivarium_gates_mncnh.utilities import get_location


class CPAPIntervention(Component):
    """Component for CPAP intervention. This is essentially a risk effect."""

    def __init__(self, preterm_csmr_target: str) -> None:
        super().__init__()
        self.preterm_csmr_target = preterm_csmr_target

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = get_location(builder)
        self.with_rds_csmr = builder.value.get_value(self.preterm_csmr_target)
        # TODO: get PAF and RR?
        builder.value.register_value_modifier(
            self.with_rds_csmr.name,
            self.calculate_cpap_path_probability,
            required_resources=[
                self.with_rds_csmr.name,
                COLUMNS.DELIVERY_FACILITY_TYPE,
                COLUMNS.CPAP_AVAILABLE,
            ],
        )

    ##################
    # Helper nethods #
    ##################

    def calculate_cpap_path_probability(self, index: pd.Index, csmr: pd.Series) -> pd.Series:

        pop = self.population_view.get(index)
        # TODO: implement
        # TODO: iterate through each facility time
        # TODO: calculate probability of CPAP path
        # TODO: modify the preterm csmr pipeline
        pass
