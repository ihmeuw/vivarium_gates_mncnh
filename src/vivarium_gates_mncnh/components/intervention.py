import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_mncnh.constants.data_keys import NO_CPAP_INTERVENTION
from vivarium_gates_mncnh.constants.data_values import COLUMNS
from vivarium_gates_mncnh.utilities import get_location


class NoCPAPIntervention(Component):
    """Component for CPAP intervention. This is essentially a risk effect."""

    @property
    def configuration_defaults(self) -> dict:
        return {
            self.name: {
                "data_sources": {
                    "relative_risk": NO_CPAP_INTERVENTION.RELATIVE_RISK,
                    "paf": NO_CPAP_INTERVENTION.PAF,
                }
            }
        }

    @property
    def required_columns(self) -> list[str]:
        return [COLUMNS.CPAP_AVAILABLE]

    def __init__(self, preterm_csmr_target: str) -> None:
        super().__init__()
        self.preterm_csmr_target = preterm_csmr_target

    def setup(self, builder: Builder) -> None:
        self._sim_step_name = builder.time.simulation_event_name()
        self.randomness = builder.randomness.get_stream(self.name)
        self.location = get_location(builder)
        self.preterm_with_rds_csmr = builder.value.get_value(self.preterm_csmr_target)
        builder.value.register_value_modifier(
            self.preterm_with_rds_csmr.name,
            self.modify_preterm_with_rds_csmr,
            required_resources=[
                self.preterm_with_rds_csmr.name,
                COLUMNS.DELIVERY_FACILITY_TYPE,
                COLUMNS.CPAP_AVAILABLE,
            ],
        )

    ##################
    # Helper nethods #
    ##################

    def modify_preterm_with_rds_csmr(
        self, index: pd.Index, preterm_with_rds_csmr: pd.Series
    ) -> pd.Series[float]:
        # No CPAP access is like a dichotomous risk factor, meaning those that have access to CPAP will
        # not have their CSMR modify by no CPAP RR
        pop = self.population_view.get(index)
        has_cpap_idx = pop.index[pop[COLUMNS.CPAP_AVAILABLE] == True]
        no_cpap_idx = pop.index[pop[COLUMNS.CPAP_AVAILABLE] == False]
        no_cpap_rr = self.lookup_tables["relative_risk"](no_cpap_idx)
        has_cpap_paf = self.lookup_tables["paf"](has_cpap_idx)
        no_cpap_paf = self.lookup_tables["paf"](no_cpap_idx)

        has_cpap_modifier = 1 - has_cpap_paf
        no_cpap_modifier = no_cpap_rr * (1 - no_cpap_paf)
        csmr_modifier = pd.concat(
            [
                has_cpap_modifier,
                no_cpap_modifier,
            ]
        ).sort_index()

        modified_csmr = preterm_with_rds_csmr * csmr_modifier
        return modified_csmr
