from __future__ import annotations

from functools import partial

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event

from vivarium_gates_mncnh.constants import data_values
from vivarium_gates_mncnh.constants.data_keys import NO_CPAP_RISK
from vivarium_gates_mncnh.constants.data_values import COLUMNS, PIPELINES
from vivarium_gates_mncnh.utilities import get_location


class NoCPAPRisk(Component):
    """Component for CPAP intervention. This is essentially a risk effect."""

    @property
    def configuration_defaults(self) -> dict:
        return {
            self.name: {
                "data_sources": {
                    "relative_risk": NO_CPAP_RISK.RELATIVE_RISK,
                    "paf": self.load_paf_data,
                }
            }
        }

    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.CPAP_AVAILABLE]

    def __init__(self) -> None:
        super().__init__()
        self.preterm_csmr_target = PIPELINES.NEONATAL_PRETERM_BIRTH_WITH_RDS

    def setup(self, builder: Builder) -> None:
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

    def load_paf_data(self, builder: Builder) -> pd.Series:
        data = builder.data.load(NO_CPAP_RISK.PAF)
        data = data.rename(columns=data_values.CHILD_LOOKUP_COLUMN_MAPPER)
        return data

    def modify_preterm_with_rds_csmr(
        self, index: pd.Index, preterm_with_rds_csmr: pd.Series[float]
    ) -> pd.Series[float]:
        # No CPAP access is like a dichotomous risk factor, meaning those that have access to CPAP will
        # not have their CSMR modify by no CPAP RR
        pop = self.population_view.get(index)
        no_cpap_idx = pop.index[pop[COLUMNS.CPAP_AVAILABLE] == False]
        # NOTE: RR is relative risk for no CPAP
        no_cpap_rr = self.lookup_tables["relative_risk"](no_cpap_idx)
        # NOTE: PAF is for no CPAP
        paf = self.lookup_tables["paf"](index)

        # Modify the CSMR pipeline
        modified_csmr = preterm_with_rds_csmr * (1 - paf)
        modified_csmr.loc[no_cpap_idx] = modified_csmr * no_cpap_rr
        return modified_csmr
