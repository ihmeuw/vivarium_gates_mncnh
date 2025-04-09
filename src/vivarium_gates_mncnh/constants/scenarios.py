from dataclasses import dataclass
from typing import NamedTuple

#############
# Scenarios #
#############


@dataclass
class InterventionScenario:
    name: str
    bemonc_cpap_access: str = "baseline"
    cemonc_cpap_access: str = "baseline"
    bemonc_antibiotics_access: str = "baseline"
    cemonc_antibiotics_access: str = "baseline"
    bemonc_probiotics_access: str = "baseline"
    cemonc_probiotics_access: str = "baseline"


class __InterventionScenarios(NamedTuple):
    BASELINE: InterventionScenario = InterventionScenario("baseline")
    # todo add additional intervention scenarios
    FULL_CPAP_BEMONC: InterventionScenario = InterventionScenario(
        "full_cpap_bemonc",
        bemonc_cpap_access="full",
    )
    FULL_CPAP_CEMONC: InterventionScenario = InterventionScenario(
        "full_cpap_cemonc",
        cemonc_cpap_access="full",
    )
    FULL_CPAP_ALL: InterventionScenario = InterventionScenario(
        "full_cpap_all",
        bemonc_cpap_access="full",
        cemonc_cpap_access="full",
    )
    FULL_ANTIBIOTICS_BEMONC: InterventionScenario = InterventionScenario(
        "full_antibiotics_bemonc",
        bemonc_antibiotics_access="full",
    )
    FULL_ANTIBIOTICS_CEMONC: InterventionScenario = InterventionScenario(
        "full_antibiotics_cemonc",
        cemonc_antibiotics_access="full",
    )
    FULL_ANTIBIOTICS_ALL: InterventionScenario = InterventionScenario(
        "full_antibiotics_all",
        bemonc_antibiotics_access="full",
        cemonc_antibiotics_access="full",
    )
    FULL_PROBIOTICS_BEMONC: InterventionScenario = InterventionScenario(
        "full_probiotics_bemonc",
        bemonc_probiotics_access="full",
    )
    FULL_PROBIOTICS_CEMONC: InterventionScenario = InterventionScenario(
        "full_probiotics_cemonc",
        cemonc_probiotics_access="full",
    )
    FULL_PROBIOTICS_ALL: InterventionScenario = InterventionScenario(
        "full_probiotics_all",
        bemonc_probiotics_access="full",
        cemonc_probiotics_access="full",
    )

    def __getitem__(self, item) -> InterventionScenario:
        for scenario in self:
            if scenario.name == item:
                return scenario
        raise KeyError(item)


INTERVENTION_SCENARIOS = __InterventionScenarios()
