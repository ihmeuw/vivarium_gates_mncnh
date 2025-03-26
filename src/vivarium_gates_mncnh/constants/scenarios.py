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


class __InterventionScenarios(NamedTuple):
    BASELINE: InterventionScenario = InterventionScenario("baseline")
    # todo add additional intervention scenarios
    FULL_CPAP_BEMONC: InterventionScenario = InterventionScenario(
        "full_cpap_bemonc",
        "full",
    )
    FULL_CPAP_CEMONC: InterventionScenario = InterventionScenario(
        "full_cpap_cemonc", "baseline", "full"
    )
    FULL_CPAP_ALL: InterventionScenario = InterventionScenario(
        "full_cpap_all",
        "full",
        "full",
    )
    FULL_ANTIBIOTICS_BEMONC: InterventionScenario = InterventionScenario(
        "full_antibiotics_bemonc", "baseline", "baseline", "full", "baseline"
    )
    FULL_ANTIBIOTICS_CEMONC: InterventionScenario = InterventionScenario(
        "full_antibiotics_cemonc", "baseline", "baseline", "baseline", "full"
    )
    FULL_ANTIBIOTICS_ALL: InterventionScenario = InterventionScenario(
        "full_antibiotics_all", "baseline", "baseline", "full", "full"
    )

    def __getitem__(self, item) -> InterventionScenario:
        for scenario in self:
            if scenario.name == item:
                return scenario
        raise KeyError(item)


INTERVENTION_SCENARIOS = __InterventionScenarios()
