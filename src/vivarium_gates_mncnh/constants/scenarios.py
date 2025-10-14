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
    home_antibiotics_access: str = "baseline"
    bemonc_probiotics_access: str = "baseline"
    cemonc_probiotics_access: str = "baseline"
    bemonc_azithromycin_access: str = "baseline"
    cemonc_azithromycin_access: str = "baseline"
    home_misoprostol_access: str = "baseline"
    ultrasound_coverage: str = "baseline"
    standard_ultrasound_coverage: str = "baseline"
    ifa_mms_coverage: str = "baseline"
    hemoglobin_screening_coverage: str = "baseline"
    ferritin_screening_coverage: str = "baseline"


class __InterventionScenarios(NamedTuple):
    BASELINE: InterventionScenario = InterventionScenario("baseline")
    CPAP_ACS_SCALEUP: InterventionScenario = InterventionScenario(
        "cpap_acs_scaleup",
        bemonc_cpap_access="full",
        cemonc_cpap_access="full",
    )
    CPAP_ACS_AI_ULTRASOUND_SCALEUP: InterventionScenario = InterventionScenario(
        "cpap_acs_ai_ultrasound_scaleup",
        ultrasound_coverage="full",
        standard_ultrasound_coverage="none",
        bemonc_cpap_access="full",
        cemonc_cpap_access="full",
    )
    NEONATAL_ANTIBIOTICS_SCALEUP: InterventionScenario = InterventionScenario(
        "neonatal_antibiotics_scaleup",
        bemonc_antibiotics_access="full",
        cemonc_antibiotics_access="full",
        home_antibiotics_access="full",
    )
    NEONATAL_PROBIOTICS_SCALEUP: InterventionScenario = InterventionScenario(
        "neonatal_probiotics_scaleup",
        bemonc_probiotics_access="full",
        cemonc_probiotics_access="full",
    )
    AZITHROMYCIN_SCALEUP: InterventionScenario = InterventionScenario(
        "azithromycin_scaleup",
        bemonc_azithromycin_access="full",
        cemonc_azithromycin_access="full",
    )
    AI_ULTRASOUND_SCALEUP: InterventionScenario = InterventionScenario(
        "ai_ultrasound_scaleup",
        ultrasound_coverage="full",
        standard_ultrasound_coverage="none",
    )
    STANDARD_ULTRASOUND_SCALEUP: InterventionScenario = InterventionScenario(
        "standard_ultrasound_scaleup",
        ultrasound_coverage="full",
        standard_ultrasound_coverage="full",
    )
    FULL_PRODUCT_SCALEUP: InterventionScenario = InterventionScenario(
        "full_product_scaleup",
        ultrasound_coverage="full",
        standard_ultrasound_coverage="none",
        bemonc_cpap_access="full",
        cemonc_cpap_access="full",
        bemonc_probiotics_access="full",
        cemonc_probiotics_access="full",
        bemonc_antibiotics_access="full",
        cemonc_antibiotics_access="full",
        home_antibiotics_access="full",
        bemonc_azithromycin_access="full",
        cemonc_azithromycin_access="full",
    )
    MMS_TOTAL_SCALEUP: InterventionScenario = InterventionScenario(
        "mms_total_scaleup",
        ifa_mms_coverage="mms",
    )
    ULTRASOUND_VV: InterventionScenario = InterventionScenario(
        "ultrasound_vv",
        ultrasound_coverage="full",
        standard_ultrasound_coverage="half",
    )
    ANEMIA_SCREENING_SCALEUP: InterventionScenario = InterventionScenario(
        "anemia_screening_scaleup",
        hemoglobin_screening_coverage="full",
        ferritin_screening_coverage="full",
    )

    def __getitem__(self, item) -> InterventionScenario:
        for scenario in self:
            if scenario.name == item:
                return scenario
        raise KeyError(item)


INTERVENTION_SCENARIOS = __InterventionScenarios()
