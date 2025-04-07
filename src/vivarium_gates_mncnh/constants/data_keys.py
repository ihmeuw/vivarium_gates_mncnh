from typing import NamedTuple

from vivarium_public_health.utilities import TargetString

#############
# Data Keys #
#############

METADATA_LOCATIONS = "metadata.locations"


class __Population(NamedTuple):
    LOCATION: str = "population.location"
    STRUCTURE: str = "population.structure"
    AGE_BINS: str = "population.age_bins"
    DEMOGRAPHY: str = "population.demographic_dimensions"
    TMRLE: str = "population.theoretical_minimum_risk_life_expectancy"
    SCALING_FACTOR: str = "population.scaling_factor"
    ACMR: str = "cause.all_causes.cause_specific_mortality_rate"

    @property
    def name(self):
        return "population"

    @property
    def log_name(self):
        return "population"


POPULATION = __Population()


class __Pregnancy(NamedTuple):
    ASFR: str = "covariate.age_specific_fertility_rate.estimate"
    SBR: str = "covariate.stillbirth_to_live_birth_ratio.estimate"
    RAW_INCIDENCE_RATE_MISCARRIAGE: str = (
        "cause.maternal_abortion_and_miscarriage.raw_incidence_rate"
    )
    RAW_INCIDENCE_RATE_ECTOPIC: str = "cause.ectopic_pregnancy.raw_incidence_rate"

    @property
    def name(self):
        return "pregnancy"

    @property
    def log_name(self):
        return self.name.replace("_", " ")


PREGNANCY = __Pregnancy()


class __LowBirthWeightShortGestation(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    BIRTH_EXPOSURE: str = "risk_factor.low_birth_weight_and_short_gestation.birth_exposure"
    EXPOSURE: str = "risk_factor.low_birth_weight_and_short_gestation.exposure"
    DISTRIBUTION: str = "risk_factor.low_birth_weight_and_short_gestation.distribution"
    CATEGORIES: str = "risk_factor.low_birth_weight_and_short_gestation.categories"
    RELATIVE_RISK: str = "risk_factor.low_birth_weight_and_short_gestation.relative_risk"
    RELATIVE_RISK_INTERPOLATOR: str = (
        "risk_factor.low_birth_weight_and_short_gestation.relative_risk_interpolator"
    )
    PAF: str = (
        "risk_factor.low_birth_weight_and_short_gestation.population_attributable_fraction"
    )

    @property
    def name(self):
        return "low_birth_weight_and_short_gestation"

    @property
    def log_name(self):
        return "low birth weight and short gestation"


LBWSG = __LowBirthWeightShortGestation()


class __ANC(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    ESTIMATE: str = "covariate.antenatal_care_1_visit_coverage_proportion.estimate"

    @property
    def name(self):
        return "ANC"

    @property
    def log_name(self):
        return "anc"


ANC = __ANC()


class __MaternalSepsis(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    RAW_INCIDENCE_RATE: str = (
        "cause.maternal_sepsis_and_other_maternal_infections.incidence_rate"
    )
    CSMR: str = (
        "cause.maternal_sepsis_and_other_maternal_infections.cause_specific_mortality_rate"
    )
    YLD_RATE: str = "cause.maternal_sepsis_and_other_maternal_infections.yld_rate"

    @property
    def name(self):
        return "maternal_sepsis"

    @property
    def log_name(self):
        return "maternal sepsis"


MATERNAL_SEPSIS = __MaternalSepsis()


class __MaternalHemorrhage(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    RAW_INCIDENCE_RATE: str = "cause.maternal_hemorrhage.incidence_rate"
    CSMR: str = "cause.maternal_hemorrhage.cause_specific_mortality_rate"
    YLD_RATE: str = "cause.maternal_hemorrhage.yld_rate"

    @property
    def name(self):
        return "maternal_hemorrhage"

    @property
    def log_name(self):
        return "maternal hemorrhage"


MATERNAL_HEMORRHAGE = __MaternalHemorrhage()


class __ObstructedLabor(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    RAW_INCIDENCE_RATE: str = (
        "cause.maternal_obstructed_labor_and_uterine_rupture.incidence_rate"
    )
    CSMR: str = (
        "cause.maternal_obstructed_labor_and_uterine_rupture.cause_specific_mortality_rate"
    )
    YLD_RATE: str = "cause.maternal_obstructed_labor_and_uterine_rupture.yld_rate"

    @property
    def name(self):
        return "obstructed_labor"

    @property
    def log_name(self):
        return "obstructed labor"


OBSTRUCTED_LABOR = __ObstructedLabor()


class __NeonatalPretermBirth(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    CSMR: str = "cause.neonatal_preterm_birth.cause_specific_mortality_rate"
    PAF: str = "cause.neonatal_preterm_birth.population_attributable_fraction"
    PREVALENCE: str = "cause.neonatal_preterm_birth.prevalence"

    @property
    def name(self):
        return "neonatal_preterm_birth"

    @property
    def log_name(self):
        return "neonatal preterm birth"


PRETERM_BIRTH = __NeonatalPretermBirth()


class __NeonatalSepsis(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    CSMR: str = (
        "cause.neonatal_sepsis_and_other_neonatal_infections.cause_specific_mortality_rate"
    )

    @property
    def name(self):
        return "neonatal_sepsis"

    @property
    def log_name(self):
        return "neonatal sepsis"


NEONATAL_SEPSIS = __NeonatalSepsis()


class __NeonatalEncephalopath(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    CSMR: str = "cause.neonatal_encephalopathy_due_to_birth_asphyxia_and_trauma.cause_specific_mortality_rate"

    @property
    def name(self):
        return "neonatal_encephalopathy"

    @property
    def log_name(self):
        return "neonatal encephalopathy"


NEONATAL_ENCEPHALOPATHY = __NeonatalEncephalopath()


class __NoCPAPRisk(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    P_RDS: str = "intervention.no_cpap_risk.p_rds"
    P_CPAP_BEMONC: str = "intervention.no_cpap_risk.probability_cpap_bemonc"
    P_CPAP_CEMONC: str = "intervention.no_cpap_risk.probability_cpap_cemonc"
    P_CPAP_HOME: str = "intervention.no_cpap_risk.probability_cpap_home"
    RELATIVE_RISK: str = "intervention.no_cpap_risk.relative_risk"
    PAF: str = "intervention.no_cpap_risk.population_attributable_fraction"

    @property
    def name(self):
        return "no_CPAP_intervention"

    @property
    def log_name(self):
        return "no CPAP intervention"


NO_CPAP_RISK = __NoCPAPRisk()


class __FacilityChoice(NamedTuple):
    P_HOME: str = "cause.facility_choice.probability_home_birth"
    P_BEmONC: str = "cause.facility_choice.probability_bemonc_birth"
    P_CEmONC: str = "cause.facility_choice.probability_cemonc_birth"

    @property
    def name(self):
        return "facility_choices"

    @property
    def log_name(self):
        return "facility choices"


FACILITY_CHOICE = __FacilityChoice()


class __NoAntibioticsRisk(NamedTuple):
    P_ANTIBIOTIC_HOME: str = "intervention.no_antibiotics_risk.probability_antibiotics_home"
    P_ANTIBIOTIC_BEMONC: str = (
        "intervention.no_antibiotics_risk.probability_antibiotics_bemonc"
    )
    P_ANTIBIOTIC_CEMONC: str = (
        "intervention.no_antibiotics_risk.probability_antibiotics_cemonc"
    )
    RELATIVE_RISK: str = "intervention.no_antibiotics_risk.relative_risk"
    PAF: str = "intervention.no_antibiotics_risk.population_attributable_fraction"

    @property
    def name(self):
        return "no_antibiotics_risk"

    @property
    def log_name(self):
        return "no antibiotics risk"


NO_ANTIBIOTICS_RISK = __NoAntibioticsRisk()


MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    # TODO: list all key groups here
    PREGNANCY,
    LBWSG,
    ANC,
    MATERNAL_SEPSIS,
    MATERNAL_HEMORRHAGE,
    OBSTRUCTED_LABOR,
    PRETERM_BIRTH,
    NEONATAL_SEPSIS,
    NEONATAL_ENCEPHALOPATHY,
    NO_CPAP_RISK,
    FACILITY_CHOICE,
    NO_ANTIBIOTICS_RISK,
]


REMAP_KEY_GROUPS = [
    LBWSG,
    PRETERM_BIRTH,
    NEONATAL_SEPSIS,
    NEONATAL_ENCEPHALOPATHY,
    NO_CPAP_RISK,
    NO_ANTIBIOTICS_RISK,
]
