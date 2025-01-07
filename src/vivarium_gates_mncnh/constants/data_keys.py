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
]
