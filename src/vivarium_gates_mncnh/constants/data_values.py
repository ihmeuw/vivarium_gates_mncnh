from datetime import datetime
from typing import NamedTuple

from vivarium_gates_mncnh.constants.data_keys import (
    FACILITY_CHOICE,
    IFA_SUPPLEMENTATION,
    MMN_SUPPLEMENTATION,
    NO_AZITHROMYCIN_RISK,
    NO_CPAP_RISK,
)
from vivarium_gates_mncnh.utilities import (
    get_lognorm_from_quantiles,
    get_norm,
    get_truncnorm,
    get_uniform_distribution_from_limits,
)

##########################
# Constant scalar values #
##########################

# Threshold for children to be considered underweight (in grams)
LOW_BIRTH_WEIGHT_THRESHOLD = 2500

# Ages (in years) for neonatal period
EARLY_NEONATAL_AGE_START = 0.0  # 0 days
LATE_NEONATAL_AGE_START = 0.01917808  # 7 days
LATE_NEONATAL_AGE_END = 0.07671233  # 28 days


class __PregnancyOutcome(NamedTuple):
    PARTIAL_TERM_OUTCOME = "partial_term"
    LIVE_BIRTH_OUTCOME = "live_birth"
    STILLBIRTH_OUTCOME = "stillbirth"
    INVALID_OUTCOME = "invalid"  ## For sex of partial births


PREGNANCY_OUTCOMES = __PregnancyOutcome()


class _Durations(NamedTuple):
    PARTIAL_TERM_LOWER_WEEKS = 6.0
    PARTIAL_TERM_UPPER_WEEKS = 24.0


DURATIONS = _Durations()


NUM_DRAWS = 250


class _SimulationEventNames(NamedTuple):
    # Constants for the simulation events. Used for string comparison in components.
    PREGNANCY = "pregnancy"
    DELIVERY_FACILITY = "delivery_facility"
    AZITHROMYCIN_ACCESS = "azithromycin_access"
    MISOPROSTOL_ACCESS = "misoprostol_access"
    CPAP_ACCESS = "cpap_access"
    ACS_ACCESS = "acs_access"
    ANTIBIOTICS_ACCESS = "antibiotics_access"
    PROBIOTICS_ACCESS = "probiotics_access"
    MATERNAL_SEPSIS = "maternal_sepsis_and_other_maternal_infections"
    MATERNAL_HEMORRHAGE = "maternal_hemorrhage"
    OBSTRUCTED_LABOR = "maternal_obstructed_labor_and_uterine_rupture"
    POSTPARTUM_DEPRESSION = "postpartum_depression"
    MORTALITY = "mortality"
    EARLY_NEONATAL_MORTALITY = "early_neonatal_mortality"
    LATE_NEONATAL_MORTALITY = "late_neonatal_mortality"


SIMULATION_EVENT_NAMES = _SimulationEventNames()


class __UltrasoundTypes(NamedTuple):
    STANDARD: str = "standard"
    NO_ULTRASOUND: str = "no_ultrasound"
    AI_ASSISTED: str = "AI_assisted"


ULTRASOUND_TYPES = __UltrasoundTypes()


class __ANCAttendanceTypes(NamedTuple):
    NONE: str = "none"
    LATER_PREGNANCY_ONLY: str = "later_pregnancy_only"
    FIRST_TRIMESTER_ONLY: str = "first_trimester_only"
    FIRST_TRIMESTER_AND_LATER_PREGNANCY: str = "first_trimester_and_later_pregnancy"


ANC_ATTENDANCE_TYPES = __ANCAttendanceTypes()


class __HemoglobinTestResults(NamedTuple):
    LOW: str = "low"
    ADEQUATE: str = "adequate"


HEMOGLOBIN_TEST_RESULTS = __HemoglobinTestResults()


class __ANCRates(NamedTuple):
    RECEIVED_ULTRASOUND = {
        # https://vivarium-research.readthedocs.io/en/latest/models/concept_models/vivarium_mncnh_portfolio/ai_ultrasound_module/module_document.html#id7
        "Ethiopia": 0.607,
        "Nigeria": 0.587,
        "Pakistan": 0.667,
    }
    # https://vivarium-research.readthedocs.io/en/latest/models/concept_models/vivarium_mncnh_portfolio/ai_ultrasound_module/module_document.html#baseline-coverage
    ULTRASOUND_TYPE = {
        "Ethiopia": {
            ULTRASOUND_TYPES.STANDARD: 1.0,
            ULTRASOUND_TYPES.AI_ASSISTED: 0.0,
        },
        "Nigeria": {
            ULTRASOUND_TYPES.STANDARD: 1.0,
            ULTRASOUND_TYPES.AI_ASSISTED: 0.0,
        },
        "Pakistan": {
            ULTRASOUND_TYPES.STANDARD: 1.0,
            ULTRASOUND_TYPES.AI_ASSISTED: 0.0,
        },
    }
    # https://vivarium-research.readthedocs.io/en/latest/models/concept_models/vivarium_mncnh_portfolio/ai_ultrasound_module/module_document.html#calculation-of-estimated-gestational-age
    STATED_GESTATIONAL_AGE_STANDARD_DEVIATION = {
        ULTRASOUND_TYPES.NO_ULTRASOUND: 10.0 / 7,
        ULTRASOUND_TYPES.STANDARD: 6.7 / 7,
        ULTRASOUND_TYPES.AI_ASSISTED: 5.0 / 7,
    }


ANC_RATES = __ANCRates()


class __Columns(NamedTuple):
    TRACKED = "tracked"
    EXIT_TIME = "exit_time"
    MOTHER_SEX = "sex"
    MOTHER_ALIVE = "alive"
    CHILD_ALIVE = "child_alive"
    MOTHER_AGE = "age"
    CHILD_AGE = "child_age"
    MOTHER_CAUSE_OF_DEATH = "cause_of_death"
    CHILD_CAUSE_OF_DEATH = "child_cause_of_death"
    MOTHER_YEARS_OF_LIFE_LOST = "years_of_life_lost"
    CHILD_YEARS_OF_LIFE_LOST = "child_years_of_life_lost"
    LOCATION = "location"
    PREGNANCY_OUTCOME = "pregnancy_outcome"
    SEX_OF_CHILD = "sex_of_child"
    BIRTH_WEIGHT_EXPOSURE = "birth_weight_exposure"
    GESTATIONAL_AGE_EXPOSURE = "gestational_age_exposure"
    ANC_STATE = "anc_state"
    ANC_ATTENDANCE = "anc_attendance"
    FIRST_TRIMESTER_ANC = "first_trimester_anc"
    LATER_PREGNANCY_ANC = "later_pregnancy_anc"
    DELIVERY_FACILITY_TYPE = "delivery_facility_type"
    ULTRASOUND_TYPE = "ultrasound_type"
    STATED_GESTATIONAL_AGE = "stated_gestational_age"
    MATERNAL_SEPSIS = "maternal_sepsis_and_other_maternal_infections"
    MATERNAL_HEMORRHAGE = "maternal_hemorrhage"
    OBSTRUCTED_LABOR = "maternal_obstructed_labor_and_uterine_rupture"
    CPAP_AVAILABLE = "cpap_available"
    ACS_AVAILABLE = "acs_available"
    ANTIBIOTICS_AVAILABLE = "antibiotics_available"
    PARTIAL_TERM_PREGNANCY_DURATION = "partial_term_pregnancy_duration"
    PROBIOTICS_AVAILABLE = "probiotics_available"
    AZITHROMYCIN_AVAILABLE = "azithromycin_available"
    MISOPROSTOL_AVAILABLE = "misoprostol_available"
    ORAL_IRON_INTERVENTION = "oral_iron_intervention"
    POSTPARTUM_DEPRESSION = "postpartum_depression"
    POSTPARTUM_DEPRESSION_CASE_TYPE = "postpartum_depression_case_type"
    POSTPARTUM_DEPRESSION_CASE_DURATION = "postpartum_depression_case_duration"
    HEMOGLOBIN_SCREENING_COVERAGE = "hemoglobin_screening_coverage"
    FERRITIN_SCREENING_COVERAGE = "ferritin_screening_coverage"
    TESTED_HEMOGLOBIN = "tested_hemoglobin"
    TESTED_FERRITIN = "tested_ferritin"
    ANEMIA_STATUS_DURING_PREGNANCY = "anemia_status_during_pregnancy"


COLUMNS = __Columns()


# TODO: add other maternal disorders when implemented
MATERNAL_DISORDERS = [
    COLUMNS.OBSTRUCTED_LABOR,
    COLUMNS.MATERNAL_HEMORRHAGE,
    COLUMNS.MATERNAL_SEPSIS,
]


CHILD_LOOKUP_COLUMN_MAPPER = {
    "sex": COLUMNS.SEX_OF_CHILD,
    "age_start": "child_age_start",
    "age_end": "child_age_end",
}


class __NeonatalCauses(NamedTuple):
    PRETERM_BIRTH_WITH_RDS = "neonatal_preterm_birth_with_rds"
    PRETERM_BIRTH_WITHOUT_RDS = "neonatal_preterm_birth_without_rds"
    NEONATAL_SEPSIS = "neonatal_sepsis_and_other_neonatal_infections"
    NEONATAL_ENCEPHALOPATHY = "neonatal_encephalopathy_due_to_birth_asphyxia_and_trauma"


NEONATAL_CAUSES = __NeonatalCauses()
CAUSES_OF_NEONATAL_MORTALITY = [
    NEONATAL_CAUSES.PRETERM_BIRTH_WITH_RDS,
    NEONATAL_CAUSES.PRETERM_BIRTH_WITHOUT_RDS,
    NEONATAL_CAUSES.NEONATAL_SEPSIS,
    NEONATAL_CAUSES.NEONATAL_ENCEPHALOPATHY,
]


class __Pipelines(NamedTuple):
    LBWSG_ACMR_PAF_MODIFIER = "lbwsg_paf_on_all_causes.all_cause_mortality_risk.paf"
    ACMR = "all_causes.all_cause_mortality_risk"
    ACMR_PAF = "all_causes.all_cause_mortality_risk.paf"
    DEATH_IN_AGE_GROUP_PROBABILITY = "death_in_age_group_probability"
    NEONATAL_PRETERM_BIRTH_WITH_RDS = (
        "neonatal_preterm_birth_with_rds.cause_specific_mortality_risk"
    )
    PRETERM_WITH_RDS_FINAL_CSMR = "neonatal_preterm_birth_with_rds.csmr"
    NEONATAL_PRETERM_BIRTH_WITHOUT_RDS = (
        "neonatal_preterm_birth_without_rds.cause_specific_mortality_risk"
    )
    NEONATAL_SEPSIS = (
        "neonatal_sepsis_and_other_neonatal_infections.cause_specific_mortality_risk"
    )
    NEONATAL_ENCEPHALOPATHY = "neonatal_encephalopathy_due_to_birth_asphyxia_and_trauma.cause_specific_mortality_risk"
    PRETERM_WITHOUT_RDS_FINAL_CSMR = "neonatal_preterm_birth_without_rds.csmr"
    NEONATAL_SEPSIS_FINAL_CSMR = "neonatal_sepsis_and_other_neonatal_infections.csmr"
    NEONATAL_ENCEPHALOPATHY_FINAL_CSMR = (
        "neonatal_encephalopathy_due_to_birth_asphyxia_and_trauma.csmr"
    )
    PRETERM_WITH_RDS_RR = "effect_of_risk_factor.low_birth_weight_and_short_gestation_on_neonatal_preterm_birth_with_rds.relative_risk"
    PRETERM_WITHOUT_RDS_RR = "effect_of_risk_factor.low_birth_weight_and_short_gestation_on_neonatal_preterm_birth_without_rds.relative_risk"
    NEONATAL_SEPSIS_RR = "effect_of_risk_factor.low_birth_weight_and_short_gestation_on_neonatal_sepsis_and_other_neonatal_infections.relative_risk"
    NEONATAL_ENCEPHALOPATHY_RR = "effect_of_risk_factor.low_birth_weight_and_short_gestation_on_neonatal_encephalopathy_due_to_birth_asphyxia_and_trauma.relative_risk"
    ACMR_RR = "effect_of_low_birth_weight_and_short_gestation_on_all_causes.relative_risk"
    BIRTH_WEIGHT_EXPOSURE = "birth_weight.birth_exposure"
    GESTATIONAL_AGE_EXPOSURE = "gestational_age.birth_exposure"
    RAW_BIRTH_WEIGHT_EXPOSURE = "raw_birth_weight.birth_exposure"
    RAW_GESTATIONAL_AGE_EXPOSURE = "raw_gestational_age.birth_exposure"
    PREGNANCY_DURATION = "pregnancy_duration"
    BIRTH_OUTCOME_PROBABILITIES = "birth_outcome_probabilities"
    MATERNAL_SEPSIS_INCIDENCE_RISK = (
        "maternal_sepsis_and_other_maternal_infections.incidence_risk"
    )
    MATERNAL_HEMORRHAGE_INCIDENCE_RISK = "maternal_hemorrhage.incidence_risk"
    IFA_SUPPLEMENTATION = "iron_folic_acid_supplementation.exposure"
    MMN_SUPPLEMENTATION = "multiple_micronutrient_supplementation.exposure"
    HEMOGLOBIN_EXPOSURE = "hemoglobin.exposure"
    FIRST_ANC_HEMOGLOBIN_EXPOSURE = "first_anc_hemoglobin.exposure"
    IFA_DELETED_HEMOGLOBIN_EXPOSURE = "ifa_deleted_hemoglobin.exposure"


PIPELINES = __Pipelines()


# https://vivarium-research.readthedocs.io/en/latest/models/causes/neonatal/preterm_birth.html#id5
PRETERM_DEATHS_DUE_TO_RDS_PROBABILITY = 0.85
# NOTE: This is an arbitrary value that must be greater than 0 and less than LATE_NEONATAL_AGE_START.
# It is used in a tricky-to-understand way to deal with stillbirths.
# Basically, to record YLLs due to stillbirths, we make stillbirth an "age group" for Vivarium purposes,
# which spans from EARLY_NEONATAL_AGE_START (zero) to CHILD_INITIALIZATION_AGE.
# Then when simulants are initialized, we put them in this stillbirth age group; if they are stillborn,
# we have them die in that age group.
# If they are a live birth, their age is updated to be greater than CHILD_INITIALIZATION_AGE so that they formally enter
# the early neonatal age group.
# It is a bit tricky to think through when we need to modify our approach to match this.
# For example, we leave all early neonatal PAFs as starting at 0, which is fine because the early neonatal
# period in GBD (0-7 days) is a superset of the early neonatal period in our sim (0.1 days-7 days) and
# we don't apply any risk effects that would use a PAF to stillbirths.
CHILD_INITIALIZATION_AGE = 0.1 / 365.0


class __DeliveryFacilityTypes(NamedTuple):
    HOME = "home"
    CEmONC = "CEmONC"
    BEmONC = "BEmONC"
    NONE = "none"


DELIVERY_FACILITY_TYPES = __DeliveryFacilityTypes()


DELIVERY_FACILITY_TYPE_PROBABILITIES = {
    "Ethiopia": {
        FACILITY_CHOICE.BEmONC_FACILITY_FRACTION: 0.160883,
    },
    "Nigeria": {
        FACILITY_CHOICE.BEmONC_FACILITY_FRACTION: 0.004423,
    },
    "Pakistan": {
        FACILITY_CHOICE.BEmONC_FACILITY_FRACTION: 0.340528,
    },
}
# Probability each of these facility types has access to CPAP
CPAP_ACCESS_PROBABILITIES = {
    "Ethiopia": {
        # https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/neonatal/cpap_intervention.html#baseline-coverage-and-rr-data
        NO_CPAP_RISK.P_CPAP_BEMONC: get_norm(0.075, 0.02**2),
        NO_CPAP_RISK.P_CPAP_CEMONC: get_norm(0.393, 0.05**2),
        NO_CPAP_RISK.P_CPAP_HOME: get_norm(0.0, 0.00**2),
    },
    # https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/neonatal/cpap_intervention.html#baseline-coverage-and-rr-data
    "Nigeria": {
        NO_CPAP_RISK.P_CPAP_BEMONC: get_norm(0.075, 0.02**2),
        NO_CPAP_RISK.P_CPAP_CEMONC: get_norm(0.393, 0.05**2),
        NO_CPAP_RISK.P_CPAP_HOME: get_norm(0.0, 0.00**2),
    },
    # https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/neonatal/cpap_intervention.html#baseline-coverage-and-rr-data
    "Pakistan": {
        NO_CPAP_RISK.P_CPAP_BEMONC: get_norm(0.075, 0.02**2),
        NO_CPAP_RISK.P_CPAP_CEMONC: get_norm(0.393, 0.05**2),
        NO_CPAP_RISK.P_CPAP_HOME: get_norm(0.0, 0.00**2),
    },
}
# https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/neonatal/cpap_intervention.html#id7
CPAP_RELATIVE_RISK_DISTRIBUTION = get_lognorm_from_quantiles(0.53, 0.34, 0.83)

# https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/intrapartum/acs_intervention.html#id22
ACS_RELATIVE_RISK_DISTRIBUTION = get_lognorm_from_quantiles(0.84, 0.72, 0.97)

# https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/neonatal/antibiotics_intervention.html#id32
# Model 8.3+ sets coverage values at location level and not birth facility/location level
ANTIBIOTIC_BASELINE_COVERAGE = {
    "Ethiopia": 0.5,
    "Nigeria": 0.0,
    "Pakistan": 0.0,
}
ANTIBIOTIC_RELATIVE_RISK_DISTRIBUTION = get_lognorm_from_quantiles(0.78, 0.60, 1.00)

# http://vivarium-research.readthedocs.io/en/latest/models/intervention_models/neonatal/probiotics_intervention.html#baseline-coverage-data
PROBIOTICS_BASELINE_COVERAGE_PROABILITY = 0.0
# https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/neonatal/probiotics_intervention.html#id14
PROBIOTICS_RELATIVE_RISK_DISTRIBUTION = get_lognorm_from_quantiles(0.81, 0.72, 0.91)


class __Interventions(NamedTuple):
    CPAP: str = "cpap"
    ACS: str = "acs"
    ANTIBIOTICS: str = "antibiotics"
    PROBIOTICS: str = "probiotics"
    AZITHROMYCIN: str = "azithromycin"
    MISOPROSTOL: str = "misoprostol"
    IV_IRON: str = "iv_iron"


INTERVENTIONS = __Interventions()
INTERVENTION_TYPE_MAPPER = {
    INTERVENTIONS.CPAP: "neonatal",
    INTERVENTIONS.ANTIBIOTICS: "neonatal",
    INTERVENTIONS.PROBIOTICS: "neonatal",
    INTERVENTIONS.AZITHROMYCIN: "maternal",
    INTERVENTIONS.MISOPROSTOL: "maternal",
    INTERVENTIONS.IV_IRON: "maternal",
}


AZITHROMYCIN_FACILITY_TYPE_DISTRIBUTION = {
    # https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/intrapartum/azithromycin_intervention.html#id12
    "Ethiopia": {
        NO_AZITHROMYCIN_RISK.P_AZITHROMYCIN_HOME: get_norm(0.0, 0.00**2),
        NO_AZITHROMYCIN_RISK.P_AZITHROMYCIN_BEMONC: get_norm(0.0, 0.00**2),
        NO_AZITHROMYCIN_RISK.P_AZITHROMYCIN_CEMONC: get_norm(0.0, 0.00**2),
    },
    "Nigeria": {
        NO_AZITHROMYCIN_RISK.P_AZITHROMYCIN_HOME: get_norm(0.0, 0.00**2),
        NO_AZITHROMYCIN_RISK.P_AZITHROMYCIN_BEMONC: get_norm(0.0, 0.00**2),
        NO_AZITHROMYCIN_RISK.P_AZITHROMYCIN_CEMONC: get_norm(0.0, 0.00**2),
    },
    "Pakistan": {
        NO_AZITHROMYCIN_RISK.P_AZITHROMYCIN_HOME: get_norm(0.0, 0.00**2),
        NO_AZITHROMYCIN_RISK.P_AZITHROMYCIN_BEMONC: get_norm(0.0, 0.00**2),
        NO_AZITHROMYCIN_RISK.P_AZITHROMYCIN_CEMONC: get_uniform_distribution_from_limits(
            0.153, 0.253
        ),
    },
}
# RR of no azithromycin intervention
# https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/intrapartum/azithromycin_intervention.html#id13
AZITHROMYCIN_RELATIVE_RISK_DISTRIBUTION = get_lognorm_from_quantiles(1.54, 1.30, 1.82)
# https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/intrapartum/misoprostol_intervention.html#id17
MISOPROSTOL_RELATIVE_RISK_DISTRIBUTION = get_lognorm_from_quantiles(0.61, 0.50, 0.74)

# Effects of IV iron intervention
IV_IRON_HEMOGLOBIN_EFFECT_SIZE = {
    # see research documentation here:  https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/mncnh_pregnancy/iv_iron_antenatal/iv_iron_mncnh.html#id16
    "Ethiopia": get_norm(20.2, ninety_five_pct_confidence_interval=(18.9, 21.5)),
    "Nigeria": get_norm(20.2, ninety_five_pct_confidence_interval=(18.9, 21.5)),
    "Pakistan": get_norm(26.3, ninety_five_pct_confidence_interval=(25.7, 26.9)),
}

ORAL_IRON_EFFECT_SIZES = {
    IFA_SUPPLEMENTATION.EFFECT_SIZE: {
        # https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/mncnh_pregnancy/oral_iron_antenatal/oral_iron_antenatal.html#id24
        "hemoglobin.exposure": get_norm(
            9.53, ninety_five_pct_confidence_interval=(6.99, 12.06)
        ),
        # https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/mncnh_pregnancy/oral_iron_antenatal/oral_iron_antenatal.html#id26
        "birth_weight.birth_exposure": get_norm(
            57.73, ninety_five_pct_confidence_interval=(7.66, 107.79)
        ),
    },
    MMN_SUPPLEMENTATION.EFFECT_SIZE: {
        # https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/mncnh_pregnancy/oral_iron_antenatal/oral_iron_antenatal.html#id26
        "birth_weight.birth_exposure": get_norm(
            45.16, ninety_five_pct_confidence_interval=(32.31, 58.02)
        )
    },
    # https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/mncnh_pregnancy/oral_iron_antenatal/oral_iron_antenatal.html#id31
    MMN_SUPPLEMENTATION.STILLBIRTH_RR: {
        "stillbirth": get_lognorm_from_quantiles(0.91, 0.86, 0.98)
    },
}


# Postpartum depression constants
# https://vivarium-research.readthedocs.io/en/latest/models/causes/maternal_disorders/gbd_2021_mncnh/postpartum_depression.html#id17
POSTPARTUM_DEPRESSION_INCIDENCE_RISK = get_truncnorm(
    0.12,
    ninety_five_pct_confidence_interval=(0.04, 0.20),
    lower_clip=0.0,
    upper_clip=1.0,
)
# Case duration is in years
# https://vivarium-research.readthedocs.io/en/latest/models/causes/maternal_disorders/gbd_2021_mncnh/postpartum_depression.html#id17
POSTPARTUM_DEPRESSION_CASE_DURATION = get_truncnorm(
    0.65,
    ninety_five_pct_confidence_interval=(0.59, 0.70),
    lower_clip=0.0,
)


class __PostpartumDepressionCaseTypes(NamedTuple):
    NONE: str = "none"
    ASYMPTOMATIC: str = "asymptomatic"
    MILD: str = "mild"
    MODERATE: str = "moderate"
    SEVERE: str = "severe"


POSTPARTUM_DEPRESSION_CASE_TYPES = __PostpartumDepressionCaseTypes()


# https://vivarium-research.readthedocs.io/en/latest/models/causes/maternal_disorders/gbd_2021_mncnh/postpartum_depression.html#id18
POSTPARTUM_DEPRESSION_CASE_SEVERITY_PROBABILITIES = {
    POSTPARTUM_DEPRESSION_CASE_TYPES.ASYMPTOMATIC: 0.14,
    POSTPARTUM_DEPRESSION_CASE_TYPES.MILD: 0.59,
    POSTPARTUM_DEPRESSION_CASE_TYPES.MODERATE: 0.17,
    POSTPARTUM_DEPRESSION_CASE_TYPES.SEVERE: 0.10,
}
# https://vivarium-research.readthedocs.io/en/latest/models/causes/maternal_disorders/gbd_2021_mncnh/postpartum_depression.html#id18
POSTPARTUM_DEPRESSION_CASE_SEVERITY_DISABILITY_WEIGHTS = {
    POSTPARTUM_DEPRESSION_CASE_TYPES.ASYMPTOMATIC: get_truncnorm(
        0.0,
        0.00**2,
        lower_clip=0,
        upper_clip=1,
    ),
    POSTPARTUM_DEPRESSION_CASE_TYPES.MILD: get_truncnorm(
        0.145,
        ninety_five_pct_confidence_interval=(0.099, 0.209),
        lower_clip=0,
        upper_clip=1,
    ),
    POSTPARTUM_DEPRESSION_CASE_TYPES.MODERATE: get_truncnorm(
        0.396,
        ninety_five_pct_confidence_interval=(0.267, 0.531),
        lower_clip=0,
        upper_clip=1,
    ),
    POSTPARTUM_DEPRESSION_CASE_TYPES.SEVERE: get_truncnorm(
        0.658,
        ninety_five_pct_confidence_interval=(0.477, 0.807),
        lower_clip=0,
        upper_clip=1,
    ),
}


# Hemoglobin constants
HEMOGLOBIN_DISTRIBUTION = "ensemble"
X_VALUES = {"min": 40, "max": 150}
HEMOGLOBIN_ENSEMBLE_DISTRIBUTION_WEIGHTS = {
    "parameter": [
        "gamma",
        "mgumbel",
        "betasr",
        "exp",
        "gumbel",
        "invgamma",
        "invweibull",
        "llogis",
        "lnorm",
        "mgamma",
        "norm",
        "weibull",
    ],
    "value": [0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
# https://vivarium-research.readthedocs.io/en/latest/models/intervention_models/mncnh_pregnancy/anemia_screening.html#hemoglobin-screening-accuracy-instructions
HEMOGLOBIN_TEST_SENSITIVITY = 0.85  # low hemoglobin that tests low
HEMOGLOBIN_TEST_SPECIFICITY = 0.8  # adequate hemoglobin that tests adequate
LOW_HEMOGLOBIN_THRESHOLD = 100

ANEMIA_THRESHOLDS = [70, 100, 110]  # ordering is severe, moderate, mild
