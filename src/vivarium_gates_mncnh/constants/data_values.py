from datetime import datetime
from typing import NamedTuple

from vivarium_gates_mncnh.constants.data_keys import (
    FACILITY_CHOICE,
    NO_ANTIBIOTICS_RISK,
    NO_AZITHROMYCIN_RISK,
    NO_CPAP_RISK,
)
from vivarium_gates_mncnh.utilities import (
    get_lognorm_from_quantiles,
    get_norm,
    get_truncnorm,
    get_uniform_distribution_from_limits,
)

############################
# Disease Model Parameters #
############################

REMISSION_RATE = 0.1
MEAN_SOJOURN_TIME = 10


##############################
# Screening Model Parameters #
##############################

PROBABILITY_ATTENDING_SCREENING_KEY = "probability_attending_screening"
PROBABILITY_ATTENDING_SCREENING_START_MEAN = 0.25
PROBABILITY_ATTENDING_SCREENING_START_STDDEV = 0.0025
PROBABILITY_ATTENDING_SCREENING_END_MEAN = 0.5
PROBABILITY_ATTENDING_SCREENING_END_STDDEV = 0.005

FIRST_SCREENING_AGE = 21
MID_SCREENING_AGE = 30
LAST_SCREENING_AGE = 65


###################################
# Scale-up Intervention Constants #
###################################
SCALE_UP_START_DT = datetime(2021, 1, 1)
SCALE_UP_END_DT = datetime(2030, 1, 1)
SCREENING_SCALE_UP_GOAL_COVERAGE = 0.50
SCREENING_SCALE_UP_DIFFERENCE = (
    SCREENING_SCALE_UP_GOAL_COVERAGE - PROBABILITY_ATTENDING_SCREENING_START_MEAN
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
    FULL_TERM_DAYS = 40 * 7
    POSTPARTUM_DAYS = 6 * 7
    PARTURITION_DAYS = 1 * 7
    DETECTION_DAYS = 6 * 7
    PARTIAL_TERM_DAYS = 24 * 7
    INTERVENTION_DELAY_DAYS = 8 * 7
    PARTIAL_TERM_LOWER_WEEKS = 6.0
    PARTIAL_TERM_UPPER_WEEKS = 24.0


DURATIONS = _Durations()


INFANT_MALE_PERCENTAGES = {
    "Ethiopia": 0.514271,
    "Nigeria": 0.511785,
    "Pakistan": 0.514583,
}


NUM_DRAWS = 500


class _SimulationEventNames(NamedTuple):
    # Constants for the simulation events. Used for string comparison in components.
    PREGNANCY = "pregnancy"
    DELIVERY_FACILITY = "delivery_facility"
    AZITHROMYCIN_ACCESS = "azithromycin_access"
    MISOPROSTOL_ACCESS = "misoprostol_access"
    CPAP_ACCESS = "cpap_access"
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


class __ANCRates(NamedTuple):
    ATTENDED_CARE_FACILITY = {
        "Ethiopia": 0.757,
        "Nigeria": 0.743,
        "Pakistan": 0.908,
    }
    RECEIVED_ULTRASOUND = {
        "Ethiopia": 0.607,
        "Nigeria": 0.587,
        "Pakistan": 0.667,
    }
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
    SUCCESSFUL_LBW_IDENTIFICATION = {
        ULTRASOUND_TYPES.NO_ULTRASOUND: 0.10,
        ULTRASOUND_TYPES.STANDARD: 0.61,
        ULTRASOUND_TYPES.AI_ASSISTED: 0.80,
    }
    STATED_GESTATIONAL_AGE_STANDARD_DEVIATION = {
        ULTRASOUND_TYPES.NO_ULTRASOUND: 45.5 / 7,
        ULTRASOUND_TYPES.STANDARD: 20.0 / 7,
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
    ATTENDED_CARE_FACILITY = "attended_care_facility"
    DELIVERY_FACILITY_TYPE = "delivery_facility_type"
    ULTRASOUND_TYPE = "ultrasound_type"
    STATED_GESTATIONAL_AGE = "stated_gestational_age"
    SUCCESSFUL_LBW_IDENTIFICATION = "successful_lbw_identification"
    ANC_STATE = "anc_state"
    MATERNAL_SEPSIS = "maternal_sepsis_and_other_maternal_infections"
    MATERNAL_HEMORRHAGE = "maternal_hemorrhage"
    OBSTRUCTED_LABOR = "maternal_obstructed_labor_and_uterine_rupture"
    CPAP_AVAILABLE = "cpap_available"
    ANTIBIOTICS_AVAILABLE = "antibiotics_available"
    PARTIAL_TERM_PREGNANCY_DURATION = "partial_term_pregnancy_duration"
    PROBIOTICS_AVAILABLE = "probiotics_available"
    AZITHROMYCIN_AVAILABLE = "azithromycin_available"
    MISOPROSTOL_AVAILABLE = "misoprostol_available"
    POSTPARTUM_DEPRESSION = "postpartum_depression"
    POSTPARTUM_DEPRESSION_CASE_TYPE = "postpartum_depression_case_type"
    POSTPARTUM_DEPRESSION_CASE_DURATION = "postpartum_depression_case_duration"


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
    PREGNANCY_DURATION = "pregnancy_duration"
    BIRTH_OUTCOME_PROBABILITIES = "birth_outcome_probabilities"
    MATERNAL_SEPSIS_INCIDENCE_RISK = (
        "maternal_sepsis_and_other_maternal_infections.incidence_risk"
    )
    MATERNAL_HEMORRHAGE_INCIDENCE_RISK = "maternal_hemorrhage.incidence_risk"


PIPELINES = __Pipelines()


PRETERM_DEATHS_DUE_TO_RDS_PROBABILITY = 0.85
CHILD_INITIALIZATION_AGE = 0.1 / 365.0


class __DeliveryFacilityTypes(NamedTuple):
    HOME = "home"
    CEmONC = "CEmONC"
    BEmONC = "BEmONC"
    NONE = "none"


DELIVERY_FACILITY_TYPES = __DeliveryFacilityTypes()


DELIVERY_FACILITY_TYPE_PROBABILITIES = {
    "Ethiopia": {
        FACILITY_CHOICE.P_HOME: 0.683,
        FACILITY_CHOICE.P_CEmONC: 0.266,
        FACILITY_CHOICE.P_BEmONC: 0.051,
    },
    "Nigeria": {
        FACILITY_CHOICE.P_HOME: 0.683,
        FACILITY_CHOICE.P_CEmONC: 0.266,
        FACILITY_CHOICE.P_BEmONC: 0.051,
    },
    "Pakistan": {
        FACILITY_CHOICE.P_HOME: 0.683,
        FACILITY_CHOICE.P_CEmONC: 0.266,
        FACILITY_CHOICE.P_BEmONC: 0.051,
    },
}
# Probability each of these facility types has access to CPAP
CPAP_ACCESS_PROBABILITIES = {
    "Ethiopia": {
        NO_CPAP_RISK.P_CPAP_BEMONC: get_norm(0.075, 0.02**2),
        NO_CPAP_RISK.P_CPAP_CEMONC: get_norm(0.393, 0.05**2),
        NO_CPAP_RISK.P_CPAP_HOME: get_norm(0.0, 0.00**2),
    },
    "Nigeria": {
        NO_CPAP_RISK.P_CPAP_BEMONC: get_norm(0.075, 0.02**2),
        NO_CPAP_RISK.P_CPAP_CEMONC: get_norm(0.393, 0.05**2),
        NO_CPAP_RISK.P_CPAP_HOME: get_norm(0.0, 0.00**2),
    },
    "Pakistan": {
        NO_CPAP_RISK.P_CPAP_BEMONC: get_norm(0.075, 0.02**2),
        NO_CPAP_RISK.P_CPAP_CEMONC: get_norm(0.393, 0.05**2),
        NO_CPAP_RISK.P_CPAP_HOME: get_norm(0.0, 0.00**2),
    },
}
CPAP_RELATIVE_RISK_DISTRIBUTION = get_lognorm_from_quantiles(0.53, 0.34, 0.83)


ANTIBIOTIC_FACILITY_TYPE_DISTRIBUTION = {
    # NOTE: This is not being used as of model 8.3
    "Ethiopia": {
        NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_HOME: get_uniform_distribution_from_limits(0, 0.10),
        NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_BEMONC: get_uniform_distribution_from_limits(
            0.302, 0.529
        ),
        NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_CEMONC: get_uniform_distribution_from_limits(
            0.768, 0.972
        ),
    },
    "Nigeria": {
        NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_HOME: get_uniform_distribution_from_limits(0, 0.10),
        NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_BEMONC: get_uniform_distribution_from_limits(
            0.302, 0.529
        ),
        NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_CEMONC: get_uniform_distribution_from_limits(
            0.768, 0.972
        ),
    },
    "Pakistan": {
        NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_HOME: get_uniform_distribution_from_limits(0, 0.10),
        NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_BEMONC: get_uniform_distribution_from_limits(
            0.302, 0.529
        ),
        NO_ANTIBIOTICS_RISK.P_ANTIBIOTIC_CEMONC: get_uniform_distribution_from_limits(
            0.768, 0.972
        ),
    },
}
ANTIBIOTIC_RELATIVE_RISK_DISTRIBUTION = get_lognorm_from_quantiles(0.78, 0.60, 1.00)


PROBIOTICS_BASELINE_COVERAGE_PROABILITY = 0.0
PROBIOTICS_RELATIVE_RISK_DISTRIBUTION = get_lognorm_from_quantiles(0.81, 0.72, 0.91)


class __Interventions(NamedTuple):
    CPAP: str = "cpap"
    ANTIBIOTICS: str = "antibiotics"
    PROBIOTICS: str = "probiotics"
    AZITHROMYCIN: str = "azithromycin"
    MISOPROSTOL: str = "misoprostol"


INTERVENTIONS = __Interventions()
INTERVENTION_TYPE_MAPPER = {
    INTERVENTIONS.CPAP: "neonatal",
    INTERVENTIONS.ANTIBIOTICS: "neonatal",
    INTERVENTIONS.PROBIOTICS: "neonatal",
    INTERVENTIONS.AZITHROMYCIN: "maternal",
    INTERVENTIONS.MISOPROSTOL: "maternal",
}


AZITHROMYCIN_FACILITY_TYPE_DISTRIBUTION = {
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
AZITHROMYCIN_RELATIVE_RISK_DISTRIBUTION = get_lognorm_from_quantiles(1.54, 1.30, 1.82)
MISOPROSTOL_RELATIVE_RISK_DISTRIBUTION = get_lognorm_from_quantiles(0.61, 0.50, 0.74)


# Postpartum depression constants
POSTPARTUM_DEPRESSION_INCIDENCE_RISK = get_truncnorm(
    0.12, ninety_five_pct_confidence_interval=(0.04, 0.20)
)
# Case duration is in years
POSTPARTUM_DEPRESSION_CASE_DURATION = get_truncnorm(
    0.65, ninety_five_pct_confidence_interval=(0.59, 0.70)
)


class __PostpartumDepressionCaseTypes(NamedTuple):
    NONE: str = "none"
    ASYMPTOMATIC: str = "asymptomatic"
    MILD: str = "mild"
    MODERATE: str = "moderate"
    SEVERE: str = "severe"


POSTPARTUM_DEPRESSION_CASE_TYPES = __PostpartumDepressionCaseTypes()


POSTPARTUM_DEPRESSION_CASE_SEVERITY_PROBABILITIES = {
    POSTPARTUM_DEPRESSION_CASE_TYPES.ASYMPTOMATIC: 0.14,
    POSTPARTUM_DEPRESSION_CASE_TYPES.MILD: 0.59,
    POSTPARTUM_DEPRESSION_CASE_TYPES.MODERATE: 0.17,
    POSTPARTUM_DEPRESSION_CASE_TYPES.SEVERE: 0.10,
}
POSTPARTUM_DEPRESSION_CASE_SEVERITY_DISABILITY_WEIGHTS = {
    POSTPARTUM_DEPRESSION_CASE_TYPES.ASYMPTOMATIC: get_truncnorm(0.0, 0.00**2),
    POSTPARTUM_DEPRESSION_CASE_TYPES.MILD: get_truncnorm(
        0.145, ninety_five_pct_confidence_interval=(0.099, 0.209)
    ),
    POSTPARTUM_DEPRESSION_CASE_TYPES.MODERATE: get_truncnorm(
        0.396, ninety_five_pct_confidence_interval=(0.267, 0.531)
    ),
    POSTPARTUM_DEPRESSION_CASE_TYPES.SEVERE: get_truncnorm(
        0.658, ninety_five_pct_confidence_interval=(0.477, 0.807)
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
