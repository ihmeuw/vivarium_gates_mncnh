from datetime import datetime
from typing import NamedTuple

from vivarium_gates_mncnh.constants.data_keys import (
    FACILITY_CHOICE,
    NO_ANTIBIOTICS_RISK,
    NO_CPAP_RISK,
)
from vivarium_gates_mncnh.utilities import (
    get_norm,
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
    CPAP_ACCESS = "cpap_access"
    ANTIBIOTICS_ACCESS = "antibiotics_access"
    MATERNAL_SEPSIS = "maternal_sepsis_and_other_maternal_infections"
    MATERNAL_HEMORRHAGE = "maternal_hemorrhage"
    OBSTRUCTED_LABOR = "maternal_obstructed_labor_and_uterine_rupture"
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
    BIRTH_WEIGHT = "birth_weight"
    GESTATIONAL_AGE = "gestational_age"
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
    LBWSG_ACMR_PAF_MODIFIER = "lbwsg_paf_on_all_causes.cause_specific_mortality_rate.paf"
    ACMR = "all_causes.cause_specific_mortality_rate"
    ACMR_PAF = "all_causes.cause_specific_mortality_rate.paf"
    DEATH_IN_AGE_GROUP_PROBABILITY = "death_in_age_group_probability"
    NEONATAL_PRETERM_BIRTH_WITH_RDS = (
        "neonatal_preterm_birth_with_rds.cause_specific_mortality_rate"
    )
    PRETERM_WITH_RDS_FINAL_CSMR = "neonatal_preterm_birth_with_rds.csmr"
    NEONATAL_PRETERM_BIRTH_WITHOUT_RDS = (
        "neonatal_preterm_birth_without_rds.cause_specific_mortality_rate"
    )
    NEONATAL_SEPSIS = (
        "neonatal_sepsis_and_other_neonatal_infections.cause_specific_mortality_rate"
    )
    NEONATAL_ENCEPHALOPATHY = "neonatal_encephalopathy_due_to_birth_asphyxia_and_trauma.cause_specific_mortality_rate"
    PRETERM_WITHOUT_RDS_FINAL_CSMR = "neonatal_preterm_birth_without_rds.csmr"
    NEONATAL_SEPSIS_FINAL_CSMR = "neonatal_sepsis_and_other_neonatal_infections.csmr"
    NEONATAL_ENCEPHALOPATHY_FINAL_CSMR = (
        "neonatal_encephalopathy_due_to_birth_asphyxia_and_trauma.csmr"
    )
    PRETERM_WITH_RDS_RR = (
        "effect_of_risk_factor.low_birth_weight_and_short_gestation_on_neonatal_preterm_birth_with_rds.relative_risk",
    )
    PRETERM_WITHOUT_RDS_RR = (
        "effect_of_risk_factor.low_birth_weight_and_short_gestation_on_neonatal_preterm_birth_without_rds.relative_risk",
    )
    NEONATAL_SEPSIS_RR = (
        "effect_of_risk_factor.low_birth_weight_and_short_gestation_on_neonatal_sepsis_and_other_neonatal_infections.relative_risk",
    )
    NEONATAL_ENCEPHALOPATHY_RR = (
        "effect_of_risk_factor.low_birth_weight_and_short_gestation_on_neonatal_encephalopathy_due_to_birth_asphyxia_and_trauma.relative_risk",
    )
    ACMR_RR = (
        "effect_of_risk_factor.low_birth_weight_and_short_gestation_on_all_causes.cause_specific_mortality_rate",
    )
    BIRTH_WEIGHT_EXPOSURE = "birth_weight.birth_exposure"
    GESTATIONAL_AGE_EXPOSURE = "gestational_age.birth_exposure"
    PREGNANCY_DURATION = "pregnancy_duration"
    BIRTH_OUTCOME_PROBABILITIES = "birth_outcome_probabilities"


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
        NO_CPAP_RISK.P_CPAP_BEMONC: 0.075,
        NO_CPAP_RISK.P_CPAP_CEMONC: 0.393,
        NO_CPAP_RISK.P_CPAP_HOME: 0.0,
    },
    "Nigeria": {
        NO_CPAP_RISK.P_CPAP_BEMONC: 0.075,
        NO_CPAP_RISK.P_CPAP_CEMONC: 0.393,
        NO_CPAP_RISK.P_CPAP_HOME: 0.0,
    },
    "Pakistan": {
        NO_CPAP_RISK.P_CPAP_BEMONC: 0.075,
        NO_CPAP_RISK.P_CPAP_CEMONC: 0.393,
        NO_CPAP_RISK.P_CPAP_HOME: 0.0,
    },
}


ANTIBIOTIC_FACILITY_TYPE_DISTRIBUTION = {
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
ANTIBIOTIC_RELATIVE_RISK_DISTRIBUTION = get_norm(1.39, 0.08**2)
