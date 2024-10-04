from datetime import datetime
from typing import NamedTuple

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


DURATIONS = _Durations()


INFANT_MALE_PERCENTAGES = {
    "Ethiopia": 0.514271,
    "Nigeria": 0.511785,
    "Pakistan": 0.514583,
}


NUM_DRAWS = 500


class _SimulationEventNames(NamedTuple):
    PREGNANCY = "pregnancy"
    INTRAPARTRUM = "intrapartum"
    NEONATAL = "neonatal"


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
    PREGNANCY_OUTCOME = "pregnancy_outcome"
    PREGNANCY_DURATION = "pregnancy_duration"
    SEX_OF_CHILD = "sex_of_child"
    BIRTH_WEIGHT = "birth_weight"
    GESTATIONAL_AGE = "gestational_age"
    ATTENDED_CARE_FACILITY = "attended_care_facility"
    RECEIVED_ULTRASOUND = "received_ultrasound"
    ULTRASOUND_TYPE = "ultrasound_type"
    STATED_GESTATIONAL_AGE = "stated_gestational_age"
    SUCCESSFUL_LBW_IDENTIFICATION = "successful_lbw_identification"


COLUMNS = __Columns()
