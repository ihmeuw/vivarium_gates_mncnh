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
