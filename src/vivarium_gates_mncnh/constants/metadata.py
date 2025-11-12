from typing import NamedTuple

import pandas as pd

####################
# Project metadata #
####################

PROJECT_NAME = "vivarium_gates_mncnh"
CLUSTER_PROJECT = "proj_simscience_prod"

CLUSTER_QUEUE = "all.q"
MAKE_ARTIFACT_MEM = 10  # GB
MAKE_ARTIFACT_CPU = 1
MAKE_ARTIFACT_RUNTIME = "3:00:00"
MAKE_ARTIFACT_SLEEP = 10

LOCATIONS = [
    "Ethiopia",
    "Nigeria",
    "Pakistan",
]

ARTIFACT_INDEX_COLUMNS = [
    "sex",
    "age_start",
    "age_end",
    "year_start",
    "year_end",
]
CHILDREN_INDEX_COLUMNS = [
    "sex_of_child",
    "child_age_start",
    "child_age_end",
    "year_start",
    "year_end",
]


DRAW_COUNT = 250
ARTIFACT_COLUMNS = pd.Index([f"draw_{i}" for i in range(DRAW_COUNT)])
ARTIFACT_YEAR_START = 2023
ARTIFACT_YEAR_END = ARTIFACT_YEAR_START + 1


class __Scenarios(NamedTuple):
    baseline: str = "baseline"
    # TODO - add scenarios here


SCENARIOS = __Scenarios()


PRETERM_AGE_CUTOFF = 37.0  # weeks
GBD_BIRTH_AGE_GROUP_ID = 164
