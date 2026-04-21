from pathlib import Path
from typing import NamedTuple

import pandas as pd
import yaml

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

# Load the full set of 20 scenario draws from the LBWSG PAF branches YAML,
# which is the canonical source for all draws used in data generation.
_SCENARIOS_YAML = (
    Path(__file__).parent.parent / "data" / "lbwsg_paf" / "code" / "lbwsg_paf_branches.yaml"
)
with open(_SCENARIOS_YAML) as _f:
    SCENARIO_DRAWS = yaml.safe_load(_f)["input_draws"]


class __Scenarios(NamedTuple):
    baseline: str = "baseline"
    # TODO - add scenarios here


SCENARIOS = __Scenarios()


PRETERM_AGE_CUTOFF = 37.0  # weeks
GBD_BIRTH_AGE_GROUP_ID = 164
