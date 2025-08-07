from pathlib import Path

import vivarium_gates_mncnh
from vivarium_gates_mncnh.constants import metadata

BASE_DIR = Path(vivarium_gates_mncnh.__file__).resolve().parent

ARTIFACT_ROOT = Path(
    f"/mnt/team/simulation_science/pub/models/{metadata.PROJECT_NAME}/artifacts/"
)

CLUSTER_DATA_DIR = Path("/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/data")
PAF_DIR = CLUSTER_DATA_DIR / "lbwsg_paf/outputs"
PRETERM_PREVALENCE_DIR = CLUSTER_DATA_DIR / "preterm_prevalence"
LBWSG_RR_CAPS_DIR = Path(__file__).parent.parent / "data" / "lbwsg_rr_caps" / "caps"
