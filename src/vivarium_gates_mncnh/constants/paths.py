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
ANC_DATA_DIR = CLUSTER_DATA_DIR / "antenatal_care"

ORAL_IRON_DATA_DIR = CLUSTER_DATA_DIR / "ifa_mms_gestation_shifts"
IFA_GA_SHIFT_DATA_DIR = ORAL_IRON_DATA_DIR / "ifa_gestational_age_shifts"
MMS_GA_SHIFT_1_DATA_DIR = ORAL_IRON_DATA_DIR / "mms_gestational_age_shifts/shift1"
MMS_GA_SHIFT_2_DATA_DIR = ORAL_IRON_DATA_DIR / "mms_gestational_age_shifts/shift2"
