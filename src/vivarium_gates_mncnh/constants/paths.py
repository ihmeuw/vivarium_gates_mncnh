from pathlib import Path

import vivarium_gates_mncnh
from vivarium_gates_mncnh.constants import metadata

BASE_DIR = Path(vivarium_gates_mncnh.__file__).resolve().parent

ARTIFACT_ROOT = Path(
    f"/mnt/team/simulation_science/pub/models/{metadata.PROJECT_NAME}/artifacts/"
)
# TODO: update so that we only use files from J drive or the repo itself
J_DIR = Path("/home/j/Project/simulation_science/mnch_grant/MNCNH portfolio")
CLUSTER_DATA_DIR = Path("/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/data")
PAF_DIR = CLUSTER_DATA_DIR / "lbwsg_paf/2023_outputs"
PRETERM_PREVALENCE_DIR = CLUSTER_DATA_DIR / "preterm_prevalence/2023_outputs"
LBWSG_RR_CAPS_DIR = Path(__file__).parent.parent / "data" / "lbwsg_rr_caps" / "caps"
ANC_DATA_DIR = CLUSTER_DATA_DIR / "antenatal_care"

ORAL_IRON_DATA_DIR = Path(__file__).parent.parent / "data" / "ifa_mms_gestation_shifts"

FACILITY_CHOICE_OPTIMIZATION_RESULTS_CSV = (
    Path(__file__).parent.parent
    / "data"
    / "facility_choice"
    / "facility_choice_optimization_results.csv"
)

FERRITIN_TESTING_COVERAGE_DATA_DIR = CLUSTER_DATA_DIR / "ferritin_testing_coverage"

STILLBIRTH_RATIO_24_WKS_CSV = "/snfs1/Project/simulation_science/mnch_grant/MNCNH portfolio/stillbirth_livebirth_ratio_24wks.csv"
