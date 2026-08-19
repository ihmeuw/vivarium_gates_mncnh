from pathlib import Path

import vivarium_gates_mncnh
from vivarium_gates_mncnh.constants import metadata

BASE_DIR = Path(vivarium_gates_mncnh.__file__).resolve().parent

ARTIFACT_ROOT = Path(
    f"/mnt/team/simulation_science/pub/models/{metadata.PROJECT_NAME}/artifacts/"
)
REPO_DATA_DIR = Path(__file__).parent.parent / "data"
CLUSTER_DATA_DIR = Path("/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/data")
ANC_DATA_DIR = CLUSTER_DATA_DIR / "antenatal_care"
HEMOGLOBIN_SCREENING_DATA_DIR = CLUSTER_DATA_DIR / "hemoglobin_screening"

LBWSG_PAF_OUTPUTS_DIR = REPO_DATA_DIR / "lbwsg_paf" / "outputs"
CALCULATED_PAFS_DIR = LBWSG_PAF_OUTPUTS_DIR / "paf_outputs"
PRETERM_PREVALENCE_DIR = LBWSG_PAF_OUTPUTS_DIR / "preterm_prevalence_outputs"

LBWSG_RR_CAPS_DIR = REPO_DATA_DIR / "lbwsg_rr_caps" / "caps"
STILLBIRTH_RATIO_DATA_DIR = REPO_DATA_DIR / "stillbirth_ratio"
ORAL_IRON_DATA_DIR = REPO_DATA_DIR / "ifa_mms_gestation_shifts"
HEMOGLOBIN_EFFECTS_DATA_DIR = REPO_DATA_DIR / "hemoglobin_effects"
HEMOGLOBIN_PAF_OUTPUTS_DIR = REPO_DATA_DIR / "hgb_pafs_maternal_disorders" / "outputs"
MODEL_SPEC_PATH = BASE_DIR / "model_specifications" / "model_spec.yaml"
FERRITIN_TESTING_COVERAGE_DATA_DIR = REPO_DATA_DIR / "ferritin_testing_coverage"
FACILITY_CHOICE_OPTIMIZATION_RESULTS_CSV = (
    REPO_DATA_DIR / "facility_choice" / "facility_choice_optimization_results.csv"
)
PPH_CROSSWALK_PARAMETERS_CSV = (
    REPO_DATA_DIR / "postpartum_hemorrhage_split" / "crosswalk_parameters.csv"
)
HEMORRHAGE_HEMOGLOBIN_SHIFT_DIR = REPO_DATA_DIR / "hemorrhage_hemoglobin_shift"
# MR-BRT causal attribution model from GBD anemia team: /mnt/team/anemia/pub/emotive/mat_hem/
HEMORRHAGE_HEMOGLOBIN_SHIFT_PRED_DATA_CSV = HEMORRHAGE_HEMOGLOBIN_SHIFT_DIR / "pred_data.csv"

# Update for new model results directory after model changes and runs
# This should match the directory after /mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/results/
MODEL_RESULTS_DIR = "model39.0"
MODEL_NOTEBOOKS_DIR = BASE_DIR.parent.parent / "tests" / "model_notebooks"
