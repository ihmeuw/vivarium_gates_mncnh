import pandas as pd
from loguru import logger
from vivarium_gbd_access import constants as gbd_constants
from vivarium_gbd_access import utilities as vi_utils
from vivarium_gbd_access.gbd import base_data
from vivarium_gbd_access.gbd.demographics import get_age_group_id
from vivarium_gbd_access.gbd.measures import get_birth_exposure
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import utility_data

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.metadata import (
    ARTIFACT_YEAR_START,
    GBD_BIRTH_AGE_GROUP_ID,
)
from vivarium_gates_mncnh.constants.paths import HEMOGLOBIN_RELEASE_33_DATA_DIR
from vivarium_gates_mncnh.data import utilities

_ALL_SEXES = gbd_constants.SEX.MALE + gbd_constants.SEX.FEMALE

HEMOGLOBIN_PUBLICATION_RELEASE_ID = 33

HEMOGLOBIN_PAF_COMPARE_VERSION_ID = 8303


def _release_33_data(name: str) -> pd.DataFrame:
    """Read a stored release-33 hemoglobin dataset.

    There is no live fallback, so a missing file is a hard error. See the README
    beside the data for provenance and repull calls.
    """
    path = HEMOGLOBIN_RELEASE_33_DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. See the README in "
            "HEMOGLOBIN_RELEASE_33_DATA_DIR and MIC-6303."
        )
    logger.warning(
        f"Reading {name} from {path}. This is a stored capture, not a live GBD pull."
    )
    return pd.read_parquet(path)


@vi_utils.cache
def get_maternal_disorder_yld_rate(key: str, location: str) -> pd.DataFrame:
    entity = utilities.get_entity(key)
    location_id = utility_data.get_location_id(location)
    data = base_data.get_machinery_estimates(
        entity="cause",
        entity_id=entity.gbd_id,
        release_id=gbd_constants.RELEASE_IDS.GBD_2023,
        estimates="draws",
        measure_id=vi_globals.MEASURES["YLDs"],
        metric_id=vi_globals.METRICS["Rate"],
        location_id=location_id,
        sex_id=_ALL_SEXES,
        age_group_id=get_age_group_id(),
        year_id=ARTIFACT_YEAR_START,
    )
    return data


@vi_utils.cache
def load_2021_lbwsg_birth_exposure(location: str) -> pd.DataFrame:
    """Pull the GBD 2021 LBWSG birth exposure, relabelled to the artifact year."""
    entity = utilities.get_entity(data_keys.LBWSG.BIRTH_EXPOSURE)
    location_id = utility_data.get_location_id(location)
    data = get_birth_exposure(
        entity.gbd_id,
        location_id,
        2022,
        "draws",
        release_id=gbd_constants.RELEASE_IDS.GBD_2021,
    )
    data["year_id"] = ARTIFACT_YEAR_START
    return data


@vi_utils.cache
def get_birth_counts(location: str) -> pd.DataFrame:
    from db_queries import get_population

    location_id = utility_data.get_location_id(location)
    births = get_population(
        release_id=gbd_constants.RELEASE_IDS.GBD_2023,
        location_id=location_id,
        age_group_id=GBD_BIRTH_AGE_GROUP_ID,
        year_id=ARTIFACT_YEAR_START,
        sex_id=[1, 2],
    )
    births = births.drop(["run_id", "age_group_id"], axis=1).set_index(
        ["location_id", "sex_id", "year_id"]
    )

    return births


@vi_utils.cache
def get_mortality_death_counts(location: str, age_group_id: int, gbd_id: int) -> pd.DataFrame:
    location_id = utility_data.get_location_id(location)
    data = base_data.get_machinery_estimates(
        entity="cause",
        entity_id=gbd_id,
        release_id=gbd_constants.RELEASE_IDS.GBD_2023,
        estimates="draws",
        measure_id=vi_globals.MEASURES["Deaths"],
        metric_id=vi_globals.METRICS["Number"],
        location_id=location_id,
        sex_id=_ALL_SEXES,
        age_group_id=age_group_id,
        year_id=ARTIFACT_YEAR_START,
    )
    return data


@vi_utils.cache
def get_hemoglobin_exposure_data(key: str, location: str) -> pd.DataFrame:
    """Get hemoglobin exposure mean or exposure standard deviation, both from file.

    The SD cannot be pulled -- its release-33 model version was never published and
    GBD have said it will not be. The mean is reachable but stored alongside it so
    the pair stays consistent. Both hold what ``get_draws`` returned, so they already
    carry ``parameter`` and must not have the exposure category re-attached.
    """
    dataset = (
        "hemoglobin_exposure"
        if key == data_keys.HEMOGLOBIN.EXPOSURE
        else "hemoglobin_exposure_sd"
    )
    return _release_33_data(f"{dataset}_{location.lower()}")


@vi_utils.cache
def get_hemoglobin_rr_data(key: str, location: str) -> pd.DataFrame:
    """Get hemoglobin relative risks, from file.

    Release 33's model version was never published into folio storage, so this
    cannot be pulled at any release. Release 16 is not a substitute: it moves the
    downstream maternal-disorder PAFs by ~0.1 on average and pushes some negative.
    """
    data = _release_33_data("hemoglobin_relative_risk")
    data["year_id"] = 2023
    return data


@vi_utils.cache
def get_hemoglobin_paf_data(key: str, location: str) -> pd.DataFrame:
    """Get Burdenator PAFs for hemoglobin, from file.

    Reachable live via ``HEMOGLOBIN_PAF_COMPARE_VERSION_ID``, and the stored copy is
    bit-identical to that pull, but held with the rest of the release-33 set because
    the run-to-compare-version mapping survives only as free text.
    """
    return _release_33_data(f"hemoglobin_paf_{location.lower()}")


@vi_utils.cache
def get_sequela_data(sequela_id: int, location: str, measure: str) -> pd.DataFrame:
    """Get sequela-level data from COMO for the given measure."""
    location_id = utility_data.get_location_id(location)
    data = base_data.get_machinery_estimates(
        entity="sequela",
        entity_id=sequela_id,
        release_id=gbd_constants.RELEASE_IDS.GBD_2023,
        estimates="draws",
        measure_id=vi_globals.MEASURES[measure],
        metric_id=vi_globals.METRICS["Rate"],
        location_id=location_id,
        sex_id=_ALL_SEXES,
        age_group_id=get_age_group_id(),
        year_id=ARTIFACT_YEAR_START,
    )
    return data


@vi_utils.cache
def get_non_pregnant_hemoglobin_exposure_data(location: str) -> pd.DataFrame:
    """Get non-pregnant hemoglobin exposure (MEID 27596), from file.

    Release 16 rather than 33, but stored with the release-33 data because it
    supplies the mean paired with the release-33 exposure SD; pulling one side live
    would draw the distribution's two parameters from different captures.
    """
    return _release_33_data(f"non_pregnant_hemoglobin_exposure_{location.lower()}")
