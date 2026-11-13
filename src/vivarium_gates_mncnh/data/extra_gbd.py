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

# "GBD 2023 Additional Publications and SDGs 2025". vivarium_gbd_access has no name for it.
_HEMOGLOBIN_RELEASE_ID = 33
_HEMOGLOBIN_REI_ID = 376
# Burdenator run v393 is addressed by this compare version.
_HEMOGLOBIN_PAF_COMPARE_VERSION_ID = 8303
_HEMOGLOBIN_EXPOSURE_ME_ID = 28869
# The release-33 SD of ME 28870 was never published, so use this one instead.
_HEMOGLOBIN_EXPOSURE_SD_ME_ID = 10488
_NON_PREGNANT_HEMOGLOBIN_EXPOSURE_ME_ID = 27596
_NUM_HEMOGLOBIN_DRAWS = 100


def _get_hemoglobin_modelable_entity_draws(
    me_id: int, location: str, release_id: int
) -> pd.DataFrame:
    """Pull female draws for a hemoglobin modelable entity.

    Keeps only the first 100 draws, since the loaders expand from 100 to 250.
    """
    location_id = utility_data.get_location_id(location)
    data = base_data.get_modelable_entity_draws(
        me_id, location_id, ARTIFACT_YEAR_START, "draws", release_id=release_id
    )
    data = data[data["sex_id"].isin(gbd_constants.SEX.FEMALE)]
    extra_draws = [
        c
        for c in data.columns
        if c.startswith("draw_") and int(c.removeprefix("draw_")) >= _NUM_HEMOGLOBIN_DRAWS
    ]
    return data.drop(columns=extra_draws).reset_index(drop=True)


def _release_33_data(name: str) -> pd.DataFrame:
    """Read a stored release-33 hemoglobin dataset."""
    path = HEMOGLOBIN_RELEASE_33_DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. See the README in "
            f"{HEMOGLOBIN_RELEASE_33_DATA_DIR} and MIC-6303."
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
def get_mortality_death_counts(
    location: str, age_group_id: int, gbd_id: int | list[int]
) -> pd.DataFrame:
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
    """Get hemoglobin exposure mean (ME 28869) or exposure SD (ME 10488)."""
    me_id = (
        _HEMOGLOBIN_EXPOSURE_ME_ID
        if key == data_keys.HEMOGLOBIN.EXPOSURE
        else _HEMOGLOBIN_EXPOSURE_SD_ME_ID
    )
    return _get_hemoglobin_modelable_entity_draws(me_id, location, _HEMOGLOBIN_RELEASE_ID)


@vi_utils.cache
def get_hemoglobin_rr_data(key: str, location: str) -> pd.DataFrame:
    """Get hemoglobin relative risks, from file.

    Release 33's model version was never published into folio storage, so this
    cannot be pulled.
    """
    data = _release_33_data("hemoglobin_relative_risk")
    data["year_id"] = 2023
    return data


@vi_utils.cache
def get_hemoglobin_paf_data(key: str, location: str) -> pd.DataFrame:
    """Get release-33 Burdenator PAFs for hemoglobin."""
    location_id = utility_data.get_location_id(location)
    data = base_data.get_machinery_estimates(
        entity="risk-cause",
        entity_id=[_HEMOGLOBIN_REI_ID, None],
        compare_version_id=_HEMOGLOBIN_PAF_COMPARE_VERSION_ID,
        estimates="draws",
        measure_id=vi_globals.MEASURES["Deaths"],
        metric_id=vi_globals.METRICS["Percent"],
        location_id=location_id,
        sex_id=gbd_constants.SEX.FEMALE,
        age_group_id=get_age_group_id(),
        year_id=ARTIFACT_YEAR_START,
    )
    return data


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
    """Get non-pregnant hemoglobin exposure (MEID 27596).

    Release 16, not 33, since there is no release-33 version of this modelable entity.
    """
    return _get_hemoglobin_modelable_entity_draws(
        _NON_PREGNANT_HEMOGLOBIN_EXPOSURE_ME_ID,
        location,
        gbd_constants.RELEASE_IDS.GBD_2023,
    )
