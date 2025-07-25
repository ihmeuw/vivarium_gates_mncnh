import pandas as pd
from vivarium_gbd_access import constants as gbd_constants
from vivarium_gbd_access import utilities as vi_utils
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import utility_data

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.constants.metadata import GBD_BIRTH_AGE_GROUP_ID
from vivarium_gates_mncnh.data import utilities


@vi_utils.cache
def get_maternal_disorder_yld_rate(key: str, location: str) -> pd.DataFrame:
    entity = utilities.get_entity(key)
    location_id = utility_data.get_location_id(location)
    data = vi_utils.get_draws(
        "cause_id",
        entity.gbd_id,
        source=gbd_constants.SOURCES.COMO,
        location_id=location_id,
        year_id=2021,
        release_id=gbd_constants.RELEASE_IDS.GBD_2021,
        measure_id=vi_globals.MEASURES["YLDs"],
        metric_id=vi_globals.METRICS["Rate"],
    )
    return data


@vi_utils.cache
def load_lbwsg_exposure(location: str) -> pd.DataFrame:
    entity = utilities.get_entity(data_keys.LBWSG.BIRTH_EXPOSURE)
    location_id = utility_data.get_location_id(location)
    data = vi_utils.get_draws(
        gbd_id_type="rei_id",
        gbd_id=entity.gbd_id,
        source=gbd_constants.SOURCES.EXPOSURE,
        location_id=location_id,
        year_id=2021,
        sex_id=gbd_constants.SEX.MALE + gbd_constants.SEX.FEMALE,
        age_group_id=164,  # Birth prevalence
        release_id=gbd_constants.RELEASE_IDS.GBD_2021,
    )
    return data


@vi_utils.cache
def get_birth_counts(location: str) -> pd.DataFrame:
    from db_queries import get_population

    location_id = utility_data.get_location_id(location)
    births = get_population(
        release_id=gbd_constants.RELEASE_IDS.GBD_2021,
        location_id=location_id,
        age_group_id=GBD_BIRTH_AGE_GROUP_ID,
        year_id=2021,
        sex_id=[1, 2],
    )
    births = births.drop(["run_id", "age_group_id"], axis=1).set_index(
        ["location_id", "sex_id", "year_id"]
    )

    return births


@vi_utils.cache
def get_mortality_death_counts(location: str, age_group_id: int, gbd_id: int) -> pd.DataFrame:
    location_id = utility_data.get_location_id(location)
    data = vi_utils.get_draws(
        release_id=gbd_constants.RELEASE_IDS.GBD_2021,
        location_id=location_id,
        age_group_id=age_group_id,
        gbd_id_type="cause_id",
        gbd_id=gbd_id,
        measure_id=vi_globals.MEASURES["Deaths"],
        source=gbd_constants.SOURCES.CODCORRECT,
        year_id=2021,
    )
    return data


@vi_utils.cache
def get_hemoglobin_exposure_data(key: str, location: str) -> pd.DataFrame:
    """
    Get hemoglobin exposure data for a given location and source.
    """
    source_map = {
        data_keys.HEMOGLOBIN.EXPOSURE: gbd_constants.SOURCES.EXPOSURE,
        data_keys.HEMOGLOBIN.STANDARD_DEVIATION: gbd_constants.SOURCES.EXPOSURE_SD,
    }
    source = source_map[key]
    location_id = utility_data.get_location_id(location)
    data = vi_utils.get_draws(
        gbd_id_type="rei_id",
        gbd_id=376,
        source=source,
        location_id=location_id,
        year_id=2023,
        sex_id=gbd_constants.SEX.FEMALE,
        release_id=16,
    )
    return data


@vi_utils.cache
def get_hemoglobin_rr_data(key: str, location: str) -> pd.DataFrame:
    data = vi_utils.get_draws(
        release_id=33,
        gbd_id_type="rei_id",
        gbd_id=376,
        source=gbd_constants.SOURCES.RR,
        sex_id=gbd_constants.SEX.FEMALE,
        location_id=1,
        year_id=2022,
    )
    return data


@vi_utils.cache
def get_hemoglobin_paf_data(key: str, location: str) -> pd.DataFrame:
    data = vi_utils.get_draws(
        release_id=33,
        version_id=393,
        gbd_id_type="rei_id",
        gbd_id=376,
        source=gbd_constants.SOURCES.BURDENATOR,
        sex_id=gbd_constants.SEX.FEMALE,
        location_id=1,
        year_id=2023,
        metric_id=vi_globals.METRICS["Percent"],
        measure_id=vi_globals.MEASURES["Deaths"],
    )
    return data
