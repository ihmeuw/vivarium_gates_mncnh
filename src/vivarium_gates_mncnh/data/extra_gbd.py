from gbd_mapping import sequelae
from vivarium_gbd_access import constants as gbd_constants
from vivarium_gbd_access import utilities as vi_utils
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import utility_data

from vivarium_gates_mncnh.constants import data_keys
from vivarium_gates_mncnh.data import utilities


@vi_utils.cache
def load_lbwsg_exposure(location: str):
    entity = utilities.get_entity(data_keys.LBWSG.EXPOSURE)
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
