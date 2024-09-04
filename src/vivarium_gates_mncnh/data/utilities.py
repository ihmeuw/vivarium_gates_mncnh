from typing import Union

from gbd_mapping import ModelableEntity, causes, covariates, risk_factors
from vivarium.framework.artifact import EntityKey
from vivarium_inputs.mapping_extension import alternative_risk_factors


def get_entity(key: Union[str, EntityKey]):
    # Map of entity types to their gbd mappings.
    type_map = {
        "cause": causes,
        "covariate": covariates,
        "risk_factor": risk_factors,
        "alternative_risk_factor": alternative_risk_factors,
    }
    key = EntityKey(key)
    return type_map[key.type][key.name]