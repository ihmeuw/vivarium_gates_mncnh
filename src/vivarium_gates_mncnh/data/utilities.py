from typing import Union

import pandas as pd
from gbd_mapping import causes, covariates, risk_factors
from vivarium.framework.artifact import EntityKey
from vivarium.framework.randomness import get_hash
from vivarium_inputs.mapping_extension import alternative_risk_factors

from vivarium_gates_mncnh.constants.metadata import (
    ARTIFACT_COLUMNS,
    ARTIFACT_INDEX_COLUMNS,
)


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


def get_intervals_from_categories(lbwsg_type: str, categories: dict[str, str]) -> pd.Series:
    if lbwsg_type == "low_birth_weight":
        category_endpoints = pd.Series(
            {
                cat: parse_low_birth_weight_description(description)
                for cat, description in categories.items()
            },
            name=f"{lbwsg_type}.endpoints",
        )
    elif lbwsg_type == "short_gestation":
        category_endpoints = pd.Series(
            {
                cat: parse_short_gestation_description(description)
                for cat, description in categories.items()
            },
            name=f"{lbwsg_type}.endpoints",
        )
    else:
        raise ValueError(
            f"Unrecognized risk type {lbwsg_type}.  Risk type must be low_birth_weight or short_gestation"
        )
    category_endpoints.index.name = "parameter"

    return category_endpoints


def parse_low_birth_weight_description(description: str) -> pd.Interval:
    # descriptions look like this: 'Birth prevalence - [34, 36) wks, [2000, 2500) g'

    endpoints = pd.Interval(
        *[
            float(val)
            for val in description.split(", [")[1].split(")")[0].split("]")[0].split(", ")
        ]
    )
    return endpoints


def parse_short_gestation_description(description: str) -> pd.Interval:
    # descriptions look like this: 'Birth prevalence - [34, 36) wks, [2000, 2500) g'

    endpoints = pd.Interval(
        *[
            float(val)
            for val in description.split("- [")[1].split(")")[0].split("+")[0].split(", ")
        ],
        closed="left",
    )
    return endpoints


def set_non_neonnatal_values(data: pd.DataFrame, value: float) -> pd.DataFrame:
    # Sets values outside neonatal age groups to a constant value to indicate that
    # these age groups are not impacted in the model.
    data = data.reset_index()
    data.loc[data["age_start"] > 7 / 365.0, ARTIFACT_COLUMNS] = value
    return data.set_index(ARTIFACT_INDEX_COLUMNS)


def rename_child_data_index_names(data: pd.DataFrame) -> pd.DataFrame:
    # Renames index names in artifact data for child age and sex
    for column in ["sex", "age_start", "age_end"]:
        if column not in data.index.names:
            continue
        if column == "sex":
            data.index.rename({column: "sex_of_child"}, inplace=True)
        else:
            data.index.rename(index={column: f"child_{column}"}, inplace=True)
    return data
