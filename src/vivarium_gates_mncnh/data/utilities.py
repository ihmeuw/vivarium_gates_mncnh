from typing import Union

import pandas as pd
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
        ]
    )
    return endpoints


def rescale_prevalence(exposure):
    """Rescales prevalences to add to 1 in LBWSG exposure data pulled from GBD 2019 by get_draws."""
    # Drop residual 'cat125' parameter with meid==NaN, and convert meid col from float to int
    exposure = exposure.dropna().astype({"modelable_entity_id": int})
    # Define some categories of columns
    draw_cols = exposure.filter(regex=r"^draw_\d{1,3}$").columns.to_list()
    category_cols = ["modelable_entity_id", "parameter"]
    index_cols = exposure.columns.difference(draw_cols)
    sum_index = index_cols.difference(category_cols)
    # Add prevalences over categories (indexed by meid and/or parameter) to get denominator for rescaling
    prevalence_sum = exposure.groupby(sum_index.to_list())[draw_cols].sum()
    # Divide prevalences by total to rescale them to add to 1, and reset index to put df back in original form
    exposure = exposure.set_index(index_cols.to_list()) / prevalence_sum
    exposure.reset_index(inplace=True)
    return exposure
