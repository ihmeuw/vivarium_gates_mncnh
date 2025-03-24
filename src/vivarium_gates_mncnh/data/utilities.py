from typing import Union

import pandas as pd
from gbd_mapping import ModelableEntity, causes, covariates, risk_factors
from scipy import stats
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
        ],
        closed="left",
    )
    return endpoints


def get_norm_from_quantiles(
    mean: float, lower: float, upper: float, quantiles: Tuple[float, float] = (0.025, 0.975)
) -> stats.norm:
    stdnorm_quantiles = stats.norm.ppf(quantiles)
    sd = (upper - lower) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
    return stats.norm(loc=mean, scale=sd)


def get_norm(
    mean: float,
    sd: float = None,
    ninety_five_pct_confidence_interval: tuple[float, float] = None,
) -> stats.norm:
    sd = _get_standard_deviation(mean, sd, ninety_five_pct_confidence_interval)
    return stats.norm(loc=mean, scale=sd)
