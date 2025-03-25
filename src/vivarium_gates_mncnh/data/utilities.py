from typing import Union

import numpy as np
import pandas as pd
from gbd_mapping import ModelableEntity, causes, covariates, risk_factors
from scipy import stats
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


def get_uniform_distribution_from_limits(lower_limit, upper_limit) -> stats.uniform:
    # stats.uniform is over [loc, loc+scal]
    loc = lower_limit
    scale = upper_limit - lower_limit
    return stats.uniform(loc=loc, scale=scale)


def get_norm_from_quantiles(
    mean: float, lower: float, upper: float, quantiles: tuple[float, float] = (0.025, 0.975)
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


def get_random_variable_draws(columns: pd.Index, seed: str, distribution) -> pd.Series:
    return pd.Series(
        [get_random_variable(x, seed, distribution) for x in range(0, columns.size)],
        index=columns,
    )


def get_random_variable(draw: int, seed: str, distribution) -> pd.Series:
    np.random.seed(get_hash(f"{seed}_draw_{draw}"))
    return distribution.rvs()


def _get_standard_deviation(
    mean: float, sd: float, ninety_five_pct_confidence_interval: tuple[float, float]
) -> float:
    if sd is None and ninety_five_pct_confidence_interval is None:
        raise ValueError(
            "Must provide either a standard deviation or a 95% confidence interval."
        )
    if sd is not None and ninety_five_pct_confidence_interval is not None:
        raise ValueError(
            "Cannot provide both a standard deviation and a 95% confidence interval."
        )
    if ninety_five_pct_confidence_interval is not None:
        lower = ninety_five_pct_confidence_interval[0]
        upper = ninety_five_pct_confidence_interval[1]
        if not (lower <= mean <= upper):
            raise ValueError(
                f"The mean ({mean}) must be between the lower ({lower}) and upper ({upper}) "
                "quantile values."
            )

        stdnorm_quantiles = stats.norm.ppf((0.025, 0.975))
        sd = (upper - lower) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
    return sd


def set_non_neonnatal_values(data: pd.DataFrame, value: float) -> pd.DataFrame:
    # Sets values outside neonatal age groups to a constant value to indicate that
    # these age groups are not impacted in the model.
    data = data.reset_index()
    data.loc[data["age_start"] > 7 / 365.0, ARTIFACT_COLUMNS] = value
    return data.set_index(ARTIFACT_INDEX_COLUMNS)
