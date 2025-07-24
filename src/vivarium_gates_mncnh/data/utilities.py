from typing import Union

import pandas as pd
from gbd_mapping import causes, covariates, risk_factors
from vivarium.framework.artifact import EntityKey
from vivarium.framework.randomness import get_hash
from vivarium_inputs.mapping_extension import alternative_risk_factors

from vivarium_gates_mncnh.constants.data_keys import REMAP_KEY_GROUPS
from vivarium_gates_mncnh.constants.metadata import (
    ARTIFACT_COLUMNS,
    ARTIFACT_INDEX_COLUMNS,
    CHILDREN_INDEX_COLUMNS,
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
    # Some data may already be mapped
    age_start_col = "child_age_start" if "child_age_start" in data.columns else "age_start"
    index_cols = (
        CHILDREN_INDEX_COLUMNS
        if age_start_col == "child_age_start"
        else ARTIFACT_INDEX_COLUMNS
    )

    data.loc[data[age_start_col] > 7 / 365.0, ARTIFACT_COLUMNS] = value
    return data.set_index(index_cols)


def rename_child_data_index_names(data: pd.DataFrame) -> pd.DataFrame:

    # Renames index names in artifact data for child age and sex
    for column in ["sex", "age_start", "age_end"]:
        if column not in data.index.names:
            continue
        if column == "sex":
            data.index.rename({column: "sex_of_child"}, inplace=True)
        else:
            data.index.rename({column: f"child_{column}"}, inplace=True)
    return data


def determine_if_remap_group(key: str) -> bool:
    """Determine whether the artifact key is in a key group that needs to be remapped
    for the children demographic columns."""

    to_remap = any(group.name in key for group in REMAP_KEY_GROUPS)
    return to_remap


def expand_draw_columns(data: pd.DataFrame, num_draws: int, num_repeats: int) -> pd.DataFrame:
    """
    Expands draw columns in the input DataFrame by repeating them num_repeats times.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing draw columns.
        num_draws (int): Number of draw columns to expand (e.g., 100).
        num_repeats (int): Number of times to repeat the draw columns (e.g., 5).

    Returns:
        pd.DataFrame: DataFrame with expanded draw columns.
    """
    draw_cols = [f"draw_{i}" for i in range(num_draws)]
    expanded_draws = []

    for i in range(num_repeats):
        df_copy = data[draw_cols].copy()
        df_copy.columns = [f"draw_{j}" for j in range(i * num_draws, (i + 1) * num_draws)]
        expanded_draws.append(df_copy)
    expanded_draws_df = pd.concat(expanded_draws, axis=1)

    return expanded_draws_df