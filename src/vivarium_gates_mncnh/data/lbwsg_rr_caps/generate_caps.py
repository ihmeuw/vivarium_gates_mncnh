"""
LBWSG RR Caps Generator
-----------------------

This script generates capped relative risk values for low birth weight and short gestation.

Usage:
    python generate_caps.py [-l LOCATION] [-o OUTPUT_DIR]

Arguments:
    -l, --location     The location for which to generate LBWSG RR caps.
                      Example: Ethiopia, Nigeria, Pakistan
                      Default: Ethiopia

    -o, --output-dir   The output directory where the generated CSV file will be saved.
                      Default: /ihme/homes/hjafari/artifacts/test_caps/

Example:
    python generate_caps.py -l Nigeria -o /path/to/output_dir

The output will be a CSV file named <location>.csv in the specified output directory.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from vivarium_gates_mncnh.constants import data_keys, data_values
from vivarium_gates_mncnh.data.loader import get_data, load_standard_data


def prepare_rr_cap_inputs(
    rrs: pd.DataFrame, location: str
) -> dict[tuple[str, str], dict[str, pd.DataFrame]]:
    """
    Returns a dictionary mapping (sex, age_group) to a dict with keys 'exposure', 'rrs', and 'acmrisk',
    each containing a pd.DataFrame.
    """
    data: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for sex in ["Male", "Female"]:
        # Early neonatal
        data[(sex, "early_neonatal")] = {
            "exposure": get_data(data_keys.LBWSG.EXPOSURE, location).query(
                f"child_age_start==0 & sex_of_child=='{sex}'"
            ),
            "rrs": rrs.query(f"age_start == 0 & sex=='{sex}'"),
            "acmrisk": get_data(
                data_keys.POPULATION.ALL_CAUSES_MORTALITY_RISK, location
            ).query(f"age_start == 0 and sex=='{sex}'"),
        }
        # Late neonatal
        data[(sex, "late_neonatal")] = {
            "exposure": get_data(data_keys.LBWSG.EXPOSURE, location).query(
                f"child_age_start==0.01917808 & sex_of_child=='{sex}'"
            ),
            "rrs": rrs.query(f"age_start == 0.01917808 & sex=='{sex}'"),
            "acmrisk": get_data(
                data_keys.POPULATION.ALL_CAUSES_MORTALITY_RISK, location
            ).query(f"age_start==0.01917808 & sex=='{sex}'"),
        }
    return data


def calculate_maximum_mr_by_draw(
    input_data: dict[tuple[str, str], dict[str, pd.DataFrame]],
    rr_cap: float,
    draw: int,
    sex: str,
    age_group: str,
) -> pd.DataFrame:
    exposure = input_data[(sex, age_group)]["exposure"]
    rrs = input_data[(sex, age_group)]["rrs"]
    acmrisk = input_data[(sex, age_group)]["acmrisk"]

    # rename index columns to be consistent with other indices
    acmrisk = acmrisk.reset_index().rename(
        columns={
            "sex": "sex_of_child",
            "age_start": "child_age_start",
            "age_end": "child_age_end",
        }
    )
    acmrisk = acmrisk.set_index(
        ["sex_of_child", "child_age_start", "child_age_end", "year_start", "year_end"]
    )

    rrs_capped = rrs[[f"draw_{draw}"]].reset_index()
    rrs_capped[f"draw_{draw}"] = np.where(
        rrs_capped[f"draw_{draw}"] > rr_cap, rr_cap, rrs_capped[f"draw_{draw}"]
    )
    rrs_capped = rrs_capped.set_index(rrs.index.names)
    mean_rr = (
        (rrs_capped * exposure[[f"draw_{draw}"]])
        .groupby([x for x in rrs_capped.index.names if x != "parameter"])
        .sum()
    )
    paf = (mean_rr - 1) / mean_rr
    mr = acmrisk[[f"draw_{draw}"]] * (1 - paf) * rrs_capped
    mr_max = (
        mr.groupby([x for x in mr.index.names if x != "parameter"]).max().dropna()
    )  # maximum across LBWSG categories
    return mr_max


def find_rr_cap(prepped, draw: int, sex: str, age_group: str, location: str) -> float:
    def objective_function(x):
        rr_cap = x[0] if isinstance(x, (list, np.ndarray)) else x
        val = calculate_maximum_mr_by_draw(prepped, rr_cap, draw, sex, age_group)
        val = val.values[0][0]
        return abs(1 - val)  # minimize distance from 1

    def constraint_lower(x):
        rr_cap = x[0] if isinstance(x, (list, np.ndarray)) else x
        val = calculate_maximum_mr_by_draw(prepped, rr_cap, draw, sex, age_group)
        val = val.values[0][0]
        return val  # must be >= 0

    def constraint_upper(x):
        rr_cap = x[0] if isinstance(x, (list, np.ndarray)) else x
        val = calculate_maximum_mr_by_draw(prepped, rr_cap, draw, sex, age_group)
        val = val.values[0][0]
        return 1 - val  # must be >= 0 (i.e., val <= 1)

    constraints = [
        {"type": "ineq", "fun": constraint_lower},  # val >= 0
        {"type": "ineq", "fun": constraint_upper},  # val <= 1
    ]

    initial_guess = [200]
    result = minimize(
        objective_function, initial_guess, constraints=constraints, method="SLSQP"
    )
    return result.x[0]


def generate_rr_caps(rr: pd.DataFrame, location: str) -> pd.DataFrame:
    input_data = prepare_rr_cap_inputs(rr, location)
    rows = []
    for sex in ["Male", "Female"]:
        for draw in range(data_values.NUM_DRAWS):
            for age_group in ["early_neonatal", "late_neonatal"]:
                rr_cap = find_rr_cap(input_data, draw, sex, age_group, location)
                if age_group == "early_neonatal":
                    age_start, age_end = 0.0, 0.01917808
                else:
                    age_start, age_end = 0.01917808, 0.07671233
                rows.append(
                    {
                        "sex": sex,
                        "age_start": age_start,
                        "age_end": age_end,
                        "year_start": 2021,
                        "year_end": 2022,
                        "draw": draw,
                        "value": rr_cap,
                    }
                )
    caps = pd.DataFrame(rows)

    df = caps.pivot(
        index=["sex", "age_start", "age_end", "year_start", "year_end"],
        columns="draw",
        values="value",
    )
    df.columns = [f"draw_{i}" for i in df.columns]

    # duplicate caps for each parameter so that the df index looks like the rr index
    param_values = rr.index.get_level_values("parameter").unique()
    df = df.reset_index()
    df = pd.concat([df.assign(parameter=value) for value in param_values], ignore_index=True)
    df = df.sort_values(rr.index.names).set_index(rr.index.names)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--location",
        type=str,
        default="Ethiopia",
        help="The location for which to generate LBWSG RR caps (defaults to Ethiopia).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        # default='/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/data/lbwsg_rr_caps',
        default="/ihme/homes/hjafari/artifacts/test_caps/",
        help="The output directory where we will write our LBWSG RR caps data.",
    )
    args = parser.parse_args()
    location = args.location
    output_dir = args.output_dir

    data = load_standard_data(data_keys.LBWSG.RELATIVE_RISK, location, 2021)
    data = data.query("year_start == 2021").droplevel(["affected_entity", "affected_measure"])
    data = data[~data.index.duplicated()]
    rr_caps = generate_rr_caps(data, location)
    rr_caps.to_csv(f"{output_dir}/{location.lower()}.csv")
