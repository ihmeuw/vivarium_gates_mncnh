"""Builders for the compare-tool tests: results directories and artifacts.

Every builder writes only under the directory it is given, so tests stay
xdist-safe when handed a ``tmp_path``.
"""

import itertools
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from vivarium.artifact import Artifact

##############
# Results    #
##############

RESULTS_ID_COLUMNS = [
    "measure",
    "entity_type",
    "entity",
    "sub_entity",
    "sex",
    "age_group",
    "input_draw",
    "random_seed",
    "scenario",
]


def results_frame(measure: str, seed: int = 0) -> pd.DataFrame:
    """A small vivarium-like results frame with positive float values."""
    rows = list(
        itertools.product(
            ["Female", "Male"],
            ["early_neonatal", "late_neonatal"],
            [0, 1],
            [0, 1],
            ["baseline", "intervention"],
        )
    )
    frame = pd.DataFrame(
        rows, columns=["sex", "age_group", "input_draw", "random_seed", "scenario"]
    )
    frame.insert(0, "measure", measure)
    frame.insert(1, "entity_type", "cause")
    frame.insert(2, "entity", "maternal_disorders")
    frame.insert(3, "sub_entity", "total")
    frame["input_draw"] = frame["input_draw"].astype("int64")
    frame["random_seed"] = frame["random_seed"].astype("int64")
    rng = np.random.default_rng(seed)
    frame["value"] = rng.uniform(1.0, 100.0, len(frame))
    return frame[RESULTS_ID_COLUMNS + ["value"]]


def write_measure(
    results_dir: Path,
    name: str,
    frame: pd.DataFrame,
    layout: str = "flat",
    n_shards: int = 3,
    shard_prefix: str = "job",
    **write_kwargs: Any,
) -> None:
    """Write one measure as ``<name>.parquet`` (flat) or ``<name>/`` shards.

    ``sharded`` splits the rows into ``n_shards`` arbitrary shards; ``per_draw``
    writes one shard per draw and seed, as psimulate does, named with
    ``shard_prefix`` so tests can give the two sides different names.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    if layout == "flat":
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, results_dir / f"{name}.parquet", **write_kwargs)
    elif layout == "sharded":
        shard_dir = results_dir / name
        shard_dir.mkdir()
        for i, shard in enumerate(np.array_split(np.arange(len(frame)), n_shards)):
            table = pa.Table.from_pandas(frame.iloc[shard], preserve_index=False)
            pq.write_table(table, shard_dir / f"shard_{i}.parquet", **write_kwargs)
    elif layout == "per_draw":
        shard_dir = results_dir / name
        shard_dir.mkdir()
        for (draw, seed), shard in frame.groupby(["input_draw", "random_seed"]):
            table = pa.Table.from_pandas(shard, preserve_index=False)
            path = shard_dir / f"{shard_prefix}_{int(draw)}_{int(seed)}.parquet"
            pq.write_table(table, path, **write_kwargs)
    else:
        raise ValueError(layout)


def make_results_dir(
    run_root: Path,
    measures: dict[str, pd.DataFrame],
    layout: str = "flat",
    requirements: Optional[list[str]] = None,
    shard_prefix: str = "job",
    **write_kwargs: Any,
) -> Path:
    """Create ``run_root/results`` holding the measures; return the results dir.

    If ``requirements`` is given, it is written as the run root's
    ``requirements.txt``, as psimulate does.
    """
    results_dir = run_root / "results"
    results_dir.mkdir(parents=True)
    for name, frame in measures.items():
        write_measure(
            results_dir, name, frame, layout=layout, shard_prefix=shard_prefix, **write_kwargs
        )
    if requirements is not None:
        (run_root / "requirements.txt").write_text("\n".join(requirements) + "\n")
    return results_dir


def standard_measures() -> dict[str, pd.DataFrame]:
    return {
        "deaths": results_frame("deaths", seed=1),
        "births": results_frame("births", seed=2),
    }


##############
# Artifacts  #
##############


def artifact_frame(seed: int = 0, n_draws: int = 3) -> pd.DataFrame:
    """A small artifact-style frame indexed by demographics with draw columns."""
    ages = [(0.0, 0.01917808), (0.01917808, 0.07671233)]
    rows = [
        (sex, age_start, age_end, year, year + 1)
        for sex in ["Female", "Male"]
        for age_start, age_end in ages
        for year in [2023, 2024]
    ]
    index = pd.MultiIndex.from_tuples(
        rows, names=["sex", "age_start", "age_end", "year_start", "year_end"]
    )
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.uniform(0.1, 1.0, (len(index), n_draws)),
        index=index,
        columns=[f"draw_{i}" for i in range(n_draws)],
    )


def standard_artifact_data() -> dict[str, Any]:
    return {
        "metadata.locations": ["Ethiopia"],
        "population.structure": artifact_frame(seed=1),
        "cause.some_cause.prevalence": artifact_frame(seed=2),
    }


def make_artifact(path: Path, data: dict[str, Any]) -> Path:
    """Write a new artifact at ``path`` holding ``data``, in the dict's key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        # Artifact warns that it is building a new file; that is the point here.
        warnings.simplefilter("ignore")
        artifact = Artifact(path)
    for key, value in data.items():
        artifact.write(key, value)
    return path
