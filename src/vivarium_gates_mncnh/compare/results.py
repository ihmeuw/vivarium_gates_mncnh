"""Compare two psimulate results directories shard by shard."""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from vivarium_gates_mncnh.compare.base import (
    Comparison,
    ComparisonError,
    compare_items,
    equal_values,
    find_files,
    reraise_read_errors,
)

RESULTS_PATTERN = "*/*.parquet"
BRANCHES_FILE = "branches.yaml"
"""The file in a run root that holds the expanded branch configurations."""
DRAW_AND_SEED_COLUMNS = ("input_draw", "random_seed")


def compare_results(reference: Path, candidate: Path) -> Comparison:
    """Compare the measure files in two psimulate results directories.

    ``reference`` and ``candidate`` are the ``results`` directories of two
    psimulate runs. Each run root, the parent of its ``results`` directory,
    must hold the ``branches.yaml`` file that psimulate writes. The file is read
    to find the branch columns. It is not compared.

    Shards are paired by measure and by the draw, seed, and branch values they
    hold, not by file name. psimulate adds one column to each shard for each of
    these values, and each column holds one value. A shard is named in the
    ``Comparison`` by its measure and these values, such as
    ``deaths: scenario=baseline, input_draw=0, random_seed=1``.

    Each pair is read with ``pd.read_parquet`` and its default options, which is
    how results are read for analysis. The two frames must be exactly equal.
    Differences that ``pd.read_parquet`` does not show, such as compression, row
    groups, or string width, are ignored.

    Raises
    ------
    ComparisonError
        If either path is not a directory, has no measure files, has no
        ``branches.yaml`` in its run root, or has a file that cannot be read.
        Also if a shard does not hold exactly one value in each draw, seed, and
        branch column, or if two shards of one measure on one side hold the same
        values.
    """
    reference_shards = _find_shards(reference, "reference")
    candidate_shards = _find_shards(candidate, "candidate")
    return compare_items(
        reference_shards,
        candidate_shards,
        lambda name: _equal_frames(reference_shards[name], candidate_shards[name]),
    )


def _find_shards(results: Path, side: str) -> dict[str, Path]:
    """Return the shards in ``results``, keyed by measure and identifying values."""
    files = find_files(results, side, RESULTS_PATTERN)
    columns = _identifying_columns(results.parent / BRANCHES_FILE, side)
    shards: dict[str, Path] = {}
    file_names: dict[str, str] = {}
    for file_name, path in sorted(files.items()):
        try:
            with reraise_read_errors(f"{side} shard {file_name}"):
                name = _shard_name(path, columns)
        except _NotIdentifiable as error:
            raise ComparisonError(f"{side} shard {file_name} has {error}") from error
        if name in shards:
            raise ComparisonError(
                f"{side} shards {file_names[name]} and {file_name} both hold {name}"
            )
        shards[name] = path
        file_names[name] = file_name
    return shards


def _identifying_columns(branches_file: Path, side: str) -> list[str]:
    """Return the columns that psimulate adds to each shard of a run.

    psimulate adds one column for each branch configuration key, named after the
    last part of the key, and then ``input_draw`` and ``random_seed``.
    """
    if not branches_file.is_file():
        raise ComparisonError(f"{side} run root has no {BRANCHES_FILE}: {branches_file}")
    with reraise_read_errors(f"{side} {BRANCHES_FILE}", (OSError, yaml.YAMLError)):
        branches = yaml.safe_load(branches_file.read_text())
    if not isinstance(branches, list) or not all(isinstance(b, dict) for b in branches):
        raise ComparisonError(
            f"{side} {BRANCHES_FILE} is not a list of branch configurations: "
            f"{branches_file}"
        )
    keys = dict.fromkeys(key for branch in branches for key in _flatten_keys(branch))
    branch_columns = [key.split(".")[-1] for key in keys]
    return list(dict.fromkeys([*branch_columns, *DRAW_AND_SEED_COLUMNS]))


def _flatten_keys(configuration: Mapping[str, Any], prefix: str = "") -> Iterator[str]:
    """Yield the dotted path of each leaf in a nested branch configuration."""
    for key, value in configuration.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            yield from _flatten_keys(value, f"{path}.")
        else:
            yield path


class _NotIdentifiable(Exception):
    """A shard does not hold exactly one value in each identifying column."""


def _shard_name(path: Path, columns: list[str]) -> str:
    """Name a shard by its measure and the one value in each identifying column.

    Only the identifying columns are read, so memory stays low.
    """
    schema_names = pq.read_schema(path).names
    missing = [column for column in columns if column not in schema_names]
    if missing:
        raise _NotIdentifiable(f"no column {', '.join(missing)}")
    frame = pd.read_parquet(path, columns=columns)
    values = []
    for column in columns:
        unique = frame[column].drop_duplicates()
        if len(unique) != 1:
            raise _NotIdentifiable(f"{len(unique)} values in column {column}, not one")
        values.append(f"{column}={unique.iloc[0]}")
    return f"{path.parent.name}: {', '.join(values)}"


def _equal_frames(reference: Path, candidate: Path) -> bool:
    """Whether two parquet files read into exactly equal frames."""
    return equal_values(pd.read_parquet(reference), pd.read_parquet(candidate))
