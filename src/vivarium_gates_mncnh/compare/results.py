"""Compare two psimulate results directories shard by shard."""

from pathlib import Path

import pandas as pd

from vivarium_gates_mncnh.compare.base import (
    Comparison,
    compare_items,
    equal_values,
    find_files,
)

RESULTS_PATTERN = "*/*.parquet"


def compare_results(reference: Path, candidate: Path) -> Comparison:
    """Compare the measure files in two psimulate results directories.

    Files are paired by their path relative to each directory. Each pair is read
    with ``pd.read_parquet`` and its default options, which is how results are
    read for analysis. The two frames must be exactly equal. Differences that
    ``pd.read_parquet`` does not show, such as compression, row groups, or
    string width, are ignored. The columns of two frames are matched by label
    before the values are compared, so a difference only in column order is not
    reported.

    Raises
    ------
    ComparisonError
        If either path is not a directory, has no measure files, or has a file
        that cannot be read.
    """
    reference_files = find_files(reference, "reference", RESULTS_PATTERN)
    candidate_files = find_files(candidate, "candidate", RESULTS_PATTERN)
    return compare_items(
        reference_files,
        candidate_files,
        lambda name: _equal_frames(reference_files[name], candidate_files[name]),
    )


def _equal_frames(reference: Path, candidate: Path) -> bool:
    """Whether two parquet files read into exactly equal frames."""
    return equal_values(pd.read_parquet(reference), pd.read_parquet(candidate))
