"""Compare two psimulate results directories shard by shard."""

from pathlib import Path

import pandas as pd

from vivarium_gates_mncnh.compare.base import (
    Comparison,
    compare_items,
    equal_values,
    find_files,
    match_column_order,
)

RESULTS_PATTERN = "*/*.parquet"

VALUE_COLUMN = "value"
"""The column that holds a shard's results. Every other column identifies a row."""


def compare_results(reference: Path, candidate: Path) -> Comparison:
    """Compare the measure files in two psimulate results directories.

    Files are paired by their path relative to each directory. Each pair is read
    with ``pd.read_parquet`` and its default options, which is how results are
    read for analysis. The two frames must be exactly equal. Differences that
    ``pd.read_parquet`` does not show, such as compression, row groups, or
    string width, are ignored.

    The columns of two frames are matched by label, and their rows by the values
    in every column except ``value``, before the values are compared. So a
    difference only in row or column order is not reported.

    Raises
    ------
    ComparisonError
        If either path is not a directory, has no measure files, or has a file
        that cannot be read. A shard whose rows are in a different order and
        have a repeated set of values outside ``value`` also raises this,
        because its rows cannot be matched.
    """
    reference_files = find_files(reference, "reference", RESULTS_PATTERN)
    candidate_files = find_files(candidate, "candidate", RESULTS_PATTERN)
    return compare_items(
        reference_files,
        candidate_files,
        lambda name: _equal_frames(reference_files[name], candidate_files[name]),
    )


def _equal_frames(reference: Path, candidate: Path) -> bool:
    """Whether two parquet files read into equal frames, in any row or column order.

    The dtypes of the columns are compared first, because the columns that
    identify a row become index labels, and ``equal_values`` compares index
    labels by value only.
    """
    reference_frame = pd.read_parquet(reference)
    candidate_frame = match_column_order(reference_frame, pd.read_parquet(candidate))
    if not reference_frame.dtypes.equals(candidate_frame.dtypes):
        return False
    return equal_values(_by_row_labels(reference_frame), _by_row_labels(candidate_frame))


def _by_row_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Move every column except ``value`` into the index of ``frame``.

    psimulate writes each shard with an unnamed ``RangeIndex``, which holds
    only row positions, so that index is dropped. Any other index is kept as
    the outer levels.
    """
    positions_only = isinstance(frame.index, pd.RangeIndex) and frame.index.name is None
    row_labels = [column for column in frame.columns if column != VALUE_COLUMN]
    return frame.set_index(row_labels, append=not positions_only)
