"""Compare two psimulate results directories shard by shard."""

from pathlib import Path

import pandas as pd

from vivarium_gates_mncnh.compare.utilities import Comparison, compare_items, find_files

RESULTS_PATTERN = "*/*.parquet"


def compare_results(reference: Path, candidate: Path) -> Comparison:
    """Compare the measure files in two psimulate results directories.

    Files are paired by their path relative to each directory. psimulate 3.0.0
    and later names each shard after a hash of its draw, seed, and branch
    configuration (``generate_task_id``), so the same task has the same file
    name in both runs. If a psimulate release changes that hash, we can no
    longer rely on this pairing.

    Each pair is read with ``pd.read_parquet`` and its default options, which is
    how results are read for analysis. The two frames must be exactly equal.
    Differences that ``pd.read_parquet`` does not show, such as compression, row
    groups, or string width, are ignored.

    Raises
    ------
    ComparisonError
        If either path is not a directory, has no measure files, or has a file
        that cannot be read.
    """
    reference_files = find_files(reference, "reference", RESULTS_PATTERN)
    candidate_files = find_files(candidate, "candidate", RESULTS_PATTERN)
    files = {"reference": reference_files, "candidate": candidate_files}
    return compare_items(
        reference=reference_files.keys(),
        candidate=candidate_files.keys(),
        load=lambda side, name: pd.read_parquet(files[side][name]),
    )
