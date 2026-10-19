"""Compare two sets of model results."""

from pathlib import Path
from typing import Union

from vivarium_gates_mncnh.tools.compare.core import ComparisonResult


def compare_results(
    reference_dir: Union[str, Path],
    candidate_dir: Union[str, Path],
    rtol: float = 0.0,
    atol: float = 0.0,
) -> ComparisonResult:
    """Compare two directories of simulation results value by value.

    Each directory is one run's ``results/`` directory for one location. A
    measure is either a ``<measure>.parquet`` file or a ``<measure>/``
    directory of parquet shards, as psimulate writes them; the two layouts are
    interchangeable, so the same data in either compares clean. Within a
    measure, ``value`` holds the values and every other column identifies a
    row, so row order, column order, compression and other file-level detail
    never register as a change.

    Reported as differences:

    - a measure present on only one side;
    - a measure whose identifying columns differ (schema change);
    - rows present on only one side, per measure;
    - values that moved beyond tolerance, per measure, with how many moved and
      the largest absolute and relative change.

    When both directories' parent (the run root) holds psimulate's
    ``requirements.txt``, the packages whose versions differ are recorded in
    :attr:`ComparisonResult.environment_changes`. If either is missing, that
    list is left empty.

    Args:
        reference_dir: The reference run's results directory.
        candidate_dir: The candidate run's results directory.
        rtol: Relative tolerance for values; see
            :func:`~vivarium_gates_mncnh.tools.compare.core.compare_values`.
        atol: Absolute tolerance for values.

    Returns:
        The comparison, with one item compared per measure on either side.

    Raises:
        ComparisonInputError: If either path is not a directory, or either
            directory contains no measures.
    """
    raise NotImplementedError
