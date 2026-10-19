"""Compare two artifacts."""

from pathlib import Path
from typing import Union

from vivarium_gates_mncnh.tools.compare.core import ComparisonResult


def compare_artifacts(
    reference_path: Union[str, Path],
    candidate_path: Union[str, Path],
    rtol: float = 0.0,
    atol: float = 0.0,
) -> ComparisonResult:
    """Compare two artifacts key by key, by their values.

    Every key is loaded from both artifacts and compared by value, metadata
    keys included, so differences in how the HDF files were written never
    register as a change. DataFrames are compared after sorting, so row and
    column order do not matter; other values are compared for equality.

    Reported as differences:

    - a key present in only one artifact;
    - a key whose data changed shape (index levels or columns differ);
    - a key whose values moved beyond tolerance, with how many moved and the
      largest absolute and relative change.

    Neither artifact is modified, and a missing path is never created.

    Args:
        reference_path: Path to the reference ``.hdf`` artifact.
        candidate_path: Path to the candidate ``.hdf`` artifact.
        rtol: Relative tolerance for values; see
            :func:`~vivarium_gates_mncnh.tools.compare.core.compare_values`.
        atol: Absolute tolerance for values.

    Returns:
        The comparison, with one item compared per key on either side.

    Raises:
        ComparisonInputError: If either path is not an existing file.
    """
    raise NotImplementedError
