"""The comparison result and the parts that every output type shares."""

from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

READ_ERRORS: tuple[type[Exception], ...] = (
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    KeyError,
)
"""The errors that a reader raises for a file it cannot read.

pyarrow raises ``ArrowInvalid``, a ``ValueError``, for a file that is not
parquet. tables raises ``HDF5ExtError``, a ``RuntimeError``, for a file that is
not HDF. ``Artifact`` raises a ``TypeError`` for an HDF file that pandas wrote.
"""


class ComparisonError(Exception):
    """The inputs cannot be compared.

    This is separate from a difference: a caller can treat it as a broken
    comparison, not a failed one.
    """


@dataclass
class Comparison:
    """The result of comparing a reference output with a candidate output.

    Each list holds relative file names or keys, in sorted order.
    """

    only_in_reference: list[str] = field(default_factory=list)
    only_in_candidate: list[str] = field(default_factory=list)
    different: list[str] = field(default_factory=list)
    """Files or keys present on both sides whose content differs."""

    @property
    def equal(self) -> bool:
        """Whether the two outputs are equal."""
        return not (self.only_in_reference or self.only_in_candidate or self.different)


@contextmanager
def reraise_read_errors(
    read_target: str, errors: tuple[type[Exception], ...] = READ_ERRORS
) -> Iterator[None]:
    """Turn an error from a reader into a ``ComparisonError`` that names ``read_target``."""
    try:
        yield
    except errors as error:
        raise ComparisonError(f"cannot read {read_target}: {error}") from error


def compare_items(
    reference: Collection[str],
    candidate: Collection[str],
    equal: Callable[[str], bool],
) -> Comparison:
    """Compare two sets of items, and each item on both sides with ``equal``.

    An item is one part of an output, named as it appears in a ``Comparison``:

    - results: a shard path relative to the results directory, such as
      ``deaths/aaaa.parquet``
    - notebooks: a notebook path relative to the notebooks directory, such as
      ``results/executed/maternal.ipynb``
    - artifacts: an artifact key, such as ``population.structure``

    ``equal`` reads both sides of one item. An error that it raises while
    reading becomes a ``ComparisonError`` that names the item.
    """
    comparison = Comparison()
    for item in sorted(set(reference) | set(candidate)):
        if item not in candidate:
            comparison.only_in_reference.append(item)
        elif item not in reference:
            comparison.only_in_candidate.append(item)
        else:
            with reraise_read_errors(item):
                item_is_equal = equal(item)
            if not item_is_equal:
                comparison.different.append(item)
    return comparison


def find_files(path: Path, side: str, pattern: str) -> dict[str, Path]:
    """Return the files under ``path`` that match ``pattern``, keyed by relative path.

    ``side`` is "reference" or "candidate", for error messages.
    """
    if not path.is_dir():
        raise ComparisonError(f"{side} path is not a directory: {path}")
    files = {
        file.relative_to(path).as_posix(): file
        for file in path.glob(pattern)
        if file.is_file()
    }
    if not files:
        raise ComparisonError(f"no files match '{pattern}' in {side} path: {path}")
    return files


def equal_values(reference: Any, candidate: Any) -> bool:
    """Whether two values are exactly equal.

    Frames and series are compared with ``equals``. Their data must be exactly
    equal and have the same dtypes, and NaN in the same place counts as equal.
    Index and column labels are compared by value only, so their dtypes are
    ignored. Index names and series names must also be equal.

    The columns and rows of two frames are matched by label with
    ``match_column_order`` and ``match_row_order`` before they are compared, so
    a difference only in their order does not count.
    """
    if type(reference) is not type(candidate):
        return False
    if isinstance(reference, pd.DataFrame):
        candidate = match_row_order(reference, match_column_order(reference, candidate))
    if isinstance(reference, pd.Series) and reference.name != candidate.name:
        return False
    if isinstance(reference, (pd.DataFrame, pd.Series)):
        return reference.index.names == candidate.index.names and reference.equals(candidate)
    return bool(reference == candidate)


def match_column_order(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Return ``candidate`` with its columns in the order of ``reference``.

    Columns are matched by label. If the two frames do not have the same column
    labels, ``candidate`` is returned unchanged, so a later comparison reports
    the difference.

    Raises
    ------
    ValueError
        If the labels are the same but in a different order, and a label
        appears more than once.
    """
    if not _same_labels(reference.columns, candidate.columns):
        return candidate
    return candidate.reindex(columns=reference.columns)


def match_row_order(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Return ``candidate`` with its rows in the order of ``reference``.

    Rows are matched by their index labels. If the two frames do not have the
    same index labels, ``candidate`` is returned unchanged, so a later
    comparison reports the difference. The index keeps the names of
    ``candidate``, so a later comparison also reports different index names.

    Raises
    ------
    ValueError
        If the labels are the same but in a different order, and a label
        appears more than once.
    """
    if not _same_labels(reference.index, candidate.index):
        return candidate
    matched = candidate.reindex(index=reference.index)
    matched.index = matched.index.set_names(candidate.index.names)
    return matched


def _same_labels(reference: pd.Index, candidate: pd.Index) -> bool:
    """Whether two indexes hold the same labels, in any order.

    Labels are compared by value, as ``equals`` compares them, so ``1`` and
    ``1.0`` match, and NaN matches NaN. Indexes with different numbers of
    levels do not match. ``isin`` raises for them.
    """
    return (
        reference.nlevels == candidate.nlevels
        and len(reference) == len(candidate)
        and bool(reference.isin(candidate).all())
        and bool(candidate.isin(reference).all())
    )
