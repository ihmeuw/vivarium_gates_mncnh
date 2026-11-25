"""The comparison result and the parts that every output type shares."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

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

Source = TypeVar("Source")


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
    reference: Mapping[str, Source],
    candidate: Mapping[str, Source],
    load: Callable[[Source, str], Any],
) -> Comparison:
    """Compare two sets of items, and each item on both sides with ``check_equality``.

    An item is one part of an output, named as it appears in a ``Comparison``:

    - results: a shard path relative to the results directory, such as
      ``deaths/aaaa.parquet``
    - notebooks: a notebook path relative to the notebooks directory, such as
      ``results/executed/maternal.ipynb``
    - artifacts: an artifact key, such as ``population.structure``

    ``reference`` and ``candidate`` map each item to where its data is, such as
    a file or an artifact. ``load`` takes that source and the item, and returns
    the item's data. An error that it raises becomes a ``ComparisonError`` that
    names the item.
    """
    comparison = Comparison()
    for item in sorted(set(reference) | set(candidate)):
        if item not in candidate:
            comparison.only_in_reference.append(item)
        elif item not in reference:
            comparison.only_in_candidate.append(item)
        else:
            with reraise_read_errors(item):
                reference_data = load(reference[item], item)
                candidate_data = load(candidate[item], item)
            if not check_equality(reference_data, candidate_data):
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


def check_equality(reference: Any, candidate: Any) -> bool:
    """Whether two values are exactly equal.

    Frames and series are compared with ``equals``. Their data must be exactly
    equal and have the same dtypes, and NaN in the same place counts as equal.
    Index and column labels are compared by value only, so their dtypes are
    ignored. Index names and series names must also be equal.
    """
    if type(reference) is not type(candidate):
        return False
    if isinstance(reference, pd.Series) and reference.name != candidate.name:
        return False
    if isinstance(reference, (pd.DataFrame, pd.Series)):
        return reference.index.names == candidate.index.names and reference.equals(candidate)
    return bool(reference == candidate)
