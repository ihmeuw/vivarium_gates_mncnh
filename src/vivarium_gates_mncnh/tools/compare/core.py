"""Shared types and value comparison for the compare tools."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class ComparisonInputError(ValueError):
    """Raised when a comparison cannot be made from the inputs given.

    For example, a path that does not exist or is the wrong kind of path. A
    difference between two valid inputs is never an error; it is reported as a
    :class:`Difference`.
    """


class DifferenceKind(str, Enum):
    """What kind of change a :class:`Difference` records."""

    VALUES = "values changed"
    ONLY_IN_REFERENCE = "only in reference"
    ONLY_IN_CANDIDATE = "only in candidate"
    SCHEMA = "schema changed"
    OUTPUT = "output changed"
    PLOT = "plot data changed"
    EXECUTION_FAILED = "execution failed"


@dataclass(frozen=True)
class ValueDiff:
    """Summary of how two aligned sets of values differ.

    Attributes:
        n_changed: Number of positions whose values differ beyond tolerance,
            including positions where exactly one side is NaN.
        max_abs_diff: Largest ``|candidate - reference|`` over changed positions
            where both values are finite numbers, or None if there are none.
        max_rel_diff: Largest ``|candidate - reference| / |reference|`` over the
            same positions; ``inf`` where the reference is zero. None if there
            are none.
    """

    n_changed: int
    max_abs_diff: Optional[float] = None
    max_rel_diff: Optional[float] = None


@dataclass(frozen=True)
class Difference:
    """One thing that moved between the reference and the candidate.

    Attributes:
        item: What moved, e.g. a measure name, an artifact key, or
            ``"<notebook>.ipynb, cell <n>"``.
        kind: The kind of change.
        detail: A short human-readable description for the report.
        value_diff: For value changes, how many values moved and by how much.
    """

    item: str
    kind: DifferenceKind
    detail: str = ""
    value_diff: Optional[ValueDiff] = None


@dataclass(frozen=True)
class PackageChange:
    """A package whose installed version differs between two runs.

    A version of None means the package is absent on that side.
    """

    name: str
    reference: Optional[str]
    candidate: Optional[str]


@dataclass
class ComparisonResult:
    """The outcome of comparing one reference against one candidate.

    Attributes:
        title: Heading for this comparison in the report, e.g. ``"Results"``.
        reference: The reference input, as given.
        candidate: The candidate input, as given.
        n_compared: How many items (measures, keys or notebooks) were compared.
        differences: Everything that moved. Empty means the two compare clean.
        notes: Informational lines that do not count as movement, such as
            plots that could not be compared.
        environment_changes: Packages whose versions differ between the two
            runs, when that is known.
    """

    title: str
    reference: str
    candidate: str
    n_compared: int = 0
    differences: list[Difference] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    environment_changes: list[PackageChange] = field(default_factory=list)

    @property
    def moved(self) -> bool:
        """Whether anything moved between the reference and the candidate."""
        return bool(self.differences)


def compare_values(
    reference: pd.Series,
    candidate: pd.Series,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Optional[ValueDiff]:
    """Compare two aligned series value by value.

    Numeric values are equal when ``|candidate - reference| <= atol + rtol *
    |reference|``; with the defaults that means exactly equal. Two NaNs are
    equal; NaN against a number is a change. Non-numeric values are compared
    for equality. Integer and float values that are numerically equal are
    equal.

    Args:
        reference: Reference values.
        candidate: Candidate values, with the same index as ``reference``.
        rtol: Relative tolerance, measured against the reference value.
        atol: Absolute tolerance.

    Returns:
        None if every value is equal within tolerance, otherwise a
        :class:`ValueDiff` describing the change.

    Raises:
        ValueError: If the two series are not indexed identically.
    """
    raise NotImplementedError
