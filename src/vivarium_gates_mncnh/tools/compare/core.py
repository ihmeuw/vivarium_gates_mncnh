"""Shared types and value comparison for the compare tools."""

import numbers
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd


class ComparisonInputError(ValueError):
    """Raised when a comparison cannot be made from the inputs given.

    For example, a path that does not exist or is the wrong kind of path, or a
    file that cannot be read as the expected format. A difference between two
    valid inputs is never an error; it is reported as a :class:`Difference`.
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
    if not reference.index.equals(candidate.index):
        raise ValueError("Reference and candidate must be indexed identically.")
    if reference.empty:
        return None

    ref_values, cand_values = _as_float(reference), _as_float(candidate)
    if ref_values is None or cand_values is None:
        n_changed = sum(
            not _same_object(r, c) for r, c in zip(reference.tolist(), candidate.tolist())
        )
        return ValueDiff(n_changed) if n_changed else None

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        abs_diff = np.abs(cand_values - ref_values)
        within = abs_diff <= atol + rtol * np.abs(ref_values)
        both_nan = np.isnan(ref_values) & np.isnan(cand_values)
        changed = ~((ref_values == cand_values) | both_nan | within)
        if not changed.any():
            return None
        finite = changed & np.isfinite(ref_values) & np.isfinite(cand_values)
        if not finite.any():
            return ValueDiff(int(changed.sum()))
        rel_diff = abs_diff[finite] / np.abs(ref_values[finite])
    return ValueDiff(
        n_changed=int(changed.sum()),
        max_abs_diff=float(abs_diff[finite].max()),
        max_rel_diff=float(rel_diff.max()),
    )


def _as_float(values: pd.Series) -> Optional[np.ndarray]:
    """Return the values as a float array, or None if they are not all numeric."""
    if pd.api.types.is_numeric_dtype(values.dtype):
        return values.to_numpy(dtype=float, na_value=np.nan)
    if values.dtype == object and all(
        isinstance(value, numbers.Number) or _is_missing(value) for value in values.tolist()
    ):
        return pd.to_numeric(values).to_numpy(dtype=float, na_value=np.nan)
    return None


def _is_missing(value: Any) -> bool:
    return pd.api.types.is_scalar(value) and bool(pd.isna(value))


def _same_object(reference: Any, candidate: Any) -> bool:
    """Whether two arbitrary values are equal, treating two missing values as equal."""
    if _is_missing(reference) or _is_missing(candidate):
        return _is_missing(reference) and _is_missing(candidate)
    try:
        equal = reference == candidate
        if isinstance(equal, (bool, np.bool_)):
            return bool(equal)
        return np.shape(reference) == np.shape(candidate) and bool(np.all(equal))
    except Exception:
        return repr(reference) == repr(candidate)


def _merge_value_diffs(diffs: list[ValueDiff]) -> Optional[ValueDiff]:
    """Combine several value diffs into one, or None if there are none."""
    if not diffs:
        return None
    abs_diffs = [d.max_abs_diff for d in diffs if d.max_abs_diff is not None]
    rel_diffs = [d.max_rel_diff for d in diffs if d.max_rel_diff is not None]
    return ValueDiff(
        n_changed=sum(d.n_changed for d in diffs),
        max_abs_diff=max(abs_diffs) if abs_diffs else None,
        max_rel_diff=max(rel_diffs) if rel_diffs else None,
    )


_OCCURRENCE = "__occurrence__"


def _keyed(
    frame: pd.DataFrame, key_columns: list[str], value_columns: list[str]
) -> pd.DataFrame:
    """Index a table by its key columns, cast to str so dtype differences don't matter.

    Rows sharing a key are told apart by their order after sorting on the values,
    so duplicate keys align independently of row order.
    """
    frame = frame.reset_index(drop=True)
    data = pd.DataFrame(
        {column: frame[column].astype(str) for column in key_columns}, index=frame.index
    )
    for column in value_columns:
        data[column] = frame[column]
    for sort_by in (key_columns + value_columns, key_columns):
        if not sort_by:
            break
        try:
            data = data.sort_values(sort_by, kind="stable")
            break
        except TypeError:
            continue
    if key_columns:
        data[_OCCURRENCE] = data.groupby(key_columns, sort=False).cumcount()
    else:
        data[_OCCURRENCE] = np.arange(len(data))
    return data.set_index(key_columns + [_OCCURRENCE])


def _compare_tables(
    item: str,
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    key_columns: list[str],
    value_columns: list[str],
    rtol: float,
    atol: float,
) -> list[Difference]:
    """Align two tables on their key columns and compare their value columns.

    Both tables must hold every key and value column. Rows present on one side
    only and values that moved are reported against ``item``.
    """
    ref = _keyed(reference, key_columns, value_columns)
    cand = _keyed(candidate, key_columns, value_columns)
    differences = []
    for kind, side, rows in (
        (DifferenceKind.ONLY_IN_REFERENCE, "reference", ref.index.difference(cand.index)),
        (DifferenceKind.ONLY_IN_CANDIDATE, "candidate", cand.index.difference(ref.index)),
    ):
        if len(rows):
            differences.append(Difference(item, kind, f"{len(rows)} row(s) only in {side}"))

    shared = ref.index.intersection(cand.index)
    ref, cand = ref.reindex(shared), cand.reindex(shared)
    numeric = [
        column
        for column in value_columns
        if pd.api.types.is_numeric_dtype(ref[column].dtype)
        and pd.api.types.is_numeric_dtype(cand[column].dtype)
    ]
    diffs = []
    if numeric:
        diffs.append(
            compare_values(
                pd.Series(ref[numeric].to_numpy(dtype=float, na_value=np.nan).ravel()),
                pd.Series(cand[numeric].to_numpy(dtype=float, na_value=np.nan).ravel()),
                rtol=rtol,
                atol=atol,
            )
        )
    for column in value_columns:
        if column not in numeric:
            diffs.append(compare_values(ref[column], cand[column], rtol=rtol, atol=atol))
    value_diff = _merge_value_diffs([diff for diff in diffs if diff is not None])
    if value_diff is not None:
        n_values = len(shared) * len(value_columns)
        differences.append(
            Difference(
                item,
                DifferenceKind.VALUES,
                f"{value_diff.n_changed} of {n_values} value(s) changed",
                value_diff,
            )
        )
    return differences
