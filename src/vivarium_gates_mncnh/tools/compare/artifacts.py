"""Compare two artifacts."""

from pathlib import Path
from typing import Any, Union

import pandas as pd
from vivarium.artifact import Artifact

from vivarium_gates_mncnh.tools.compare.core import (
    ComparisonInputError,
    ComparisonResult,
    Difference,
    DifferenceKind,
    _compare_tables,
    _same_object,
)

KEYSPACE_KEY = "metadata.keyspace"
"""Lists the artifact's own keys, so it is covered by comparing the key sets."""


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
    reference_path, candidate_path = Path(reference_path), Path(candidate_path)
    # Check before constructing an Artifact, which creates a missing file.
    for side, path in (("reference", reference_path), ("candidate", candidate_path)):
        if not path.is_file():
            raise ComparisonInputError(f"{side} artifact is not an existing file: {path}")
    reference, candidate = Artifact(reference_path), Artifact(candidate_path)
    ref_keys = {str(key) for key in reference.keys}
    cand_keys = {str(key) for key in candidate.keys}

    result = ComparisonResult(
        title="Artifacts",
        reference=str(reference_path),
        candidate=str(candidate_path),
        n_compared=len(ref_keys | cand_keys),
    )
    for key in sorted(ref_keys | cand_keys):
        if key not in cand_keys:
            result.differences.append(
                Difference(key, DifferenceKind.ONLY_IN_REFERENCE, "key")
            )
        elif key not in ref_keys:
            result.differences.append(
                Difference(key, DifferenceKind.ONLY_IN_CANDIDATE, "key")
            )
        elif key != KEYSPACE_KEY:
            try:
                result.differences.extend(
                    _compare_data(key, reference.load(key), candidate.load(key), rtol, atol)
                )
            finally:
                reference.clear_cache()
                candidate.clear_cache()
    return result


def _compare_data(
    key: str, reference: Any, candidate: Any, rtol: float, atol: float
) -> list[Difference]:
    if isinstance(reference, pd.Series) and isinstance(candidate, pd.Series):
        reference, candidate = reference.to_frame("value"), candidate.to_frame("value")
    ref_is_frame = isinstance(reference, pd.DataFrame)
    if ref_is_frame != isinstance(candidate, pd.DataFrame):
        return [
            Difference(
                key,
                DifferenceKind.SCHEMA,
                f"type changed from {type(reference).__name__} to {type(candidate).__name__}",
            )
        ]
    if not ref_is_frame:
        if _same_object(reference, candidate):
            return []
        return [Difference(key, DifferenceKind.VALUES, "value changed")]

    ref_levels, cand_levels = _level_names(reference), _level_names(candidate)
    ref_columns, cand_columns = set(reference.columns), set(candidate.columns)
    schema_changes = []
    if set(ref_levels) != set(cand_levels):
        schema_changes.append(f"index levels {ref_levels} -> {cand_levels}")
    if ref_columns != cand_columns:
        schema_changes.append(
            f"columns {sorted(map(str, ref_columns))} -> {sorted(map(str, cand_columns))}"
        )
    if schema_changes:
        return [Difference(key, DifferenceKind.SCHEMA, "; ".join(schema_changes))]

    value_columns = list(reference.columns)
    return _compare_tables(
        key,
        _flatten(reference),
        _flatten(candidate),
        key_columns=ref_levels,
        value_columns=value_columns,
        rtol=rtol,
        atol=atol,
    )


def _level_names(data: pd.DataFrame) -> list[str]:
    return [
        str(name) if name is not None else f"level_{i}"
        for i, name in enumerate(data.index.names)
    ]


def _flatten(data: pd.DataFrame) -> pd.DataFrame:
    """Move the index into columns named as :func:`_level_names` names them."""
    data = data.copy()
    data.index = data.index.set_names(_level_names(data))
    return data.reset_index()
