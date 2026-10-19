"""Compare two sets of model results."""

import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from loguru import logger

from vivarium_gates_mncnh.tools.compare.core import (
    ComparisonInputError,
    ComparisonResult,
    Difference,
    DifferenceKind,
    PackageChange,
    _compare_tables,
)

VALUE_COLUMN = "value"
REQUIREMENTS_FILE = "requirements.txt"


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
    never register as a change. Identifying columns are matched by value, so a
    column stored as integers on one side and floats on the other (``0`` and
    ``0.0``) identifies the same rows.

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
        ComparisonInputError: If either path is not a directory, either
            directory contains no measures, or a measure cannot be read as
            parquet.
    """
    reference_dir, candidate_dir = Path(reference_dir), Path(candidate_dir)
    ref_measures = _find_measures(reference_dir, "reference")
    cand_measures = _find_measures(candidate_dir, "candidate")

    result = ComparisonResult(
        title="Results",
        reference=str(reference_dir),
        candidate=str(candidate_dir),
        n_compared=len(ref_measures.keys() | cand_measures.keys()),
        environment_changes=_environment_changes(reference_dir, candidate_dir),
    )
    for measure in sorted(ref_measures.keys() | cand_measures.keys()):
        if measure not in cand_measures:
            result.differences.append(
                Difference(measure, DifferenceKind.ONLY_IN_REFERENCE, "measure")
            )
        elif measure not in ref_measures:
            result.differences.append(
                Difference(measure, DifferenceKind.ONLY_IN_CANDIDATE, "measure")
            )
        else:
            result.differences.extend(
                _compare_measure(
                    measure,
                    _load_measure(ref_measures[measure]),
                    _load_measure(cand_measures[measure]),
                    rtol,
                    atol,
                )
            )
    return result


def _find_measures(results_dir: Path, side: str) -> dict[str, Path]:
    """Map each measure name in a results directory to its file or shard directory."""
    if not results_dir.is_dir():
        raise ComparisonInputError(f"{side} results path is not a directory: {results_dir}")
    measures = {}
    for path in sorted(results_dir.iterdir()):
        is_flat = path.is_file() and path.suffix == ".parquet"
        is_sharded = path.is_dir() and any(path.glob("*.parquet"))
        if not (is_flat or is_sharded):
            continue
        name = path.stem if is_flat else path.name
        if name in measures:
            logger.warning(f"Measure '{name}' is both a file and a directory; using {path}")
        measures[name] = path
    if not measures:
        raise ComparisonInputError(f"No parquet measures found in {side} {results_dir}")
    return measures


def _load_measure(path: Path) -> pd.DataFrame:
    if path.is_file():
        return _read_parquet(path)
    return pd.concat(
        [_read_parquet(shard) for shard in sorted(path.glob("*.parquet"))],
        ignore_index=True,
    )


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        raise ComparisonInputError(
            f"Could not read {path} as parquet: {type(e).__name__}: {e}"
        ) from e


def _compare_measure(
    measure: str, reference: pd.DataFrame, candidate: pd.DataFrame, rtol: float, atol: float
) -> list[Difference]:
    ref_columns, cand_columns = set(reference.columns), set(candidate.columns)
    if ref_columns != cand_columns:
        detail = "; ".join(
            f"columns only in {side}: {sorted(map(str, columns))}"
            for side, columns in (
                ("reference", ref_columns - cand_columns),
                ("candidate", cand_columns - ref_columns),
            )
            if columns
        )
        return [Difference(measure, DifferenceKind.SCHEMA, detail)]
    value_columns = [VALUE_COLUMN] if VALUE_COLUMN in ref_columns else []
    key_columns = sorted(ref_columns - {VALUE_COLUMN})
    return _compare_tables(
        measure, reference, candidate, key_columns, value_columns, rtol, atol
    )


def _environment_changes(reference_dir: Path, candidate_dir: Path) -> list[PackageChange]:
    ref_file = reference_dir.parent / REQUIREMENTS_FILE
    cand_file = candidate_dir.parent / REQUIREMENTS_FILE
    if not (ref_file.is_file() and cand_file.is_file()):
        return []
    ref_packages, cand_packages = _read_requirements(ref_file), _read_requirements(cand_file)
    changes = []
    for key in sorted(ref_packages.keys() | cand_packages.keys()):
        ref_name, ref_version = ref_packages.get(key, (None, None))
        cand_name, cand_version = cand_packages.get(key, (None, None))
        if ref_version != cand_version or ref_name is None or cand_name is None:
            changes.append(PackageChange(ref_name or cand_name, ref_version, cand_version))
    return changes


def _read_requirements(path: Path) -> dict[str, tuple[str, str]]:
    """Read pip-freeze style requirements as {normalized name: (name, version)}."""
    packages = {}
    for line in path.read_text().splitlines():
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith(("#", "-")):
            continue
        if " @ " in line:
            name, version = line.split(" @ ", 1)
        else:
            line = line.split(";", 1)[0].strip()
            match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._\-]*)(\[[^\]]*\])?\s*(.*)$", line)
            if match is None:
                continue
            name, version = match.group(1), match.group(3)
            version = version[2:] if version.startswith("==") else version
        name = name.split("[", 1)[0].strip()
        packages[_normalize(name)] = (name, version.strip())
    return packages


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()
