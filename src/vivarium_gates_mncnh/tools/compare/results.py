"""Compare two sets of model results."""

import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from vivarium_gates_mncnh.tools.compare.core import (
    DEFAULT_RTOL,
    ComparisonInputError,
    ComparisonResult,
    Difference,
    DifferenceKind,
    PackageChange,
    TableDiff,
    _align_tables,
    _compare_tables,
    _select,
    _selection_notes,
)

VALUE_COLUMN = "value"
REQUIREMENTS_FILE = "requirements.txt"


def compare_results(
    reference_dir: Union[str, Path],
    candidate_dir: Union[str, Path],
    rtol: float = DEFAULT_RTOL,
    atol: float = 0.0,
    measures: Optional[Sequence[str]] = None,
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
        measures: Glob patterns of the measures to compare, such as
            ``deaths`` or ``*_counts``. All measures are compared if none are
            given.

    Returns:
        The comparison, with one item compared per selected measure on either
        side.

    Raises:
        ComparisonInputError: If either path is not a directory, either
            directory contains no measures, a pattern in ``measures`` matches
            no measure on either side, or a measure cannot be read as parquet.
    """
    reference_dir, candidate_dir = Path(reference_dir), Path(candidate_dir)
    ref_measures = _find_measures(reference_dir, "reference")
    cand_measures = _find_measures(candidate_dir, "candidate")
    all_measures = ref_measures.keys() | cand_measures.keys()
    selected = _select(all_measures, measures, "measure")
    ref_measures = {name: path for name, path in ref_measures.items() if name in selected}
    cand_measures = {name: path for name, path in cand_measures.items() if name in selected}

    result = ComparisonResult(
        title="Results",
        reference=str(reference_dir),
        candidate=str(candidate_dir),
        n_compared=len(ref_measures.keys() | cand_measures.keys()),
        environment_changes=_environment_changes(reference_dir, candidate_dir),
        notes=_selection_notes("measure", measures, len(selected), len(all_measures)),
        rtol=rtol,
        atol=atol,
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
                _compare_measure_paths(
                    measure, ref_measures[measure], cand_measures[measure], rtol, atol
                )
            )
    return result


def _compare_measure_paths(
    measure: str, reference: Path, candidate: Path, rtol: float, atol: float
) -> list[Difference]:
    """Compare one measure a group of shards at a time when the shards can be paired.

    A whole measure can hold hundreds of millions of rows, so it is read in
    groups when both sides are shard directories (see :func:`_shard_groups`).
    Shards whose draw and seed are on one side only are counted as rows only on
    that side without being read. Rows with no match within their group are
    compared again across all groups at the end, so the result is the same as
    comparing the whole measure at once. The one exception is a key repeated in
    different shards, whose copies may pair up differently; psimulate never
    writes those, as every shard is one draw and seed.
    """
    groups = _shard_groups(reference, candidate)
    if groups is None:
        logger.info(f"Comparing {measure}")
        return _compare_whole(measure, reference, candidate, rtol, atol)

    logger.info(f"Comparing {measure} in {len(groups)} groups of shards")
    total = TableDiff()
    ref_unmatched, cand_unmatched = [], []
    columns = None
    for ref_shards, cand_shards in groups:
        if not cand_shards:
            total.only_in_reference += _n_rows(ref_shards)
            continue
        if not ref_shards:
            total.only_in_candidate += _n_rows(cand_shards)
            continue
        ref_frame, cand_frame = _read_shards(ref_shards), _read_shards(cand_shards)
        schema_change = _schema_change(measure, ref_frame, cand_frame)
        if schema_change is not None:
            return [schema_change]
        if columns is None:
            columns = set(ref_frame.columns)
        elif set(ref_frame.columns) != columns:
            # Shards disagree with each other, so let the whole-measure path report it.
            return _compare_whole(measure, reference, candidate, rtol, atol)
        key_columns, value_columns = _key_and_value_columns(columns)
        counts, ref_rows, cand_rows = _align_tables(
            ref_frame, cand_frame, key_columns, value_columns, rtol, atol
        )
        total.n_values += counts.n_values
        total.value_diffs.extend(counts.value_diffs)
        ref_unmatched.append(ref_frame.iloc[ref_rows])
        cand_unmatched.append(cand_frame.iloc[cand_rows])

    if columns is None:
        # No draw and seed is on both sides, so check the columns on one shard of each.
        ref_shard = next(path for shards, _ in groups for path in shards)
        cand_shard = next(path for _, shards in groups for path in shards)
        schema_change = _schema_change(
            measure, _read_parquet(ref_shard), _read_parquet(cand_shard)
        )
        return [schema_change] if schema_change is not None else total.differences(measure)

    # A row unmatched within its group may still match a row in another group.
    counts, _, _ = _align_tables(
        pd.concat(ref_unmatched, ignore_index=True),
        pd.concat(cand_unmatched, ignore_index=True),
        key_columns,
        value_columns,
        rtol,
        atol,
    )
    total.only_in_reference += counts.only_in_reference
    total.only_in_candidate += counts.only_in_candidate
    total.n_values += counts.n_values
    total.value_diffs.extend(counts.value_diffs)
    return total.differences(measure)


def _compare_whole(
    measure: str, reference: Path, candidate: Path, rtol: float, atol: float
) -> list[Difference]:
    return _compare_measure(
        measure, _load_measure(reference), _load_measure(candidate), rtol, atol
    )


_SHARD_ID_COLUMNS = ["input_draw", "random_seed"]


def _shard_groups(
    reference: Path, candidate: Path
) -> Optional[list[tuple[list[Path], list[Path]]]]:
    """Group the shards of two shard directories so each group can be compared alone.

    Shards are paired by file name when both sides hold the same names, which
    psimulate derives from the draw, seed and branch. Otherwise shards are
    grouped by the draw and seed they hold, read from the files, since names
    change whenever psimulate changes how it makes them. A group may hold
    shards on one side only.

    Returns:
        The groups as (reference shards, candidate shards), or None if the two
        sides cannot be grouped: either is not a shard directory, or a shard
        lacks the draw and seed columns or holds more than one draw and seed.
    """
    if not (reference.is_dir() and candidate.is_dir()):
        return None
    ref_names = {path.name for path in reference.glob("*.parquet")}
    cand_names = {path.name for path in candidate.glob("*.parquet")}
    if ref_names == cand_names:
        return [([reference / name], [candidate / name]) for name in sorted(ref_names)]

    ref_ids, cand_ids = _shard_ids(reference), _shard_ids(candidate)
    if ref_ids is None or cand_ids is None:
        return None
    groups: dict[tuple, tuple[list[Path], list[Path]]] = {}
    for side, ids in ((0, ref_ids), (1, cand_ids)):
        for path, shard_id in ids.items():
            groups.setdefault(shard_id, ([], []))[side].append(path)
    return [groups[shard_id] for shard_id in sorted(groups)]


def _shard_ids(directory: Path) -> Optional[dict[Path, tuple]]:
    """Map each shard to the one draw and seed it holds, or None if any shard has other."""
    ids = {}
    for path in sorted(directory.glob("*.parquet")):
        try:
            values = pd.read_parquet(path, columns=_SHARD_ID_COLUMNS).drop_duplicates()
        except Exception:
            # Unreadable files are reported when the whole measure is read.
            return None
        if len(values) != 1:
            return None
        # As floats, so a draw stored as 3 on one side and 3.0 on the other pairs up.
        ids[path] = tuple(float(value) for value in values.iloc[0])
    return ids


def _read_shards(paths: list[Path]) -> pd.DataFrame:
    if len(paths) == 1:
        return _read_parquet(paths[0])
    return pd.concat([_read_parquet(path) for path in paths], ignore_index=True)


def _n_rows(paths: list[Path]) -> int:
    """Count rows from the parquet footers, without reading the data."""
    try:
        return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    except Exception as e:
        raise ComparisonInputError(
            f"Could not read {paths[0].parent} as parquet: {type(e).__name__}: {e}"
        ) from e


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
    schema_change = _schema_change(measure, reference, candidate)
    if schema_change is not None:
        return [schema_change]
    key_columns, value_columns = _key_and_value_columns(reference.columns)
    return _compare_tables(
        measure, reference, candidate, key_columns, value_columns, rtol, atol
    )


def _schema_change(
    measure: str, reference: pd.DataFrame, candidate: pd.DataFrame
) -> Optional[Difference]:
    ref_columns, cand_columns = set(reference.columns), set(candidate.columns)
    if ref_columns == cand_columns:
        return None
    detail = "; ".join(
        f"columns only in {side}: {sorted(map(str, columns))}"
        for side, columns in (
            ("reference", ref_columns - cand_columns),
            ("candidate", cand_columns - ref_columns),
        )
        if columns
    )
    return Difference(measure, DifferenceKind.SCHEMA, detail)


def _key_and_value_columns(columns: Iterable[str]) -> tuple[list[str], list[str]]:
    columns = set(columns)
    value_columns = [VALUE_COLUMN] if VALUE_COLUMN in columns else []
    return sorted(columns - {VALUE_COLUMN}), value_columns


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
        # Normalize as pip does, so Foo_Bar and foo-bar are the same package.
        packages[re.sub(r"[-_.]+", "-", name).lower()] = (name, version.strip())
    return packages
