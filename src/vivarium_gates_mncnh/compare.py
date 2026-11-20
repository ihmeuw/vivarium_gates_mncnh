"""
Compare model outputs from two environments.

These functions answer one question after an environment update: did the
artifacts, the simulation results, or the executed notebooks change? Each
returns a ``Comparison``, which later CI stages can act on or save.

Two outputs are the same when they have the same set of files and each pair of
files is equal. Which files count, and how a pair is compared, depends on the
output type:

- artifacts: the keys in two artifact files, as vivarium's ``Artifact`` reads them
- results: ``<measure>/*.parquet`` in a psimulate ``results`` directory, as
  ``pd.read_parquet`` reads them
- notebooks: ``**/executed/*.ipynb`` in a notebooks directory, byte for byte

This module imports nothing from the model, so a broken model cannot make the
comparison itself fail. ``compare_artifacts`` imports vivarium, so that it reads
artifacts as a simulation does.
"""

import filecmp
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from vivarium.artifact.artifact import Artifact

RESULTS_PATTERN = "*/*.parquet"
NOTEBOOKS_PATTERN = "**/executed/*.ipynb"
KEYSPACE_KEY = "metadata.keyspace"
"""The artifact key that lists the other keys. The key set already covers it."""


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
    def same(self) -> bool:
        """Whether the two outputs are the same."""
        return not (self.only_in_reference or self.only_in_candidate or self.different)


def compare_artifacts(reference: Path, candidate: Path) -> Comparison:
    """Compare the keys in two artifact files. The file names can differ.

    Both artifacts are read with vivarium's ``Artifact``, as a simulation reads
    them. The key sets must be the same, and each key's data must be exactly
    equal. Differences in the HDF file itself, such as the order in which keys
    were written, are ignored. The ``metadata.keyspace`` key is not compared,
    because the key set already covers it.

    Raises
    ------
    ComparisonError
        If either path is not an artifact file with keys, or a key cannot be read.
    """
    for side, path in (("reference", reference), ("candidate", candidate)):
        if not path.is_file():
            raise ComparisonError(f"{side} artifact is not a file: {path}")
    reference_artifact = _open_artifact(reference, "reference")
    candidate_artifact = _open_artifact(candidate, "candidate")
    reference_keys = set(reference_artifact.keys) - {KEYSPACE_KEY}
    candidate_keys = set(candidate_artifact.keys) - {KEYSPACE_KEY}
    comparison = Comparison()
    for key in sorted(reference_keys | candidate_keys):
        if key not in candidate_keys:
            comparison.only_in_reference.append(key)
        elif key not in reference_keys:
            comparison.only_in_candidate.append(key)
        elif not _equal(
            _load(reference_artifact, key, reference),
            _load(candidate_artifact, key, candidate),
        ):
            comparison.different.append(key)
    return comparison


def compare_results(reference: Path, candidate: Path) -> Comparison:
    """Compare the measure files in two psimulate results directories.

    Files are paired by their path relative to each directory. Each pair is read
    with ``pd.read_parquet`` and its default options, which is how results are
    read for analysis. The two frames must be exactly equal. Differences that
    ``pd.read_parquet`` does not show, such as compression, row groups, or
    string width, are ignored.

    Raises
    ------
    ComparisonError
        If either path is not a directory, has no measure files, or has a file
        that cannot be read.
    """
    reference_files = _find_files(reference, "reference", RESULTS_PATTERN)
    candidate_files = _find_files(candidate, "candidate", RESULTS_PATTERN)
    comparison = Comparison()
    for name in sorted(reference_files.keys() | candidate_files.keys()):
        if name not in candidate_files:
            comparison.only_in_reference.append(name)
        elif name not in reference_files:
            comparison.only_in_candidate.append(name)
        elif not _same_frame(reference_files[name], candidate_files[name]):
            comparison.different.append(name)
    return comparison


def compare_notebooks(reference: Path, candidate: Path) -> Comparison:
    """Compare the executed notebooks in two notebook directories, byte for byte.

    Notebooks are paired by their path relative to each directory.

    Raises
    ------
    ComparisonError
        If either path is not a directory, has no executed notebooks, or has a
        notebook that cannot be read.
    """
    reference_files = _find_files(reference, "reference", NOTEBOOKS_PATTERN)
    candidate_files = _find_files(candidate, "candidate", NOTEBOOKS_PATTERN)
    comparison = Comparison()
    for name in sorted(reference_files.keys() | candidate_files.keys()):
        if name not in candidate_files:
            comparison.only_in_reference.append(name)
        elif name not in reference_files:
            comparison.only_in_candidate.append(name)
        elif not _same_bytes(reference_files[name], candidate_files[name]):
            comparison.different.append(name)
    return comparison


def _find_files(path: Path, side: str, pattern: str) -> dict[str, Path]:
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


def _same_bytes(reference: Path, candidate: Path) -> bool:
    """Whether two files are equal byte for byte."""
    try:
        return filecmp.cmp(reference, candidate, shallow=False)
    except OSError as error:
        raise ComparisonError(f"cannot read {error.filename}: {error}") from error


def _same_frame(reference: Path, candidate: Path) -> bool:
    """Whether two parquet files read into exactly equal frames."""
    return _equal(_read_parquet(reference), _read_parquet(candidate))


def _equal(reference: Any, candidate: Any) -> bool:
    """Whether two values are exactly equal. Frames and series count NaN as equal."""
    if type(reference) is not type(candidate):
        return False
    try:
        if isinstance(reference, pd.DataFrame):
            pd.testing.assert_frame_equal(reference, candidate, check_exact=True)
        elif isinstance(reference, pd.Series):
            pd.testing.assert_series_equal(reference, candidate, check_exact=True)
        else:
            return bool(reference == candidate)
    except AssertionError:
        return False
    return True


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file as results are read for analysis."""
    try:
        return pd.read_parquet(path)
    # pyarrow raises ArrowInvalid, a ValueError, for a file that is not parquet.
    except (OSError, ValueError) as error:
        raise ComparisonError(f"cannot read {path}: {error}") from error


def _open_artifact(path: Path, side: str) -> "Artifact":
    """Open an artifact for reading. ``side`` is "reference" or "candidate"."""
    import tables  # type: ignore[import-untyped]
    from vivarium.artifact.artifact import Artifact, ArtifactException

    # Artifact writes a keyspace into an HDF file that has no keys. Refuse such a
    # file first, so the comparison never changes its inputs.
    # Artifact raises TypeError for an HDF file that pandas wrote, and tables
    # raises a RuntimeError for a file that is not HDF.
    errors = (OSError, RuntimeError, TypeError, ValueError, ArtifactException)
    try:
        with tables.open_file(str(path), "r") as file:
            if next(iter(file.walk_nodes("/", "Leaf")), None) is None:
                raise ComparisonError(f"{side} artifact has no keys: {path}")
        return Artifact(path)
    except errors as error:
        raise ComparisonError(f"cannot read {side} artifact {path}: {error}") from error


def _load(artifact: "Artifact", key: str, path: Path) -> Any:
    """Load one key and drop it from the artifact's cache, so memory stays low."""
    try:
        return artifact.load(key)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        raise ComparisonError(f"cannot read key {key} in {path}: {error}") from error
    finally:
        artifact.clear_cache()
