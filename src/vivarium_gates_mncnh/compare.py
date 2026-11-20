"""
Compare model outputs from two environments.

These functions answer one question after an environment update: did the
artifacts, the simulation results, or the executed notebooks change? Each
returns an ``OutputDiff``, which later CI stages can act on or save.

Two outputs are the same when they have the same set of files and each pair of
files is equal byte for byte. Which files count depends on the output type:

- artifacts: the two artifact files
- results: ``<measure>/*.parquet`` in a psimulate ``results`` directory
- notebooks: ``**/executed/*.ipynb`` in a notebooks directory

This module imports nothing from the model, so a broken candidate environment
cannot make the comparison itself fail.
"""

import filecmp
from dataclasses import dataclass, field
from pathlib import Path

RESULTS_PATTERN = "*/*.parquet"
NOTEBOOKS_PATTERN = "**/executed/*.ipynb"


class ComparisonError(Exception):
    """The inputs cannot be compared.

    This is separate from a difference: a caller can treat it as a broken
    comparison, not a failed one.
    """


@dataclass
class OutputDiff:
    """The result of comparing a reference output with a candidate output.

    Each list holds relative file names or keys, in sorted order.
    """

    only_in_reference: list[str] = field(default_factory=list)
    only_in_candidate: list[str] = field(default_factory=list)
    different: list[str] = field(default_factory=list)
    """Files or keys present on both sides whose bytes differ."""

    @property
    def same(self) -> bool:
        """Whether the two outputs are the same."""
        return not (self.only_in_reference or self.only_in_candidate or self.different)


def compare_artifacts(reference: Path, candidate: Path) -> OutputDiff:
    """Compare two artifact files byte for byte. The file names can differ.

    Raises
    ------
    ComparisonError
        If either path is not a file or cannot be read.
    """
    for side, path in (("reference", reference), ("candidate", candidate)):
        if not path.is_file():
            raise ComparisonError(f"{side} artifact is not a file: {path}")
    if _same_bytes(reference, candidate):
        return OutputDiff()
    return OutputDiff(different=["artifact"])


def compare_results(reference: Path, candidate: Path) -> OutputDiff:
    """Compare the measure files in two psimulate results directories, byte for byte.

    Files are paired by their path relative to each directory.

    Raises
    ------
    ComparisonError
        If either path is not a directory, has no measure files, or has a file
        that cannot be read.
    """
    reference_files = _find_files(reference, "reference", RESULTS_PATTERN)
    candidate_files = _find_files(candidate, "candidate", RESULTS_PATTERN)
    diff = OutputDiff()
    for name in sorted(reference_files.keys() | candidate_files.keys()):
        if name not in candidate_files:
            diff.only_in_reference.append(name)
        elif name not in reference_files:
            diff.only_in_candidate.append(name)
        elif not _same_bytes(reference_files[name], candidate_files[name]):
            diff.different.append(name)
    return diff


def compare_notebooks(reference: Path, candidate: Path) -> OutputDiff:
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
    diff = OutputDiff()
    for name in sorted(reference_files.keys() | candidate_files.keys()):
        if name not in candidate_files:
            diff.only_in_reference.append(name)
        elif name not in reference_files:
            diff.only_in_candidate.append(name)
        elif not _same_bytes(reference_files[name], candidate_files[name]):
            diff.different.append(name)
    return diff


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
