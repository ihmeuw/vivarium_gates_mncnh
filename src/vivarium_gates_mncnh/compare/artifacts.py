"""Compare two artifact files key by key."""

from pathlib import Path

from vivarium.artifact import Artifact, ArtifactException

from vivarium_gates_mncnh.compare.base import (
    READ_ERRORS,
    Comparison,
    ComparisonError,
    compare_items,
    equal_values,
    reraise_read_errors,
)

KEYSPACE_KEY = "metadata.keyspace"
"""The artifact key that lists the other keys. The key set already covers it."""


def compare_artifacts(reference: Path, candidate: Path) -> Comparison:
    """Compare the keys in two artifact files. The file names can differ.

    Both artifacts are read with vivarium's ``Artifact``, as a simulation reads
    them. The key sets must be equal, and each key's data must be exactly
    equal. Differences in the HDF file itself, such as the order in which keys
    were written, are ignored. The ``metadata.keyspace`` key is not compared,
    because the key set already covers it.

    Raises
    ------
    ComparisonError
        If either path is not an artifact file with keys, or a key cannot be read.
    """
    reference_artifact = _open_artifact(reference, "reference")
    candidate_artifact = _open_artifact(candidate, "candidate")
    return compare_items(
        set(reference_artifact.keys) - {KEYSPACE_KEY},
        set(candidate_artifact.keys) - {KEYSPACE_KEY},
        lambda key: equal_values(reference_artifact.load(key), candidate_artifact.load(key)),
    )


def _open_artifact(path: Path, side: str) -> Artifact:
    """Open an artifact for reading. ``side`` is "reference" or "candidate".

    An artifact with no keys besides ``metadata.keyspace`` is refused. ``Artifact``
    writes a keyspace into a file that has none, so such a file is changed
    before it is refused.
    """
    # Artifact creates a file at a path that has none.
    if not path.is_file():
        raise ComparisonError(f"{side} artifact is not a file: {path}")
    with reraise_read_errors(f"{side} artifact {path}", (*READ_ERRORS, ArtifactException)):
        artifact = Artifact(path)
    if not set(artifact.keys) - {KEYSPACE_KEY}:
        raise ComparisonError(f"{side} artifact has no keys: {path}")
    return artifact
