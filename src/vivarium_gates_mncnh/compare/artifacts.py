"""Compare two artifact files key by key."""

from pathlib import Path

from vivarium.artifact import Artifact, ArtifactException

from vivarium_gates_mncnh.compare.utilities import (
    READ_ERRORS,
    Comparison,
    ComparisonError,
    compare_items,
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
    return compare_items(
        reference=_open_artifact(reference, "reference"),
        candidate=_open_artifact(candidate, "candidate"),
        load=lambda artifact, key: artifact.load(key),
    )


def _open_artifact(path: Path, side: str) -> dict[str, Artifact]:
    """Open an artifact and map each key to it. ``side`` is "reference" or "candidate".

    ``metadata.keyspace`` is left out of the mapping. The file is not changed.
    An artifact with no other keys is refused. ``Artifact`` writes a keyspace
    into a file that has none, so such a file is changed before it is refused.
    """
    # Artifact creates a file at a path that has none.
    if not path.is_file():
        raise ComparisonError(f"{side} artifact is not a file: {path}")
    with reraise_read_errors(f"{side} artifact {path}", (*READ_ERRORS, ArtifactException)):
        artifact = Artifact(path)
    keys = {key: artifact for key in artifact.keys if key != KEYSPACE_KEY}
    if not keys:
        raise ComparisonError(f"{side} artifact has no keys: {path}")
    return keys
