"""Unit tests for ``vivarium_gates_mncnh.tools.run_main_sim``.

Covers the command-line surface and the guard rails around it. The
end-to-end test at the bottom exercises ``run_sim`` itself with psimulate and
the git helpers faked out, so the wiring is checked without a cluster.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest
import yaml

from vivarium_gates_mncnh.constants.metadata import LOCATIONS
from vivarium_gates_mncnh.tools import run_main_sim

MODEL_NUMBER = "29.0.2"

#: The flags every invocation needs, so each test only has to say what it is
#: actually about.
BASE_ARGS = [
    "--queue",
    "all.q",
    "--project",
    "proj_simscience_prod",
    "--model-number",
    MODEL_NUMBER,
]


def parse(*extra: str) -> argparse.Namespace:
    """Parse ``BASE_ARGS`` plus *extra* with the script's own parser."""
    return run_main_sim._build_parser().parse_args([*BASE_ARGS, *extra])


def joined(command: List[Any]) -> str:
    """One searchable string for a recorded argv."""
    return " ".join(str(token) for token in command)


@pytest.fixture()
def sim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace everything ``run_sim`` reaches outside the process.

    psimulate, the git helpers and the two files the script rewrites are all
    faked, so the test exercises the wiring and nothing else. Notably
    ``PATHS_MODULE`` and ``MODEL_SPEC_PATH`` are redirected at copies under
    ``tmp_path``: ``run_sim`` edits both, and the suite must not edit the
    checkout it is running from.
    """
    state = SimpleNamespace(
        events=[],
        commands=[],
        tags=[],
        tree_checks=0,
        # Locations whose psimulate run should report an incomplete result.
        unfinished=set(),
        output_root=tmp_path / "results",
        # The directory ``--artifact-dir`` points at, and the one the model
        # spec pins. Keeping them distinct is what lets a test tell "the
        # override was honoured" apart from "the default was used".
        artifact_dir=tmp_path / "artifacts",
        spec_artifact_dir=tmp_path / "spec_artifacts",
    )

    # ``run_sim`` refuses to launch anything if a requested location's artifact
    # is missing, so both candidate directories get one per location. The
    # filename is lowercased, matching the path pinned in the model spec.
    for directory in (state.artifact_dir, state.spec_artifact_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for location in LOCATIONS:
            (directory / f"{location.lower()}.hdf").touch()

    def fake_require_environment(env_type: str) -> None:
        state.events.append(("require_environment", env_type))

    def fake_run_command(
        cmd: List[str],
        description: str,
        env: Any = None,
        auto_confirm: bool = False,
        capture_full_output: bool = False,
    ) -> str:
        state.events.append(("run_command", list(cmd)))
        state.commands.append(list(cmd))
        # Echo the argv back so the completion check can tell the per-location
        # runs apart without knowing how the command was assembled.
        return (
            "Running jobs...\n"
            "(2 of 2 total jobs completed successfully overall)\n"
            f"Results written to: {state.output_root}\n"
            f"ARGV {joined(cmd)}\n"
        )

    def fake_check_psimulate_finished(output: str) -> bool:
        return not any(location.lower() in output.lower() for location in state.unfinished)

    def fake_extract_results_dir(output: str) -> Optional[str]:
        return str(state.output_root)

    def fake_check_clean_tree(repo: Optional[Path] = None) -> None:
        state.tree_checks += 1
        state.events.append(("check_clean_tree", repo))

    def fake_create_and_push_tag(model_number: str) -> None:
        state.tags.append(model_number)
        state.events.append(("create_and_push_tag", model_number))

    monkeypatch.setattr(run_main_sim, "require_environment", fake_require_environment)
    monkeypatch.setattr(run_main_sim, "run_command", fake_run_command)
    monkeypatch.setattr(
        run_main_sim, "check_psimulate_finished", fake_check_psimulate_finished
    )
    monkeypatch.setattr(run_main_sim, "extract_results_dir", fake_extract_results_dir)
    monkeypatch.setattr(run_main_sim, "check_clean_tree", fake_check_clean_tree)
    monkeypatch.setattr(run_main_sim, "create_and_push_tag", fake_create_and_push_tag)

    paths_module = tmp_path / "paths.py"
    paths_module.write_text('MODEL_RESULTS_DIR = "model0.0"\n')
    monkeypatch.setattr(run_main_sim, "PATHS_MODULE", paths_module)

    model_spec = tmp_path / "model_spec.yaml"
    model_spec.write_text(
        yaml.safe_dump(
            {
                "configuration": {
                    "input_data": {
                        "artifact_path": f"{state.spec_artifact_dir}/ethiopia.hdf",
                        "input_draw_number": 60,
                    }
                }
            }
        )
    )
    monkeypatch.setattr(run_main_sim, "MODEL_SPEC_DIR", tmp_path)
    monkeypatch.setattr(run_main_sim, "MODEL_SPEC_PATH", model_spec)

    state.paths_module = paths_module
    state.model_spec = model_spec
    return state


def run(sim: SimpleNamespace, **overrides: Any) -> None:
    """Call ``run_sim`` with the harness's defaults, overridden per test."""
    kwargs: dict = {
        "queue": "all.q",
        "project": "proj_simscience_prod",
        "model_number": MODEL_NUMBER,
        "locations": ["Ethiopia"],
        "artifact_dir": sim.artifact_dir,
        "output_root": sim.output_root,
        "tag": False,
        "check_tree": False,
    }
    kwargs.update(overrides)
    run_main_sim.run_sim(**kwargs)


def launches(sim: SimpleNamespace) -> List[str]:
    """The recorded psimulate invocations, one searchable string each."""
    return [joined(cmd) for cmd in sim.commands if "psimulate" in joined(cmd)]


class TestCliArguments:
    """The flag surface the framework-check pipeline drives the script with."""

    def test_locations_defaults_to_every_location(self, sim: SimpleNamespace) -> None:
        """Omitting ``--locations`` runs the full set, preserving existing behaviour."""
        default = parse().locations
        # Either spelling of "all of them" is fine; what matters is the run.
        assert not default or list(default) == list(LOCATIONS)

        run(sim, locations=default)

        assert len(launches(sim)) == len(LOCATIONS)
        for location in LOCATIONS:
            assert any(location.lower() in launch.lower() for launch in launches(sim))

    def test_locations_accepts_a_repeated_subset(self) -> None:
        """``--locations`` can be given more than once to run fewer locations."""
        args = parse("--locations", "Ethiopia", "--locations", "Nigeria")

        assert list(args.locations) == ["Ethiopia", "Nigeria"]

    def test_locations_rejects_an_unknown_location(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """A location not in LOCATIONS is rejected at parse time, not after the tag is pushed."""
        with pytest.raises(SystemExit):
            parse("--locations", "Atlantis")

        captured = capsys.readouterr()
        assert "Atlantis" in captured.out + captured.err

    def test_artifact_dir_overrides_the_model_spec_path(self, sim: SimpleNamespace) -> None:
        """``--artifact-dir`` replaces the production artifact directory pinned in the model spec."""
        args = parse("--artifact-dir", str(sim.artifact_dir))
        assert Path(args.artifact_dir) == sim.artifact_dir

        run(sim, locations=["Ethiopia"], artifact_dir=sim.artifact_dir)

        (launch,) = launches(sim)
        assert str(sim.artifact_dir) in launch
        assert str(sim.spec_artifact_dir) not in launch

    def test_artifact_dir_defaults_to_the_model_spec_directory(
        self, sim: SimpleNamespace
    ) -> None:
        """With no --artifact-dir, artifacts resolve from the parent of the model spec's artifact_path."""
        assert parse().artifact_dir is None

        run(sim, locations=["Ethiopia"], artifact_dir=None)

        (launch,) = launches(sim)
        assert str(sim.spec_artifact_dir) in launch
        # The override directory exists and is populated, so this could only
        # match by the default having been resolved from the spec.
        assert str(sim.artifact_dir) not in launch

    def test_output_root_overrides_the_results_mount(self, sim: SimpleNamespace) -> None:
        """``--output-root`` replaces the team results mount."""
        args = parse("--output-root", str(sim.output_root))
        assert Path(args.output_root) == sim.output_root

        run(sim, output_root=sim.output_root)

        (launch,) = launches(sim)
        assert str(run_main_sim.RESULTS_ROOT) not in launch


class TestSkipTagGuardRail:
    """An untagged run must not be able to write into production results."""

    def rejects(self, capsys: pytest.CaptureFixture, *extra: str) -> str:
        """Assert the flag combination is refused, and return the complaint."""
        args = parse(*extra)
        with pytest.raises(SystemExit) as excinfo:
            run_main_sim._validate_cli_args(args)
        captured = capsys.readouterr()
        # ``parser.error`` writes to stderr and exits; a bare SystemExit
        # carries the message itself. Either is a correct implementation.
        return f"{captured.out}{captured.err}{excinfo.value}"

    def test_skip_tag_without_output_root_is_rejected(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """``--skip-tag`` alone would write untagged results to the production root."""
        assert "--output-root" in self.rejects(capsys, "--skip-tag")

    def test_skip_tag_with_the_production_output_root_is_rejected(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """Passing the production root explicitly is the same hazard and is caught too."""
        message = self.rejects(
            capsys, "--skip-tag", "--output-root", str(run_main_sim.RESULTS_ROOT)
        )

        assert "--output-root" in message

    def test_skip_tag_with_a_scratch_output_root_is_accepted(self, tmp_path: Path) -> None:
        """The intended framework-check invocation is allowed."""
        args = parse("--skip-tag", "--output-root", str(tmp_path / "scratch"))

        assert run_main_sim._validate_cli_args(args) is None

    def test_a_tagging_run_does_not_require_an_output_root(self) -> None:
        """The normal production path is unaffected by the new rule."""
        assert run_main_sim._validate_cli_args(parse()) is None


class TestRunSim:
    """End-to-end wiring, with psimulate and git faked."""

    def test_launches_psimulate_once_per_requested_location(
        self, sim: SimpleNamespace
    ) -> None:
        """Each requested location gets its own psimulate invocation with its own artifact."""
        run(sim, locations=["Ethiopia", "Nigeria"], artifact_dir=sim.artifact_dir)

        assert len(launches(sim)) == 2
        for location in ("Ethiopia", "Nigeria"):
            matching = [
                launch for launch in launches(sim) if location.lower() in launch.lower()
            ]
            assert len(matching) == 1, f"expected exactly one run for {location}"
        # Two runs over the same artifact would be the interesting failure.
        assert launches(sim)[0] != launches(sim)[1]

    def test_checks_the_environment_before_launching_anything(
        self, sim: SimpleNamespace
    ) -> None:
        """``require_executable`` runs before the first psimulate call, so a bad environment
        costs seconds rather than a partial run."""
        run(sim, locations=["Ethiopia", "Nigeria"])

        kinds = [kind for kind, _payload in sim.events]
        assert "require_environment" in kinds
        assert kinds.index("require_environment") < kinds.index("run_command")

    def test_skips_tagging_when_tag_is_false(self, sim: SimpleNamespace) -> None:
        """``tag=False`` creates and pushes no git tag."""
        run(sim, tag=False)
        assert sim.tags == []

        # The same run with tagging on does tag, so the assertion above is not
        # passing for want of a working spy.
        run(sim, tag=True)
        assert sim.tags == [MODEL_NUMBER]

    def test_skips_the_clean_tree_guard_when_check_tree_is_false(
        self, sim: SimpleNamespace
    ) -> None:
        """``check_tree=False`` bypasses the guard for a deliberately dirty check run."""
        run(sim, check_tree=False)
        assert sim.tree_checks == 0

        run(sim, check_tree=True)
        assert sim.tree_checks == 1

    def test_raises_when_a_location_does_not_finish_every_job(
        self, sim: SimpleNamespace
    ) -> None:
        """A partial psimulate result fails the run rather than being reported as success."""
        sim.unfinished = {"Nigeria"}

        with pytest.raises(RuntimeError) as excinfo:
            run(sim, locations=["Ethiopia", "Nigeria"])

        # Case-insensitive: naming the location that failed is the actionable
        # part, whether the message says "Nigeria" or "nigeria.hdf".
        assert "nigeria" in str(excinfo.value).lower()

    def test_raises_when_a_requested_artifact_is_missing(self, sim: SimpleNamespace) -> None:
        """A requested location with no artifact aborts before psimulate is launched."""
        missing = sim.artifact_dir / "nigeria.hdf"
        missing.unlink()

        with pytest.raises(RuntimeError) as excinfo:
            run(sim, locations=["Ethiopia", "Nigeria"])

        assert str(missing) in str(excinfo.value)
        # Only the absent one is reported; Ethiopia's artifact is there.
        assert "ethiopia.hdf" not in str(excinfo.value)
        # Ethiopia's artifact being present is also what makes the next
        # assertion bite: a gate that fired per location rather than up front
        # would already have launched it. That no command was recorded at all
        # is the property worth having.
        assert sim.commands == []

    def test_writes_results_under_the_requested_output_root(
        self, sim: SimpleNamespace
    ) -> None:
        """Results land in ``<output_root>/model{model_number}``."""
        run(sim, output_root=sim.output_root)

        (launch,) = launches(sim)
        assert str(sim.output_root / f"model{MODEL_NUMBER}") in launch
