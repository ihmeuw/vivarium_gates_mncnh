"""Unit tests for the decomposed LBWSG PAF workflow script.

The script no longer dispatches commands into another environment; it runs one
step wholly inside whatever environment the caller activated. These tests pin
that contract: every step declares its flavour, asserts it before doing any
work, and runs the right commands.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest

# The script lives under data/, not in an importable package path, so load it
# by location the same way the workflow runner invokes it by path.
_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src/vivarium_gates_mncnh/data/lbwsg_paf/code/run_paf_sim.py"
)
_spec = importlib.util.spec_from_file_location("run_paf_sim", _SCRIPT)
run_paf_sim = importlib.util.module_from_spec(_spec)
sys.modules["run_paf_sim"] = run_paf_sim
_spec.loader.exec_module(run_paf_sim)


@pytest.fixture()
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace everything the script reaches outside the process."""
    state = SimpleNamespace(events=[], commands=[], moved=[], required=[])

    def fake_require_environment(env_type: str) -> None:
        state.events.append(("require_environment", env_type))
        state.required.append(env_type)

    def fake_run_command(
        cmd: List[str],
        description: str,
        auto_confirm: bool = False,
        capture_full_output: bool = False,
    ) -> Optional[str]:
        state.events.append(("run_command", list(cmd)))
        state.commands.append(list(cmd))
        return "(2 of 2 total jobs completed successfully overall)\nResults written to: /r\n"

    def fake_move_results(source: str, dest: str, what: str) -> None:
        state.moved.append((source, dest))

    monkeypatch.setattr(run_paf_sim, "require_environment", fake_require_environment)
    monkeypatch.setattr(run_paf_sim, "run_command", fake_run_command)
    monkeypatch.setattr(run_paf_sim, "move_results", fake_move_results)
    monkeypatch.setattr(run_paf_sim, "check_psimulate_finished", lambda out: True)
    monkeypatch.setattr(run_paf_sim, "extract_results_dir", lambda out: "/r")
    monkeypatch.setattr(run_paf_sim, "warn_if_dirty", lambda paths, who: False)
    monkeypatch.setattr(run_paf_sim, "CLUSTER_DATA_DIR", str(tmp_path / "scratch"))

    state.artifact_dir = tmp_path / "artifacts"
    state.artifact_dir.mkdir()
    return state


def run(step: str, state: SimpleNamespace, location: str = "Ethiopia") -> None:
    """Invoke the script's entry point for one step."""
    sys.argv = [
        "run_paf_sim.py",
        "--step",
        step,
        "-a",
        "test",
        "-l",
        location,
        "-o",
        str(state.artifact_dir),
    ]
    run_paf_sim.main()


def joined(cmd: List[Any]) -> str:
    return " ".join(str(c) for c in cmd)


class TestStepEnvironmentContract:
    """Each step names the environment it needs, and checks before it works."""

    def test_every_step_declares_an_environment(self) -> None:
        """No step may be left without a flavour."""
        assert set(run_paf_sim.STEP_ENVIRONMENTS.values()) == {"artifact", "simulation"}
        assert len(run_paf_sim.STEP_ENVIRONMENTS) == 5

    def test_psimulate_steps_want_the_simulation_environment(self) -> None:
        """The two PAF steps are the simulation-environment ones."""
        sim = {s for s, e in run_paf_sim.STEP_ENVIRONMENTS.items() if e == "simulation"}
        assert sim == {"enn-paf", "lnn-paf"}

    def test_artifact_steps_want_the_artifact_environment(self) -> None:
        """The three make_artifacts steps are the artifact-environment ones."""
        art = {s for s, e in run_paf_sim.STEP_ENVIRONMENTS.items() if e == "artifact"}
        assert art == {"initial-artifact", "enn-artifact", "final-artifact"}

    @pytest.mark.parametrize("step", sorted(run_paf_sim.STEP_ENVIRONMENTS))
    def test_the_environment_is_checked_before_any_command_runs(
        self, step: str, harness: SimpleNamespace
    ) -> None:
        """A step in the wrong environment must cost seconds, not a whole job."""
        run(step, harness)

        assert harness.events, f"step {step} did nothing"
        assert harness.events[0][0] == "require_environment"
        assert harness.events[0][1] == run_paf_sim.STEP_ENVIRONMENTS[step]

    def test_an_unknown_step_is_rejected_at_parse_time(
        self, harness: SimpleNamespace
    ) -> None:
        """A typo in the workflow's step name fails immediately."""
        with pytest.raises(SystemExit):
            run("enn-pafs", harness)


class TestStepBehaviour:
    """Each step runs the commands it is supposed to, and no others."""

    def test_initial_artifact_builds_the_artifact(self, harness: SimpleNamespace) -> None:
        """With no artifact present, the first step builds a full one."""
        run("initial-artifact", harness)

        (cmd,) = harness.commands
        assert cmd[0] == "make_artifacts"
        assert "-r" not in cmd, "the initial build is not measure-restricted"

    def test_initial_artifact_reuses_an_existing_artifact(
        self, harness: SimpleNamespace
    ) -> None:
        """An existing artifact is reused rather than silently rebuilt."""
        (harness.artifact_dir / "ethiopia.hdf").touch()

        run("initial-artifact", harness)

        assert harness.commands == []

    @pytest.mark.parametrize("step", ["enn-artifact", "final-artifact"])
    def test_later_artifact_steps_restrict_to_the_paf_measures(
        self, step: str, harness: SimpleNamespace
    ) -> None:
        """Rebuilding after a PAF run only regenerates the PAF measures."""
        run(step, harness)

        (cmd,) = harness.commands
        requested = [cmd[i + 1] for i, token in enumerate(cmd) if token == "-r"]
        assert requested == run_paf_sim.PAF_MEASURES

    def test_enn_paf_runs_psimulate_and_harvests_its_own_results(
        self, harness: SimpleNamespace
    ) -> None:
        """A PAF step collects its outputs itself, so no later step needs its scratch dir."""
        run("enn-paf", harness)

        (cmd,) = harness.commands
        assert cmd[0] == "psimulate"
        assert "lbwsg_paf_enn.yaml" in joined(cmd)
        assert len(harness.moved) == 1
        assert "paf_outputs" in harness.moved[0][1]

    def test_lnn_paf_harvests_both_output_sets(self, harness: SimpleNamespace) -> None:
        """The second PAF run produces PAFs and preterm prevalence."""
        run("lnn-paf", harness)

        (cmd,) = harness.commands
        assert "lbwsg_paf.yaml" in joined(cmd)
        destinations = " ".join(dest for _, dest in harness.moved)
        assert "paf_outputs" in destinations
        assert "preterm_prevalence_outputs" in destinations

    def test_a_failed_psimulate_aborts_the_step(
        self, harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial completion must not be reported as success."""
        monkeypatch.setattr(run_paf_sim, "check_psimulate_finished", lambda out: False)

        with pytest.raises(RuntimeError):
            run("enn-paf", harness)

        assert harness.moved == [], "nothing should be harvested from a failed run"

    def test_an_unparseable_results_directory_aborts_the_step(
        self, harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Success with no results directory is a failure, not a silent skip."""
        monkeypatch.setattr(run_paf_sim, "extract_results_dir", lambda out: None)

        with pytest.raises(RuntimeError):
            run("enn-paf", harness)

        assert harness.moved == []
