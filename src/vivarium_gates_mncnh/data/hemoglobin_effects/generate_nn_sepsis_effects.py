"""Generate neonatal sepsis effect data (direct and indirect RRs) for scenario draws.

Usage:
    python generate_nn_sepsis_effects.py [--skip-existing] [--num-steps N] [draw_numbers...]

Options:
    --skip-existing    Skip draws that already have output files.
                       Default behavior is to regenerate and overwrite all outputs.
    --num-steps N      Run only N simulation time steps per location and exit.
                       Use this to test parallelization overhead without running
                       the full computation. No output files are written.

If no draw numbers are provided, generates all draws from the
SCENARIO_DRAWS list defined in vivarium_gates_mncnh.constants.metadata.
"""

import os
import sys
import time
from pathlib import Path

# Must run from the hemoglobin_effects directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from vivarium_gates_mncnh.constants.metadata import LOCATIONS, SCENARIO_DRAWS


def _run_test_steps(draw, num_steps):
    """Start real simulations but only run a limited number of time steps.

    Use this to verify parallelization overhead (imports, artifact loading,
    simulation setup) without waiting for the full RR computation.
    """
    from vivarium import InteractiveContext
    from vivarium.framework.configuration import build_model_specification

    spec_path = Path(os.getcwd()) / ".." / ".." / "model_specifications" / "model_spec.yaml"
    # Build once just to discover the artifact base path
    base_spec = build_model_specification(spec_path)
    artifact_base = base_spec.configuration.input_data.artifact_path.rsplit("/", 1)[0] + "/"

    for location in [loc.lower() for loc in LOCATIONS]:
        t0 = time.time()
        # Build a fresh spec each time — InteractiveContext mutates it
        spec = build_model_specification(spec_path)
        del spec.configuration.observers
        spec.configuration.input_data.artifact_path = artifact_base + location + ".hdf"
        spec.configuration.input_data.input_draw_number = draw
        spec.configuration.population.population_size = 200_000

        sim = InteractiveContext(spec)
        for step in range(num_steps):
            sim.step()

        elapsed = time.time() - t0
        print(f"[draw {draw}] {location}: {num_steps} step(s) in {elapsed:.0f}s")
        sys.stdout.flush()


def run_single_draw(draw, num_steps=None):
    """Run generation for a single draw in this process."""
    t0 = time.time()
    print(f"[draw {draw}] Starting...")
    sys.stdout.flush()

    try:
        if num_steps is not None:
            _run_test_steps(draw, num_steps)
            elapsed = time.time() - t0
            print(f"[draw {draw}] Test complete in {elapsed:.0f}s")
        else:
            from hgb_nn_sepsis_effect_generation import calculate_direct_effect

            results_dir = os.getcwd()
            direct, _ = calculate_direct_effect(results_dir, draw)
            elapsed = time.time() - t0
            print(f"[draw {draw}] Complete in {elapsed:.0f}s  shape={direct.shape}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[draw {draw}] FAILED after {elapsed:.0f}s: {e}")
    sys.stdout.flush()


def main():
    args = sys.argv[1:]

    # Parse --skip-existing flag
    skip_existing = False
    if "--skip-existing" in args:
        skip_existing = True
        args.remove("--skip-existing")

    # Parse --num-steps N flag
    num_steps = None
    if "--num-steps" in args:
        idx = args.index("--num-steps")
        num_steps = int(args[idx + 1])
        del args[idx : idx + 2]

    draws = [int(d) for d in args] if args else list(SCENARIO_DRAWS)

    results_dir = os.getcwd()
    os.makedirs(f"{results_dir}/direct_sepsis_effects", exist_ok=True)
    os.makedirs(f"{results_dir}/indirect_sepsis_effects", exist_ok=True)

    existing = set()
    if skip_existing and num_steps is None:
        existing = {
            int(f.replace("draw_", "").replace(".csv", ""))
            for f in os.listdir(f"{results_dir}/direct_sepsis_effects")
            if f.startswith("draw_") and f.endswith(".csv")
        }
    draws_to_run = [d for d in draws if d not in existing]

    if not draws_to_run:
        print("All requested draws already exist. Nothing to do.")
        return

    mode = f"test ({num_steps} step(s))" if num_steps is not None else "full"
    print(f"Mode:            {mode}")
    print(f"Draws requested: {draws}")
    print(f"Skip existing:   {skip_existing}")
    if existing:
        print(f"Already exist:   {sorted(existing & set(draws))}")
    print(f"To generate:     {draws_to_run}")
    print(f"{'='*60}")
    sys.stdout.flush()

    for i, draw in enumerate(draws_to_run, 1):
        print(f"\n[{i}/{len(draws_to_run)}] Starting draw {draw}...")
        sys.stdout.flush()
        run_single_draw(draw, num_steps=num_steps)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
