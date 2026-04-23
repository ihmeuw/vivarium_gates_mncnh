"""Generate neonatal sepsis effect data (direct and indirect RRs) for scenario draws.

Usage:
    python generate_nn_sepsis_effects.py [--skip-existing] [--num-steps N]
                                         [--locations LOC [LOC ...]] [draw_numbers...]

Options:
    --skip-existing           Skip draws that already have output files.
                              Default behavior is to regenerate and overwrite all outputs.
    --num-steps N             Run only N simulation time steps per location and exit.
                              Use this to test parallelization overhead without running
                              the full computation. No output files are written.
    --locations LOC [LOC ..]  Run only the specified locations (case-insensitive).
                              Defaults to all locations in metadata.LOCATIONS.

If no draw numbers are provided, generates all draws from the
SCENARIO_DRAWS list defined in vivarium_gates_mncnh.constants.metadata.
"""

import sys
import time
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

from vivarium_gates_mncnh.constants.metadata import LOCATIONS as _LOCATIONS
from vivarium_gates_mncnh.constants.metadata import SCENARIO_DRAWS

ALL_LOCATIONS = [loc.lower() for loc in _LOCATIONS]


def _run_test_steps(draw, num_steps, locations=None):
    """Start real simulations but only run a limited number of time steps.

    Use this to verify parallelization overhead (imports, artifact loading,
    simulation setup) without waiting for the full RR computation.
    """
    from vivarium_gates_mncnh.data.sim_utils import initialize_simulation

    if locations is None:
        locations = ALL_LOCATIONS
    for location in locations:
        t0 = time.time()
        sim = initialize_simulation(location, draw, population_size=200_000)
        for step in range(num_steps):
            sim.step()

        elapsed = time.time() - t0
        print(f"[draw {draw}] {location}: {num_steps} step(s) in {elapsed:.0f}s")
        sys.stdout.flush()


def run_single_draw(draw, num_steps=None, locations=None):
    """Run generation for a single draw in this process."""
    t0 = time.time()
    print(f"[draw {draw}] Starting...")
    sys.stdout.flush()

    try:
        if num_steps is not None:
            _run_test_steps(draw, num_steps, locations=locations)
            elapsed = time.time() - t0
            print(f"[draw {draw}] Test complete in {elapsed:.0f}s")
        else:
            from hgb_nn_sepsis_effect_generation import calculate_direct_effect

            results_dir = str(_DIR)
            direct, _ = calculate_direct_effect(results_dir, draw, locations=locations)
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

    # Parse --locations LOC [LOC ...] flag
    locations = None
    if "--locations" in args:
        idx = args.index("--locations")
        locs = []
        for val in args[idx + 1 :]:
            if val.startswith("--") or val.isdigit():
                break
            locs.append(val.lower())
        if not locs:
            print("Error: --locations requires at least one location.", file=sys.stderr)
            sys.exit(1)
        unknown = set(locs) - set(ALL_LOCATIONS)
        if unknown:
            print(
                f"Error: unknown location(s): {', '.join(sorted(unknown))}. "
                f"Valid: {', '.join(ALL_LOCATIONS)}",
                file=sys.stderr,
            )
            sys.exit(1)
        locations = locs
        del args[idx : idx + 1 + len(locs)]

    draws = [int(d) for d in args] if args else list(SCENARIO_DRAWS)

    results_dir = _DIR
    (results_dir / "direct_sepsis_effects").mkdir(parents=True, exist_ok=True)
    (results_dir / "indirect_sepsis_effects").mkdir(parents=True, exist_ok=True)

    existing = set()
    if skip_existing and num_steps is None:
        existing = {
            int(f.stem.replace("draw_", ""))
            for f in (results_dir / "direct_sepsis_effects").iterdir()
            if f.name.startswith("draw_") and f.name.endswith(".csv")
        }
    draws_to_run = [d for d in draws if d not in existing]

    if not draws_to_run:
        print("All requested draws already exist. Nothing to do.")
        return

    mode = f"test ({num_steps} step(s))" if num_steps is not None else "full"
    print(f"Mode:            {mode}")
    print(f"Locations:       {', '.join(locations or ALL_LOCATIONS)}")
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
        run_single_draw(draw, num_steps=num_steps, locations=locations)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
