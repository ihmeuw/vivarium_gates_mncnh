#!/usr/bin/env python
"""Parallel runner for draw-based data generation scripts.

Spawns one subprocess per draw, up to --workers at a time.
Each subprocess runs: python <script> [script_args...] <draw>

Usage:
    python parallel_runner.py [--workers N] [--draws D ...] [--log-dir DIR] -- <script> [script_args...]

Examples:
    # Test parallelization: 1 sim step, 4 draws, 2 workers
    python parallel_runner.py --workers 2 --draws 0 1 2 3 -- generate_nn_sepsis_effects.py --num-steps 1

    # Full generation: 5 draws, 4 workers
    python parallel_runner.py --workers 4 --draws 0 1 2 3 4 -- generate_nn_sepsis_effects.py

    # All scenario draws, 4 workers, skip existing, custom log dir
    python parallel_runner.py --workers 4 --log-dir ~/my_logs -- generate_nn_sepsis_effects.py --skip-existing
"""

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_one_draw(script_cmd, draw, log_dir):
    """Run a single draw as a subprocess, capturing output to a log file.

    Parameters
    ----------
    script_cmd
        Base command list (e.g. ["generate_nn_sepsis_effects.py", "--num-steps", "1"]).
    draw
        Draw number to append to the command.
    log_dir
        Directory for per-draw log files.

    Returns
    -------
    tuple
        (draw, returncode, elapsed_seconds, log_path)
    """
    log_path = log_dir / f"draw_{draw}.log"
    cmd = [sys.executable] + script_cmd + [str(draw)]
    t0 = time.time()
    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent),
        )
    elapsed = time.time() - t0
    return draw, proc.returncode, elapsed, log_path


def parse_args():
    """Parse arguments, splitting on '--' into runner args and script command."""
    argv = sys.argv[1:]
    if "--" not in argv:
        print("Error: use '--' to separate runner args from the script command.")
        print("  python parallel_runner.py --workers 2 --draws 0 1 -- script.py [args]")
        sys.exit(1)

    sep = argv.index("--")
    runner_args = argv[:sep]
    script_cmd = argv[sep + 1 :]

    if not script_cmd:
        print("Error: no script specified after '--'.")
        sys.exit(1)

    # Parse runner args manually for simplicity
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    draws = None
    log_dir = None

    i = 0
    while i < len(runner_args):
        if runner_args[i] == "--workers":
            workers = int(runner_args[i + 1])
            i += 2
        elif runner_args[i] == "--draws":
            draws = []
            i += 1
            while i < len(runner_args) and not runner_args[i].startswith("--"):
                draws.append(int(runner_args[i]))
                i += 1
        elif runner_args[i] == "--log-dir":
            log_dir = Path(runner_args[i + 1]).expanduser()
            i += 2
        else:
            print(f"Error: unknown runner argument '{runner_args[i]}'")
            sys.exit(1)

    return workers, draws, log_dir, script_cmd


def main():
    workers, draws, log_dir, script_cmd = parse_args()

    # If no draws specified, let the generation script decide (import metadata)
    if draws is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from vivarium_gates_mncnh.constants.metadata import SCENARIO_DRAWS

        draws = list(SCENARIO_DRAWS)

    if log_dir is None:
        log_dir = Path.home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Script:    {' '.join(script_cmd)}")
    print(f"Workers:   {workers}")
    print(f"Draws:     {draws} ({len(draws)} total)")
    print(f"Logs:      {log_dir}/")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0_total = time.time()
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one_draw, script_cmd, draw, log_dir): draw for draw in draws
        }
        for future in as_completed(futures):
            draw, returncode, elapsed, log_path = future.result()
            completed += 1
            status = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
            if returncode != 0:
                failed += 1
            print(
                f"  [{completed}/{len(draws)}] draw {draw}: {status} in {elapsed:.0f}s "
                f"-> {log_path.name}"
            )
            sys.stdout.flush()

    total_elapsed = time.time() - t0_total
    print(f"{'='*60}")
    print(f"Finished: {completed - failed}/{len(draws)} succeeded in {total_elapsed:.0f}s")
    if failed:
        print(f"Failures: {failed} (check logs)")
        sys.exit(1)


if __name__ == "__main__":
    main()
