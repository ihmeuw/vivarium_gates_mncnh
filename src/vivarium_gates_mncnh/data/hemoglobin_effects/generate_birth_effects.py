"""Generate hemoglobin --> birth effects (LBWSG shifts) per draw -- STUB.

This is a placeholder. It does nothing today.

PURPOSE
-------
Drive the draw-parallelizable functions in `hgb_birth_effect_generation.py`
from the CLI so that `parallel_runner.py` can fan them out across draws,
mirroring the pattern used by `generate_nn_sepsis_effects.py`.

WORKFLOW FIT
------------
This step must run BEFORE `hemoglobin_effects_sepsis`. The sepsis script
reads `lbwsg_shifts/draw_<N>.csv` from disk (see
`hgb_nn_sepsis_effect_generation.load_lbwsg_shifts`); without those files
the sepsis step will fail. Outputs of this step are also consumed by
`make_artifacts` (see `data/loader.py:1038` and `data/loader.py:1060`).

WHAT TO IMPLEMENT
-----------------
Mirror `generate_nn_sepsis_effects.py`:

1. Accept the same CLI shape as the sepsis generator:

       python generate_birth_effects.py [--skip-existing] [draw_numbers...]

   `parallel_runner.py` appends a single draw as the last positional arg,
   so the script must accept one draw per invocation.

2. For each draw, call (from `hgb_birth_effect_generation`):
       - calculate_and_save_lbwsg_shifts(results_dir, draw)
             --> lbwsg_shifts/draw_<N>.csv
       - calculate_and_save_iv_iron_lbwsg_shifts(results_dir, draw)
             --> iv_iron_lbwsg_shifts/draw_<N>.csv
   `results_dir` should be this file's parent directory.

3. `--skip-existing` should skip draws whose output CSVs already exist in
   BOTH `lbwsg_shifts/` and `iv_iron_lbwsg_shifts/`.

NOT DRAW-PARALLEL
-----------------
`calculate_iv_iron_stillbirth_effects()` loops all draws internally
(~7 min total) and produces `iv_iron_stillbirth_rrs.csv`. Do NOT call it
from this script -- handle it as a separate single-invocation workflow
step (see workflow_config.yaml).

EXPECTED RUNTIME
----------------
~10-15 min per draw, sequential within a worker. With 20 SCENARIO_DRAWS
and 4 workers, expect ~1 hour wall time (pad accordingly in the workflow
resources block).
"""
