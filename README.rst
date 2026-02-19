===============================
vivarium_gates_mncnh
===============================

Research repository for the vivarium_gates_mncnh project.

.. contents::
   :depth: 1

Installation
------------

You will need ``conda`` to install all of this repository's requirements.
We recommend installing `Miniforge <https://github.com/conda-forge/miniforge>`_.

Once you have conda installed, you should open up your normal shell
(if you're on linux or OSX) or the ``git bash`` shell if you're on windows.

You'll then clone this repository and make the necessary environments.
The first step is to clone the repo::

  :~$ git clone https://github.com/ihmeuw/vivarium_gates_mncnh.git
  ...git will copy the repository from github and place it in your current directory...
  :~$ cd vivarium_gates_mncnh

Users can create environments by running
``bash environment.sh`` and ``bash environment.sh -t artifact`` which will automatically create and active conda environments
for running the simulation and artifact generation respectively.
The environment.sh script has extra options for users. To see these options, pass the 
``-h`` flag.

Alternatively, users can manually create the environments as follows::

  :~$ conda create --name=vivarium_gates_mncnh_simulation python=3.11 git git-lfs
  ...conda will download python and base dependencies...
  :~$ conda activate vivarium_gates_mncnh_simulation
  (vivarium_gates_mncnh_simulation) :~$ pip install -r requirements.txt
  (vivarium_gates_mncnh_simulation) :~$ pip install -e .[dev]
  ...pip will install vivarium and other requirements...
  (vivarium_gates_mncnh_simulation) :~$ conda deactivate
  :~$ conda create --name=vivarium_gates_mncnh_artifact python=3.11 git git-lfs
  ...conda will download python and base dependencies...
  :~$ conda activate vivarium_gates_mncnh_artifact
  (vivarium_gates_mncnh_artifact) :~$ pip install -r artifact_requirements.txt
  (vivarium_gates_mncnh_artifact) :~$ pip install -e .[dev]
  ...pip will install vivarium and other requirements...

Supported Python versions: 3.10, 3.11

Note the ``-e`` flag that follows pip install. This will install the python
package in-place, which is important for making the model specifications later.

Vivarium uses the Hierarchical Data Format (HDF) as the backing storage
for the data artifacts that supply data to the simulation. You may not have
the needed libraries on your system to interact with these files, and this is
not something that can be specified and installed with the rest of the package's
dependencies via ``pip``. If you encounter HDF5-related errors, you should
install hdf tooling from within your environment like so::

  (vivarium_gates_mncnh) :~$ conda install hdf5

The ``(vivarium_gates_mncnh)`` that precedes your shell prompt will probably show
up by default, though it may not.  It's just a visual reminder that you
are installing and running things in an isolated programming environment
so it doesn't conflict with other source code and libraries on your
system.


Usage
-----

You'll find six directories inside the main
``src/vivarium_gates_mncnh`` package directory:

- ``artifacts``

  This directory contains all input data used to run the simulations.
  You can open these files and examine the input data using the vivarium
  artifact tools.  A tutorial can be found at https://vivarium.readthedocs.io/en/latest/tutorials/artifact.html#reading-data

- ``components``

  This directory is for Python modules containing custom components for
  the vivarium_gates_mncnh project. You should work with the
  engineering staff to help scope out what you need and get them built.

- ``data``

  If you have **small scale** external data for use in your sim or in your
  results processing, it can live here. This is almost certainly not the right
  place for data, so make sure there's not a better place to put it first.

- ``model_specifications``

  This directory should hold all model specifications and branch files
  associated with the project.

- ``results_processing``

  Any post-processing and analysis code or notebooks you write should be
  stored in this directory.

- ``tools``

  This directory hold Python files used to run scripts used to prepare input
  data or process outputs.


Running Simulations
-------------------

You will need to repeat the entire process documented here for each location you want to run for.
Only Pakistan, Nigeria, and Ethiopia are supported currently.
In all commands here, we use Pakistan as an example;
replace "Pakistan" with the name of the location of interest.

To run this simulation, the first step is to analyze GBD data to generate "caps" (maximum values)
for relative risks of low birthweight and short gestation (LBWSG).
Note that this takes a while to run (about an hour).
If you don't want to re-generate the RR caps, you can skip this step and simply use the pre-generated
files included in this repo.
Generating the caps is achieved with:::

  :~$ conda activate vivarium_gates_mncnh_artifact
  (vivarium_gates_mncnh_artifact) :~$ python src/vivarium_gates_mncnh/data/lbwsg_rr_caps/generate_caps.py -l Pakistan -o src/vivarium_gates_mncnh/data/lbwsg_rr_caps/caps/

The next step is to generate an artifact with base GBD data in it.
This will only work on the IHME cluster, because it pulls draw-level data from internal GBD databases.:::

  :~$ conda activate vivarium_gates_mncnh_artifact
  (vivarium_gates_mncnh_artifact) :~$ make_artifacts -vvv -l "Pakistan" -o artifacts/

This command will create an artifact file in the ``artifacts/`` directory within the repo;
omit the ``-o`` argument to output to the default location of ``/mnt/team/simulation_science/pub/models/vivarium_gates_mncnh/artifacts``,
or change to a different path.

The next step is to run an initial simulation to calculate population-attributable fractions (PAFs)
for LBWSG in the early neonatal period.
*Edit* the ``time`` section of ``src/vivarium_gates_mncnh/data/lbwsg_paf.yaml`` so that the ``end``
is only one day after the ``start``, then run:::

  :~$ conda activate vivarium_gates_mncnh_simulation
  (vivarium_gates_mncnh_simulation) :~$ simulate run -vvv src/vivarium_gates_mncnh/data/lbwsg_paf.yaml -i artifacts/pakistan.hdf -o paf_sim_results/

The ``-v`` flag will log verbosely, so you will get log messages every time
step. For more ways to run simulations, see the tutorials at
https://vivarium.readthedocs.io/en/latest/tutorials/running_a_simulation/index.html
and https://vivarium.readthedocs.io/en/latest/tutorials/exploration.html

This command will output results in the ``paf_sim_results/`` directory within the repo;
omit the ``-o`` argument to output to the default location in your home directory (``~/vivarium_results/lbwsg_paf/``),
or change to a different path.

The last line of output will tell you the specific directory to which results were written.
Make a directory for holding these results, and copy them there, as follows:::

  :~$ mkdir -p calculated_pafs/temp_outputs/pakistan/
  :~$ cp <your results directory>/calculated_lbwsg_paf*.parquet calculated_pafs/temp_outputs/pakistan/

Now *edit* the ``PAF_DIR =`` line of ``src/vivarium_gates_mncnh/constants/paths.py`` to set the value to
``Path("calculated_pafs/")``.
You'll now re-run the ``make_artifacts`` command, updating the relevant PAFs:::

  :~$ conda activate vivarium_gates_mncnh_artifact
  (vivarium_gates_mncnh_artifact) :~$ make_artifacts -vvv -l "Pakistan" -o artifacts/ -r risk_factor.low_birth_weight_and_short_gestation.population_attributable_fraction -r cause.neonatal_preterm_birth.population_attributable_fraction

Next we'll repeat the process to calculate PAFs and preterm prevalence for late neonatals.
*Undo* your edits in the ``time`` section of ``src/vivarium_gates_mncnh/data/lbwsg_paf.yaml``
and re-run:::

  :~$ conda activate vivarium_gates_mncnh_simulation
  (vivarium_gates_mncnh_simulation) :~$ simulate run -vvv src/vivarium_gates_mncnh/data/lbwsg_paf.yaml -i artifacts/pakistan.hdf -o paf_sim_results/

*Edit* the ``PRETERM_PREVALENCE_DIR =`` line of ``src/vivarium_gates_mncnh/constants/paths.py`` to set the value to
``Path("calculated_preterm_prevalence/")``.
Copy your results to ``calculated_pafs`` and ``calculated_preterm_prevalence``, overwriting the previous results:

  :~$ cp <your results directory>/calculated_lbwsg_paf*.parquet calculated_pafs/temp_outputs/pakistan/
  :~$ mkdir -p calculated_preterm_prevalence/pakistan/
  :~$ cp <your results directory>/calculated_late_neonatal_preterm*.parquet calculated_preterm_prevalence/pakistan/

You'll now re-run the ``make_artifacts`` command, updating the relevant PAFs:::

  :~$ conda activate vivarium_gates_mncnh_artifact
  (vivarium_gates_mncnh_artifact) :~$ make_artifacts -vvv -l "Pakistan" -o artifacts/ -r risk_factor.low_birth_weight_and_short_gestation.population_attributable_fraction -r cause.neonatal_preterm_birth.population_attributable_fraction -r cause.neonatal_preterm_birth.prevalence

You are now ready to run the main simulation with::

  :~$ conda activate vivarium_gates_mncnh_simulation
  (vivarium_gates_mncnh_simulation) :~$ simulate run -vvv src/vivarium_gates_mncnh/model_specifications/model_spec.yaml -i artifacts/pakistan.hdf -o sim_results/

Results of the simulation will be written to ``sim_results/``.
For example, you can check the total deaths due to maternal disorders by
summing the ``value`` column in the Parquet file at
``sim_results/pakistan/<timestamp>/results/maternal_disorders_burden_observer_disorder_deaths.parquet``.

V&V process
-----------

When we make changes to the model, we do so in a git branch, so that changes are not merged into the main branch until they have been verified and validated (V&V).

However, this process is a bit more involved than typical software testing, and may involve multiple people:
the model-runner person and the V&V person.
Often the model-runner is an engineer, though with task shifting it is becoming more common for researchers to take on this role as well.
The general process is as follows:

1. The model-runner creates a new git branch (the name of which should include the model number), and makes the changes to the model.
2. If necessary, the model-runner creates a new artifact in a model-number-named directory, and updates the path in the model spec.
3. The model-runner runs the pytest suite *without updating the MODEL_RESULTS_DIR constant*.
   If the tests fail, they post on Slack and do a cursory (time-boxed: 15 minutes) investigation into why.
   If that identifies the root cause, and solution to, the issue, they fix it and repeat this step.
   Otherwise, the V&V person investigates, by making a new branch off the model-runner's branch,
   running the tests, and updating the tests if needed or summarizing the issue.
4. The model-runner runs the simulation with `psimulate` saving results to a model-number-named directory.
   They update the MODEL_RESULTS_DIR constant in the code to reflect this new directory,
   and create a git tag for the new model version.
5. The model-runner runs the pytest suite *again*, to ensure that the tests pass with the new results.
   If the tests fail, they post on Slack and do a cursory (time-boxed: 15 minutes) investigation into why.
   If that identifies the root cause, and solution to, the issue, they fix it and return to step 4 (if the issue doesn't involve the artifact)
   or step 2 (if the issue does involve the artifact).
   Otherwise, the V&V person investigates, by making a new branch off the model-runner's branch,
   running the tests, and updating the tests if needed or summarizing the issue.
6. If all existing tests have now passed, and additional tests are needed for new functionality,
   the V&V person now makes a branch off the model-runner's branch, and adds the new tests.
   If the new tests do not pass, the V&V person investigates why.
7. If all tests have now passed, the model-runner's branch can be merged to the main branch.
   If the V&V person also has a branch, this can be merged to the model-runner's branch first,
   or can be merged directly to main *after* the model-runner's branch is merged to main;
   this decision should be made based on the potential merge conflicts and who would be better placed to resolve them.
   If all tests have *not* passed and a follow-up run is needed to fix issues or investigate further,
   the whole process repeats, branching off the *last* branch made during this process
   (so, the V&V person's branch, if there is one).