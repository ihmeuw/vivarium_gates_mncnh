/* This Jenkinsfile simply loads the `reusable_pipeline` pipeline from the
vivariu_build_utils repository (https://github.com/ihmeuw/vivarium_build_utils).

vivarium_build_utils is loaded as a Jenkins shared library 
(https://www.jenkins.io/doc/book/pipeline/shared-libraries/).
Jenkins shared library convention dictates that importable modules must be stored
in the 'vars' folder.

Jenkins shared libraries can be configured in the Jenkins UI:
* Manage Jenkins
  * Configure System
    * Global Pipeline Libraries section
      * Library subsection
        * Name: The Name for the lib
        * Version: The branch you want to use. Throws an error
                   for nonexistent branches.
        * Project Repository: Url to the shared lib
        * Credentials: SSH key to access the repo

Note that updating the shared repo will take affect on the next pipeline invocation.
*/ 

// Load the get_vbu_version function from vivarium_build_utils/bootstrap/
// (the directory to load from is defined in the Jenkins shared library configuration)
// TODO: load get_vbu_version@main at the end of the uv epic (MIC-7687 section 7).
// Until then, use the bootstrap on the vivarium-suite epic branch, which reads uv.lock.
@Library("get_vbu_version@epic/uv-environments") _

// Load the full vivarium_build_utils library at the expected version
library("vivarium_build_utils@${get_vbu_version()}")

reusable_pipeline(
    // Build the environment from uv.lock instead of a conda environment.
    uv_lock: true,
    // "Github App Enterprise", which can read the ihme-internal git sources in uv.lock.
    // TODO: remove once uv.lock has no ihme-internal git sources, that is, once
    // vivarium_inputs and vivarium_gbd_access are released.
    uv_git_credentials_id: "892b2730-76c1-41b5-a1b6-41daee5b1812",
    scheduled_branches: [
      // Add additional branches for cron jobs below.
      // Note: Newly added branches will require a manual trigger of the first build 
      // for subsequent scheduled builds to run.
      "main"
    ],
    requires_slurm: true,
    skip_doc_build: true,
)
