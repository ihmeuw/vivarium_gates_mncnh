## Title: Summary, imperative, start upper case, don't end with a period
<!-- Ideally, <=50 chars. 50 chars is here..: -->

### Description
<!-- For use in commit message, wrap at 72 chars. 72 chars is here: -->
- *Category*: <!-- one of bugfix, data artifact, implementation, observers,
                   post-processing, refactor, revert, test, release, other/misc -->
- *JIRA issue*: https://jira.ihme.washington.edu/browse/MIC-XYZ
- *Research reference*: <!--Link to research documentation for code -->

### Changes and notes
<!-- 
Change description – why, what, anything unexplained by the above.
Include guidance to reviewers if changes are complex.
--> 

### Verification and Testing
<!--
Details on how code was verified. Consider: plots, images, (small) csv files.
-->


*** REMINDER ***
CI WILL NOT RUN ANY TESTS.
MANUALLY RUN TESTS WITH EACH PR.
MAKE SURE CONSTANTS/PATHS.PY IS USING THE CORRECT MODEL RESULTS DIRECTORY.
-->
- [ ] model results directory is up to date
- [ ] all tests pass (`pytest --runslow` with both *vivarium_gates_mncnh_artifact* and *vivarium_gates_mncnh_simulation*)