Prior to November 2025, these files were in [the research repository](https://github.com/ihmeuw/vivarium_research_mncnh_portfolio).
See the `facility_choice` subdirectory there for extended Git history, and many ancillary/scratch files, at [this link](https://github.com/ihmeuw/vivarium_research_mncnh_portfolio/tree/21553cec52f351d910d90ae3647cb59d83886592/facility_choice).

Not all of the data needed to run this is included in this repository due to access concerns.
You will need to download [this file from SharePoint](https://uwnetid.sharepoint.com/:x:/r/sites/ihme_simulation_science_team/_layouts/15/Doc.aspx?sourcedoc=%7BF3B3DD5F-641F-413B-A353-7C4CA3364CDF%7D&file=facility_choice_data.xlsx&action=default&mobileredirect=true)
to this directory before running.

Note that running the main notebook here requires a different environment than either building the artifact
or running the simulation; instructions to create it are included in the notebook itself.

In effect, the code in this directory re-implements a large part of the simulation outside of Vivarium (we commonly refer to a microsim without Vivarium as a "nanosim").
We could consider using the actual simulation for this optimization/calibration in the future.