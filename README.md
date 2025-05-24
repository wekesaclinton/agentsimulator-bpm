# AgentSimulator
This is the supplementary GitHub repository of the paper: "AgentSimulator: An Agent-based Approach for Data-driven Business Process Simulation".

![](https://github.com/lukaskirchdorfer/AgentSimulator/blob/main/AgentSim.png)

## Prerequisites

`python3.11 -m venv agent_sim_env

source agent_sim_env/bin/activate`

## How to run the AgentSimulator

`python simulate.py --log_path raw_data/Loans.csv --case_id case_id --activity_name activity --resource_name resource --end_timestamp end_time --start_timestamp start_time
`

This is by renaming the --activity_name and --resource_name to the exact ones on the excel shared.

