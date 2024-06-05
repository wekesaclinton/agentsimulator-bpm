# AgentSimulator
This is the supplementary GitHub repository of the paper: "AgentSimulator: An Agent-based Approach for Data-driven Business Process Simulation".

![](https://github.com/lukaskirchdorfer/AgentSimulator/blob/main/AgentSim.png)

## Prerequisites
To execute the code, you need to create an environment (e.g., with conda create -n myenv python=3.9) and install the following packages
- pip install mesa==2.1.2
- pip install pm4py
- pip install polars
- pip install pyarrow
- pip install log_distance_measures

## How to run the AgentSimulator
To run the script MAS_Simulation.py you need to specify the following parameters:

--log_path : path to the entire log which you need to store in the folder raw_data

--case_id : name of the case_id column

--activity_name : name of the activity column

--resource_name : name of the resource column

--end_timestamp : name of the end_timestamp column

--start_timestamp : name of the start_timestamp column

--central_orchestration : whether the MAS is centrally orchestrated or agents have autonomous handover behavior (True or False)

Commands to run the datasets evaluated in our paper:

Loan Application:
```
python MAS_Simulation.py --log_path raw_data/LoanApp.csv.gz --case_id case_id --activity_name activity --resource_name resource --end_timestamp end_time --start_timestamp start_time --central_orchestration False
```

P2P
```
python MAS_Simulation.py --log_path raw_data/P2P.csv --case_id case:concept:name --activity_name concept:name --resource_name Resource --end_timestamp time:timestamp --start_timestamp start_timestamp --central_orchestration False
```

Production
```
python MAS_Simulation.py --log_path raw_data/Production.csv --case_id caseid --activity_name task --resource_name user --end_timestamp end_timestamp --start_timestamp start_timestamp --central_orchestration False
```

ACR (Consulta Data Mining):
```
python MAS_Simulation.py --log_path raw_data/ConsultaDataMining.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp
```

BPIC 2012 W:
```
python MAS_Simulation.py --log_path raw_data/BPIC_2012_W.csv --case_id case:concept:name --activity_name Activity --resource_name Resource --end_timestamp time:timestamp --start_timestamp start_timestamp --central_orchestration False
```

CVS Pharmacy:
```
python MAS_Simulation.py --log_path raw_data/cvs_pharmacy.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp --central_orchestration False
```

BPIC 2017 W:
```
python MAS_Simulation.py --log_path raw_data/BPIC_2017_W.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp --central_orchestration False
```

Confidential 1000:
```
python MAS_Simulation.py --log_path raw_data/Confidential_1000.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp --central_orchestration False
```

Confidential 2000:
```
python MAS_Simulation.py --log_path raw_data/Confidential_2000.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp --central_orchestration False
```

The simulated logs (10 logs) as well as train and test log splits are then stored in the folder simulated_data.

## Evaluation
The simulation results can be evaluated using the evaluate_simulation_extended.ipynb notebook.
The raw event logs as well as all simulated event logs for the 9 mentioned processes can be found in this [Google Drive folder](https://drive.google.com/drive/folders/1D0jgBcPYNw-yBFyc_Ro3a681_c-BzM0u?usp=sharing).

## Additional Results 
The following table complements the results mentioned in the paper and reports the AgentSimulator results using orchestrated handovers. The underlying simulated log files can also be found in the Google Drive folder.

| Log        | NGD  | AED  | CED  | RED  | CTD |
| -----------|:----:| ----:|:----:| ----:|----:|
| Loan Appl. |0.08|2.76      |0.22|1.36|1.50|
| P2P     | 0.25    | 1098.16  |1.07|674.83|528.99|
| CVS     |0.12|94.99|7.46|88.66|111.07|
| Conf. 1000 |0.26|135.70|1.66|13.22|23.16|
| Conf. 2000 |0.25|236.91|1.49|9.30|17.73|
| ACR |0.37|281.58|7.21|27.32|77.26|
| Production |0.62|61.80|5.69|17.77|23.45|
| BPI12W |0.15|81.52|1.89|49.61|92.99|
| BPI17W |0.19|221.49|1.79|50.01|54.82|

## Authors
Lukas Kirchdorfer, Robert Blümel, Timotheus Kampik, Han van der Aa, Heiner Stuckenschmidt
