# AgentSimulator
This is the supplementary GitHub repository of the paper: "AgentSimulator: An Agent-based Approach for Data-driven Business Process Simulation".

## Prerequisites
To execute the code, you need to create an environment and install the following packages
- mesa
- pm4py
- pandas
- numpy
- scipy
- polars

## How to run the AgentSimulator
To run the script MAS_Simulation.py you need to specify the following parameters:

--log_path : path to the entire log which should be stored in the folder raw_data

--case_id : name of the case_id column

--activity_name : name of the activity column

--resource_name : name of the resource column

--end_timestamp : name of the end_timestamp column

--start_timestamp : name of the start_timestamp column

--extr_delays : whether to discover extraneous delays (True or False)

--central_orchestration : whether the MAS is centrally orchestrated or agents have autonomous handover behavior (True or False)

Commands to run the datasets evaluated in our paper:
Loan Application:
```
python MAS_Simulation.py --log_path raw_data/LoanApp.csv.gz --case_id case_id --activity_name activity --resource_name resource --end_timestamp end_time --start_timestamp start_time --extr_delays False --central_orchestration False
```

P2P
```
python MAS_Simulation.py --log_path raw_data/P2P.csv --case_id case:concept:name --activity_name concept:name --resource_name Resource --end_timestamp time:timestamp --start_timestamp start_timestamp --extr_delays True --central_orchestration False
```

Production
```
python MAS_Simulation.py --log_path raw_data/Production.csv --case_id caseid --activity_name task --resource_name user --end_timestamp end_timestamp --start_timestamp start_timestamp --extr_delays False --central_orchestration False
```

Academic Credentials (ACR):
```
python MAS_Simulation.py --log_path raw_data/ConsultaDataMining.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp --extr_delays False --central_orchestration False
```

Consulta Data Mining:
```
python MAS_Simulation.py --log_path raw_data/ConsultaDataMining.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp
```

BPIC 2012 W:
```
python MAS_Simulation.py --log_path raw_data/BPIC_2012_W.csv --case_id case:concept:name --activity_name Activity --resource_name Resource --end_timestamp time:timestamp --start_timestamp start_timestamp --extr_delays True --central_orchestration False
```

CVS Pharmacy:
```
python MAS_Simulation.py --log_path raw_data/cvs_pharmacy.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp --extr_delays True --central_orchestration False
```

BPIC 2017 W:
```
python MAS_Simulation.py --log_path raw_data/BPIC_2017_W.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp --extr_delays True --central_orchestration False
```

Confidential 1000:
```
python MAS_Simulation.py --log_path raw_data/Confidential_1000.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp --extr_delays False --central_orchestration False
```

Confidential 2000:
```
python MAS_Simulation.py --log_path raw_data/Confidential_2000.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp --extr_delays False --central_orchestration False
```

The simulated logs (10 logs) as well as train and test log splits are then stored in the folder simulated_data.

## Evaluation
The simulation results can be evaluated using the evaluate_simulation_extended.ipynb notebook.

## Authors
Lukas Kirchdorfer, Robert Blümel, Timotheus Kampik, Han van der Aa, Heiner Stuckenschmidt
