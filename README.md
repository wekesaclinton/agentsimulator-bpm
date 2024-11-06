# AgentSimulator
This is the supplementary GitHub repository of the paper: "AgentSimulator: An Agent-based Approach for Data-driven Business Process Simulation".

![](https://github.com/lukaskirchdorfer/AgentSimulator/blob/main/AgentSim.png)

## Prerequisites
To execute the code, you need to create an environment (e.g., with `conda create -n myenv python=3.9`) and install the dependencies in `requirements.txt` (e.g., with `install --yes --file requirements.txt`).

## How to run the AgentSimulator
To run the script MAS_Simulation.py you need to specify the following parameters:

`--log_path`: path to the entire log which you need to store in the folder raw_data

`--case_id`: name of the case_id column

`--activity_name`: name of the activity column

`--resource_name`: name of the resource column

`--end_timestamp`: name of the end timestamp column

`--start_timestamp`: name of the start timestamp column

The hyperparameters handover type and whether to consider extraneous delays are determined automatically. If you do not want them to be automatically determined but want to run a specific configuration, just add the following parameters:

`--central_orchestration`: True or False (whether handovers are centrally orchestrated or autonomous)

`--extr_dealys`: True or False (whether to discover extraneous delays)

Commands to run the datasets evaluated in our paper:

Loan Application:
```
python MAS_Simulation.py --log_path raw_data/LoanApp.csv.gz --case_id case_id --activity_name activity --resource_name resource --end_timestamp end_time --start_timestamp start_time
```

P2P
```
python MAS_Simulation.py --log_path raw_data/P2P.csv --case_id case:concept:name --activity_name concept:name --resource_name Resource --end_timestamp time:timestamp --start_timestamp start_timestamp
```

Production
```
python MAS_Simulation.py --log_path raw_data/Production.csv --case_id caseid --activity_name task --resource_name user --end_timestamp end_timestamp --start_timestamp start_timestamp
```

ACR (Consulta Data Mining):
```
python MAS_Simulation.py --log_path raw_data/ConsultaDataMining.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp
```

BPIC 2012 W:
```
python MAS_Simulation.py --log_path raw_data/BPIC_2012_W.csv --case_id case:concept:name --activity_name Activity --resource_name Resource --end_timestamp time:timestamp --start_timestamp start_timestamp
```

CVS Pharmacy:
```
python MAS_Simulation.py --log_path raw_data/cvs_pharmacy.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp
```

BPIC 2017 W:
```
python MAS_Simulation.py --log_path raw_data/BPIC_2017_W.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp
```

Confidential 1000:
```
python MAS_Simulation.py --log_path raw_data/Confidential_1000.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp
```

Confidential 2000:
```
python MAS_Simulation.py --log_path raw_data/Confidential_2000.csv --case_id case:concept:name --activity_name concept:name --resource_name org:resource --end_timestamp time:timestamp --start_timestamp start_timestamp
```

The 10 simulated logs as well as train and test splits are then stored in the folder simulated_data.

## Evaluation
The simulation results can be evaluated using the evaluation.ipynb notebook.
The raw event logs as well as all simulated event logs for the 9 mentioned processes can be found in this [Google Drive folder](https://drive.google.com/drive/folders/1D0jgBcPYNw-yBFyc_Ro3a681_c-BzM0u?usp=sharing).

## Additional Results 
The following table complements the results mentioned in the paper and reports the AgentSimulator results (incl. std) for both orchestrated and autonomous handovers. The underlying simulated log files can also be found in the Google Drive folder.

| Log | Handover        | NGD  | AED  | CED  | RED  | CTD |
|-------| -----------|----| ----|----| ----|----|
| Loan Appl.|orchestrated |0.08 (0.02)|2.91 (0.75)      |0.23 (0.04)|1.38 (0.27)|1.85 (0.58)|
| Loan Appl.|autonomous |0.07 (0.02)|2.76 (0.64)      |0.21 (0.03)|1.34 (0.34)|1.49 (0.63)|
| P2P  |orchestrated   | 0.25 (0.03)   | 1161.32 (10.64)  |1.02 (0.10)|658.61 (9.76)|525.15 (13.98)|
| P2P|autonomous     | 0.25 (0.02)    | 1122.42 (16.96)  |0.98 (0.07)|668.21 (16.10)|529.08 (6.58)|
| CVS |orchestrated    |0.12 (0.01)|89.31 (1.50)|7.48 (0.04)|81.76 (1.52)|101.49 (2.35)|
| CVS|autonomous     |0.12 (0.00)|87.81 (1.00)|7.43 (0.04)|81.35 (0.94)|100.27 (1.73)|
| Conf. 1000|orchestrated |0.26 (0.01)|172.00 (2.57)|1.77 (0.14)|18.07 (3.55)|31.37 (4.55)|
| Conf. 1000|autonomous |0.25 (0.01)|127.01 (5.80)|1.68 (0.14)|16.85 (1.98)|26.10 (3.55)|
| Conf. 2000|orchestrated |0.26 (0.01)|98.60 (11.52)|1.40 (0.05)|8.61 (1.32)|15.59 (1.65)|
| Conf. 2000|autonomous |0.26 (0.01)|212.54 (8.25)|1.41 (0.10)|9.30 (1.51)|17.87 (2.67)|
| ACR|orchestrated |0.36 (0.02)|333.28 (1.90)|7.59 (0.23)|27.12 (0.44)|76.50 (0.61)|
| ACR|autonomous |0.49 (0.02)|335.17 (2.89)|7.07 (0.25)|26.57 (0.81)|76.30 (0.69)|
| Production|orchestrated |0.61 (0.01)|65.29 (11.44)|5.83 (0.16)|14.20 (6.93)|25.79 (6.34)|
| Production|autonomous |0.77 (0.02)|82.49 (9.05)|5.61 (0.35)|22.93 (5.72)|45.65 (6.95)|
| BPI12W|orchestrated |0.15 (0.01)|79.89 (12.83)|1.87 (0.09)|47.84 (6.34)|90.12 (3.78)|
| BPI12W|autonomous |0.21 (0.02)|92.01 (17.08)|1.92 (0.11)|52.05 (10.20)|96.91 (3.66)|
| BPI17W|orchestrated |0.19 (0.00)|221.49 (3.75)|1.79 (0.01)|50.01 (2.68)|54.82 (2.03)|
| BPI17W|autonomous |0.30 (0.01)|220.98 (3.43)|1.64 (0.02)|26.03 (2.16)|22.75 (1.45)|

## Authors
Lukas Kirchdorfer, Robert Blümel, Timotheus Kampik, Han van der Aa, Heiner Stuckenschmidt
