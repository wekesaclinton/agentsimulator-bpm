import pandas as pd

from source.utils import store_preprocessed_data
from source.agent_types.discover_roles import discover_roles_and_calendars
from source.agent_types.discover_resource_calendar import discover_calendar_per_agent
from source.arrival_distribution import get_best_fitting_distribution
from source.arrival_times import get_case_arrival_times
from source.activity_transition import compute_activity_transition_dict, compute_activity_transition_dict_global
from source.interaction_probabilities import calculate_agent_handover_probabilities_per_activity
from source.extraneous_delays.config import (
    Configuration as ExtraneousActivityDelaysConfiguration,
    TimerPlacement,
)
from source.extraneous_delays.delay_discoverer import compute_complex_extraneous_activity_delays, compute_naive_extraneous_activity_delays
from source.extraneous_delays.event_log import EventLogIDs
from source.simulation import BusinessProcessModel, Case

def discover_simulation_parameters(df_train, df_test, df_val, data_dir, num_cases_to_simulate, num_cases_to_simulate_val, determine_automatically=False, central_orchestration=False, discover_extr_delays=False):
    """
    Discover the simulation model from the training data.
    """
    df_train, agent_to_resource = preprocess(df_train)
    df_test, _ = preprocess(df_test)
    df_val, _ = preprocess(df_val)
    START_TIME = min(df_test.groupby('case_id')['start_timestamp'].min().to_list())
    START_TIME_VAL = min(df_val.groupby('case_id')['start_timestamp'].min().to_list())

    activity_durations_dict = compute_activity_duration_distribution_per_agent(df_train)

    df_train_without_end_activity = store_preprocessed_data(df_train, df_test, df_val, data_dir)

    activities_without_waiting_time = activities_with_zero_waiting_time(df_train)

    # extract roles and calendars  
    roles = discover_roles_and_calendars(df_train_without_end_activity)
    res_calendars, _, _, _, _ = discover_calendar_per_agent(df_train_without_end_activity)

    # define mapping of agents to activities based on event log
    agent_activity_mapping = df_train.groupby('agent')['activity_name'].unique().apply(list).to_dict()

    transition_probabilities_autonomous = compute_activity_transition_dict(df_train)
    agent_transition_probabilities_autonomous = calculate_agent_handover_probabilities_per_activity(df_train)
    agent_transition_probabilities = None
    transition_probabilities = compute_activity_transition_dict_global(df_train)

    prerequisites, parallel_activities = get_prerequisites_per_activity(df_train)

    # get maximum activity frequency per case
    activity_counts = df_train.groupby(['case_id', 'activity_name']).size().reset_index(name='count')
    max_activity_count_per_case = activity_counts.groupby('activity_name')['count'].max().to_dict()


    # sample arrival times for training and validation data
    case_arrival_times, train_params = get_case_arrival_times(df_train, start_timestamp=START_TIME, num_cases_to_simulate=num_cases_to_simulate, train=True)
    case_arrival_times_val, _ = get_case_arrival_times(df_val, start_timestamp=START_TIME_VAL, num_cases_to_simulate=num_cases_to_simulate_val, train=False, train_params=train_params)

    simulation_parameters = {
        'activity_durations_dict': activity_durations_dict,
        'activities_without_waiting_time': activities_without_waiting_time,
        'roles': roles,
        'res_calendars': res_calendars,
        'agent_activity_mapping': agent_activity_mapping,
        'transition_probabilities_autonomous': transition_probabilities_autonomous,
        'agent_transition_probabilities_autonomous': agent_transition_probabilities_autonomous,
        'agent_transition_probabilities': agent_transition_probabilities,
        'transition_probabilities': transition_probabilities,
        'max_activity_count_per_case': max_activity_count_per_case,
        'case_arrival_times': case_arrival_times,
        'case_arrival_times_val': case_arrival_times_val,
        'agent_to_resource': agent_to_resource,
        'determine_automatically': determine_automatically,
        'prerequisites': prerequisites,
    }

    simulation_parameters = determine_agent_behavior_type_and_extraneous_delays(simulation_parameters, df_train, df_val, case_arrival_times_val, central_orchestration, discover_extr_delays)
    simulation_parameters['start_timestamp'] = START_TIME


    return df_train, simulation_parameters

def preprocess(df):
    """
    Preprocess event log 
    """
    # transfrom case_id into int
    if df['case_id'].dtype == 'object':
        df['case_id'] = df['case_id'].str.extract('(\d+)').astype(int)
    # fill NaN values of resource column
    df['resource'].fillna('artificial', inplace=True)
    df['resource'] = df['resource'].astype(str)


    def rename_artificial(row):
        if row['resource'] == 'artificial':
            return row['resource'] + '_' + row['activity_name']
        else:
            return row['resource']
        
    def insert_rows_before_case_change(df):
        # Ensure the DataFrame is sorted by case_id and start_timestamp
        df.sort_values(by=['case_id', 'start_timestamp'], inplace=True)
        # Group by case_id and find the row with the latest start_timestamp in each group
        last_rows = df.groupby('case_id').tail(1)
        # Create a copy of the last row for each case and update the activity_name to 'zzz_end'
        end_rows = last_rows.copy()
        end_rows['activity_name'] = 'zzz_end'
        end_rows['start_timestamp'] = end_rows['end_timestamp']
        # Concatenate the original DataFrame with the new end_rows
        df_with_end_rows = pd.concat([df, end_rows], ignore_index=True)
        df_with_end_rows.sort_values(by=['case_id', 'start_timestamp'], inplace=True)
        df_with_end_rows.reset_index(drop=True, inplace=True)
        return df_with_end_rows

    # Apply the function to each row
    df['resource'] = df.apply(rename_artificial, axis=1)
    # name agents with plain integers 
    df['agent'] = pd.factorize(df['resource'])[0]
    # Create a mapping of integers to resource values
    integers = df['agent'].unique()
    resources = df['resource'].unique()
    integers_to_resources = dict(zip(integers, resources))
    # Create the agent_to_resource mapping dictionary by reversing the integers_to_resources dictionary
    agent_to_resource = {k: v for k, v in integers_to_resources.items()}
    # insert a new row after every ending case
    df = insert_rows_before_case_change(df)
    df['start_timestamp']= pd.to_datetime(df['start_timestamp'], utc=True, format='mixed')
    df['end_timestamp']= pd.to_datetime(df['end_timestamp'], utc=True, format='mixed')

    return df, agent_to_resource


def _compute_activity_duration_distribution(df):
    """
    computes mean and std for each activity duration in the log and returns this in form of a dict
    """
    activities = sorted(set(df['activity_name']))
    agents = sorted(set(df['agent']))
    act_durations = {key: {k: [] for k in activities} for key in agents}
    for agent in agents:
        for activity in activities:
            for i in range(len(df)):
                if df['agent'][i] == agent:
                    if df['activity_name'][i] == activity:
                        duration = (df['end_timestamp'][i] - df['start_timestamp'][i]).total_seconds()
                        act_durations[agent][activity].append(duration)

    return act_durations

def compute_activity_duration_distribution_per_agent(df_train):
    """
    Compute the best fitting distribution of activity durations per agent.

    Args:
        df_train: Event log in pandas format

    Returns:
        dict: A dict storing for each agent the distribution for each activity.
    """
    activity_durations_dict = _compute_activity_duration_distribution(df_train)

    agents = activity_durations_dict.keys()
    activities = []
    for k,v in activity_durations_dict.items():
        for kk, vv in v.items():
            activities.append(kk)
    activities = set(activities)

    act_duration_distribution_per_agent = {agent: {act: [] for act in activities} for agent in agents}

    for agent, val in activity_durations_dict.items():
        for act, duration_list in val.items():
            if len(duration_list) > 0:
                duration_distribution = get_best_fitting_distribution(
                    data=duration_list,
                    filter_outliers=True,
                    outlier_threshold=20.0,
                )
                act_duration_distribution_per_agent[agent][act] = duration_distribution

    return act_duration_distribution_per_agent

def activities_with_zero_waiting_time(df, threshold=0.99):
    """
    Returns a list of activities that have zero waiting time in the log.
    """
    # Sort the DataFrame by start timestamp
    df_sorted = df.sort_values(by='start_timestamp')
    df_sorted = df_sorted.groupby('case_id')

    group_list = []

    for case_id, group in df_sorted:
        # Calculate waiting time for each group
        group['waiting_time'] = group['start_timestamp'] - group['end_timestamp'].shift(1)
        # Append the modified group to the list
        group_list.append(group)

    # Concatenate the list of groups back into a single DataFrame
    df_with_waiting_time = pd.concat(group_list)

    # replace NaT values (first activity per case) with 0
    df_with_waiting_time['waiting_time'] = df_with_waiting_time['waiting_time'].fillna(pd.Timedelta(seconds=0))
    
    # Filter rows where waiting time is 0
    zero_waiting_time_df = df_with_waiting_time[df_with_waiting_time['waiting_time'] <= pd.Timedelta(seconds=0)]
    
    # Group by activity name and count occurrences
    counts = zero_waiting_time_df.groupby('activity_name').size()
    
    # Get the total counts for each activity
    total_counts = df_with_waiting_time['activity_name'].value_counts()
    
    # Filter activities with count equal to total occurrences
    always_zero_waiting_time_activities = counts[counts >= total_counts[counts.index] * threshold]
    
    return always_zero_waiting_time_activities.index.tolist()


def get_prerequisites_per_activity(data, discover_parallel_work=True):
    # Check for parallel activities
    result = []
    # Group by case_id
    grouped = data.groupby('case_id')
    # Iterate over groups
    for case_id, group in grouped:
        # Call the function to check for parallel activities with a minimum of 2
        parallel_activities = check_parallel_activities(group, min_activities=2)
        
        # Extend the result list with the detected parallel activities
        result.extend([(case_id,) + tuple(activities) for activities in parallel_activities])

    parallel_activities = get_unique_parallel_activities(result=result)

    preceding_activities_dict = generate_preceding_activities_dict(data)
    # print(f"preceeding activties: {preceding_activities_dict}")
    # remove parallel activities as prerequisite for each other
    for i in range(len(parallel_activities)):
        for key, value in preceding_activities_dict.items():
            if key in parallel_activities[i]:
                par = parallel_activities[i]
                related_activities = [item for item in par if item != key]
                for j in related_activities:
                    if j in value: # TODO understand why we need to add this
                        value.remove(j)
            # join parallel activities in prerequisite of other activities to mark that they are both required
            parallels = parallel_activities[i]
            value_flattened = [item for sublist in value for item in sublist]

            if set(parallels).issubset(set(value_flattened)):
                # Check if 'parallels' is a subset of 'value_flattened'
                disjoint_part = list(set(parallels).symmetric_difference(set(value)))

                # Convert 'parallels' to a tuple before using it as a key
                parallels_tuple = tuple(parallels)

                # Use the tuple as a key in the dictionary
                preceding_activities_dict[key] = disjoint_part + [parallels_tuple]

    # remove the activity itself as prerequisite for itself
    new_dict = {key: [] for key, value in preceding_activities_dict.items()}
    for key, value in preceding_activities_dict.items():     
        for i in range(len(value)):
            if not isinstance(value[i], list):
                if not value[i] == key:
                    new_dict[key].append(value[i])

            # if value contains sublists
            else:
                if key in value[i]:
                    value[i].remove(key)
                    new_dict[key].append(value[i])
                else:
                    new_dict[key].append(value[i])

    preceding_activities_dict = new_dict

    if discover_parallel_work == False:
        parallel_activities = []

    return preceding_activities_dict, parallel_activities
    

def check_parallel_activities(group, min_activities=2):
    sorted_group = group.sort_values(by='start_timestamp')
    result = []

    # Iterate over the range of parallel activities (2 or more)
    for i in range(len(sorted_group) - min_activities + 1):
        current_end_time = sorted_group.iloc[i]['end_timestamp']
        parallel_activities = [sorted_group.iloc[j]['activity_name'] for j in range(i + 1, i + min_activities) if current_end_time > sorted_group.iloc[j]['start_timestamp']]

        if len(parallel_activities) == min_activities - 1:
            result.append(parallel_activities + [sorted_group.iloc[i]['activity_name']])

    return result

def get_unique_parallel_activities(result):
    activities_in_parallel = []
    if result:
        for item in result:
            activities = (item[1], item[2])
            activities_in_parallel.append((item[1], item[2]))
        activities_in_parallel = sorted(set(activities_in_parallel))
        activities_in_parallel = [list(set(item)) for item in activities_in_parallel]

        unique_tuples = set(tuple(sorted(inner_list)) for inner_list in activities_in_parallel)
        unique_list_of_lists = [list(unique_tuple) for unique_tuple in unique_tuples]
        unique_list_of_lists = [sublist for sublist in unique_list_of_lists if len(sublist) > 1]

        unique_list_of_lists = discover_connected_pairs(unique_list_of_lists)
    else:
        unique_list_of_lists = []

    return unique_list_of_lists

def discover_connected_pairs(activity_list):
    connections = []
    unconnected_pairs = []

    for pair in activity_list:
        connected_lists = []
        
        # Check if each entity in the current pair has other common pairs
        for entity in pair:
            entity_connections = []
            
            for other_pair in activity_list:
                if entity in other_pair and other_pair != pair:
                    entity_connections.extend(other_pair)

            connected_lists.append(entity_connections)

        # Check if both entities have a common pair
        if any(connected_lists):
            common_entities = set.intersection(*map(set, connected_lists))
            
            if common_entities:
                # Merge the pairs into a single list
                merged_pair = list(common_entities)
                merged_pair.extend(pair)
                connections.append(merged_pair)
        else:
            # If no connections, add the pair to unconnected pairs
            unconnected_pairs.append(pair)

        
    unique_set = set()

    for sublist in connections:
        tuple_sublist = tuple(sorted(sublist))
        unique_set.add(tuple_sublist)

    connections = [list(sublist) for sublist in unique_set]

    # get unconnected pairs
    long_list_connections = [item for sublist in connections for item in sublist]
    for pair in activity_list:
        for entity in pair:
            if entity in long_list_connections:
                pass
            else:
                unconnected_pairs.append(pair)

    unique_set = set()

    for sublist in unconnected_pairs:
        tuple_sublist = tuple(sorted(sublist))
        unique_set.add(tuple_sublist)

    unconnected_pairs = [list(sublist) for sublist in unique_set]

    connections.extend(unconnected_pairs)

    # add removed pairs again to connections
    for sublist_A in activity_list:
        if sublist_A not in connections:
            connections.append(sublist_A)
    # print(connections)

    return connections


def generate_preceding_activities_dict(data):
    preceding_activities_dict = {}

    # Group by case_id
    grouped = data.groupby('case_id')

    # Iterate over groups
    for case_id, group in grouped:
        sorted_group = group.sort_values(by='start_timestamp')

        # Iterate through the sorted activities
        for i in range(1, len(sorted_group)):
            current_activity = sorted_group.iloc[i]['activity_name']
            preceding_activity = sorted_group.iloc[i - 1]['activity_name']

            # Update the dictionary with the preceding activity
            if current_activity not in preceding_activities_dict:
                preceding_activities_dict[current_activity] = set()
            preceding_activities_dict[current_activity].add(preceding_activity)

    # Convert sets to lists
    preceding_activities_dict = {key: list(value) for key, value in preceding_activities_dict.items()}

    return preceding_activities_dict

def _get_times_for_extr_delays(df_train, discover_extr_delays=True):
    if discover_extr_delays == True:
        discovery_method = "complex"
        # Set-up configuration for extraneous delay discovery
        configuration = ExtraneousActivityDelaysConfiguration(
            log_ids=EventLogIDs(),
            # process_name=self.event_log.process_name,
            num_iterations=1,
            num_evaluation_simulations=3,
            training_partition_ratio=0.5,
            optimization_metric="relative_emd",
            discovery_method="eclipse-aware",
            timer_placement=TimerPlacement.BEFORE,
        )

        # print("defined configuration")

        if discovery_method == "naive":
            timers = compute_naive_extraneous_activity_delays(
                df_train,
                configuration,
                configuration.should_consider_timer,
                )
        elif discovery_method == "complex":
            timers = compute_complex_extraneous_activity_delays(
                df_train,
                configuration,
                configuration.should_consider_timer,
            )
    else:
        timers = {}

    return timers


def determine_agent_behavior_type_and_extraneous_delays(simulation_parameters, df_train, df_val, case_arrival_times_val, central_orchestration_parameter, discover_extr_delays_parameter):
    """
    Determine the agent behavior type and extraneous delays.
    """
    timers_extr = _get_times_for_extr_delays(df_train, discover_extr_delays=True)
    timers = _get_times_for_extr_delays(df_train, discover_extr_delays=False)
    # create a copy of the simulation parameters such that we can modify it without changing the original one
    simulation_parameters_copy = simulation_parameters.copy()
    
    if simulation_parameters_copy['determine_automatically']:
        # 1) simulate val log with extr delays and central orchestration
        central_orchestration = True
        sampled_case_starting_times = case_arrival_times_val
        start_timestamp = sampled_case_starting_times[0]
        sampled_case_starting_times = sampled_case_starting_times[1:]

        simulation_parameters_copy['sampled_case_starting_times'] = sampled_case_starting_times
        simulation_parameters_copy['start_timestamp'] = start_timestamp
        simulation_parameters_copy['timers'] = timers_extr
        simulation_parameters_copy['central_orchestration'] = central_orchestration
        # Create the model using the loaded data
        business_process_model = BusinessProcessModel(df_train, simulation_parameters_copy)
        # define list of cases
        case_id = 0
        case_ = Case(case_id=case_id, start_timestamp=start_timestamp) # first case
        cases = [case_]
        # Run the model for a specified number of steps
        while business_process_model.sampled_case_starting_times: # while cases list is not empty
            business_process_model.step(cases)
        simulated_log_val_extr = pd.DataFrame(business_process_model.simulated_events)

        # 2) simulate val log without extr delays and central orchestration
        simulation_parameters_copy['timers'] = timers
        sampled_case_starting_times = case_arrival_times_val
        simulation_parameters_copy['start_timestamp'] = sampled_case_starting_times[0]
        simulation_parameters_copy['sampled_case_starting_times'] = sampled_case_starting_times[1:]
        simulation_parameters_copy['central_orchestration'] = True
        # Create the model using the loaded data
        business_process_model = BusinessProcessModel(df_train, simulation_parameters_copy)
        # define list of cases
        case_id = 0
        case_ = Case(case_id=case_id, start_timestamp=start_timestamp) # first case
        cases = [case_]
        # Run the model for a specified number of steps
        while business_process_model.sampled_case_starting_times: # while cases list is not empty
            business_process_model.step(cases)
        simulated_log_val_ = pd.DataFrame(business_process_model.simulated_events)

        # 3) simulate val log with extr delays and autonomous handover
        simulation_parameters_copy['central_orchestration'] = False
        simulation_parameters_copy['timers'] = timers_extr
        sampled_case_starting_times = case_arrival_times_val
        simulation_parameters_copy['start_timestamp'] = sampled_case_starting_times[0]
        simulation_parameters_copy['sampled_case_starting_times'] = sampled_case_starting_times[1:]
        simulation_parameters_copy['transition_probabilities'] = simulation_parameters_copy['transition_probabilities_autonomous']
        simulation_parameters_copy['agent_transition_probabilities'] = simulation_parameters_copy['agent_transition_probabilities_autonomous']
        # Create the model using the loaded data
        business_process_model = BusinessProcessModel(df_train, simulation_parameters_copy)
        # define list of cases
        case_id = 0
        case_ = Case(case_id=case_id, start_timestamp=start_timestamp) # first case
        cases = [case_]
        # Run the model for a specified number of steps
        while business_process_model.sampled_case_starting_times: # while cases list is not empty
            business_process_model.step(cases)
        simulated_log_val_extr_autonomous = pd.DataFrame(business_process_model.simulated_events)

        # 4) simulate val log without extr delays and autonomous handover
        simulation_parameters_copy['timers'] = timers
        sampled_case_starting_times = case_arrival_times_val
        simulation_parameters_copy['start_timestamp'] = sampled_case_starting_times[0]
        simulation_parameters_copy['sampled_case_starting_times'] = sampled_case_starting_times[1:]
        simulation_parameters_copy['central_orchestration'] = False
        simulation_parameters_copy['transition_probabilities'] = simulation_parameters_copy['transition_probabilities_autonomous']
        simulation_parameters_copy['agent_transition_probabilities'] = simulation_parameters_copy['agent_transition_probabilities_autonomous']
        # Create the model using the loaded data
        business_process_model = BusinessProcessModel(df_train, simulation_parameters_copy)
        # define list of cases
        case_id = 0
        case_ = Case(case_id=case_id, start_timestamp=start_timestamp) # first case
        cases = [case_]
        # Run the model for a specified number of steps
        while business_process_model.sampled_case_starting_times: # while cases list is not empty
            business_process_model.step(cases)
        simulated_log_val_autonomous = pd.DataFrame(business_process_model.simulated_events)

        # 5) compute cycle time and check which one is closer to the val log
        from log_distance_measures.config import EventLogIDs
        from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance

        # Set event log column ID mapping
        event_log_ids = EventLogIDs(  # These values are stored in DEFAULT_CSV_IDS
            case="case_id",
            activity="activity_name",
            start_time="start_timestamp",
            end_time="end_timestamp",
            resource='resource'
        )

        ctdd_extr = cycle_time_distribution_distance(
                    df_val, event_log_ids,  # First event log and its column id mappings
                    simulated_log_val_extr, event_log_ids,  # Second event log and its column id mappings
                    bin_size=pd.Timedelta(hours=1)  # Bins of 1 hour
                )
        
        ctdd = cycle_time_distribution_distance(
                    df_val, event_log_ids,  # First event log and its column id mappings
                    simulated_log_val_, event_log_ids,  # Second event log and its column id mappings
                    bin_size=pd.Timedelta(hours=1)  # Bins of 1 hour
                )
        
        ctdd_extr_autonomous = cycle_time_distribution_distance(
                    df_val, event_log_ids,  # First event log and its column id mappings
                    simulated_log_val_extr_autonomous, event_log_ids,  # Second event log and its column id mappings
                    bin_size=pd.Timedelta(hours=1)  # Bins of 1 hour
                )
        
        ctdd_autonomous = cycle_time_distribution_distance(
                    df_val, event_log_ids,  # First event log and its column id mappings
                    simulated_log_val_autonomous, event_log_ids,  # Second event log and its column id mappings
                    bin_size=pd.Timedelta(hours=1)  # Bins of 1 hour
                )
        print(f"CTD extr + central: {ctdd_extr}")
        print(f"CTD without extr + central: {ctdd}")
        print(f"CTD extr + autonomous: {ctdd_extr_autonomous}")
        print(f"CTD without extr + autonomous: {ctdd_autonomous}")

        # 4) set the hyperparameter for extr_delays and the architecture
        ct_results_val = [ctdd_extr,ctdd,ctdd_extr_autonomous,ctdd_autonomous]
        best_val_result = min(ct_results_val)

        if ctdd_extr == best_val_result:
            simulation_parameters['timers'] = timers_extr
            discover_delays = True
            simulation_parameters['central_orchestration'] = True
        elif ctdd == best_val_result:
            simulation_parameters['timers'] = timers
            discover_delays = False
            simulation_parameters['central_orchestration'] = True
        elif ctdd_extr_autonomous == best_val_result:
            simulation_parameters['timers'] = timers_extr
            discover_delays = True
            simulation_parameters['central_orchestration'] = False
            simulation_parameters['transition_probabilities'] = simulation_parameters['transition_probabilities_autonomous']
            simulation_parameters['agent_transition_probabilities'] = simulation_parameters['agent_transition_probabilities_autonomous']
        elif ctdd_autonomous == best_val_result:
            simulation_parameters['timers'] = timers
            discover_delays = False
            simulation_parameters['central_orchestration'] = False
            simulation_parameters['transition_probabilities'] = simulation_parameters['transition_probabilities_autonomous']
            simulation_parameters['agent_transition_probabilities'] = simulation_parameters['agent_transition_probabilities_autonomous']
    else:
        if discover_extr_delays_parameter:
            simulation_parameters['timers'] = timers_extr
        else:
            simulation_parameters['timers'] = timers

        simulation_parameters['central_orchestration'] = central_orchestration_parameter
        if central_orchestration_parameter == False:
            simulation_parameters['transition_probabilities'] = simulation_parameters['transition_probabilities_autonomous']
            simulation_parameters['agent_transition_probabilities'] = simulation_parameters['agent_transition_probabilities_autonomous']


    # print(f"discover extr. delays: {discover_delays}")
    # print(f"central orchestration: {central_orchestration}")

    return simulation_parameters