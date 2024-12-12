import random
import math
from datetime import datetime
from mesa import Agent
import scipy.stats as st

class ContractorAgent(Agent):
    """
    One contractor agent to assign tasks using the contraction net protocol
    """
    def __init__(self, unique_id, model, activities, transition_probabilities, agent_activity_mapping):
        super().__init__(unique_id, model)
        self.activities = activities
        self.transition_probabilities = transition_probabilities
        self.agent_activity_mapping = agent_activity_mapping
        self.model = model
        self.current_activity_index = None
        self.activity_performed = False

    def step(self, scheduler, agent_keys, cases):
        method = "step"
        agent_keys = agent_keys[1:] # exclude contractor agent here as we only want to perform resource agent steps
        # 1) sort by specialism
        # bring the agents in an order to first ask the most specialized agents to not waste agent capacity for future cases -> principle of specialization
        def get_key_length(key):
            return len(self.agent_activity_mapping[key])

        # Sort the keys using the custom key function
        if isinstance(agent_keys[0], list):
            sorted_agent_keys = []
            for agent_list in agent_keys:
                sorted_agent_keys.append(sorted(agent_list, key=get_key_length))
        else:
            sorted_agent_keys = sorted(agent_keys, key=get_key_length)
        # print(f"Agents sorted by specialism: {sorted_agent_keys}")
        
        # # 2) sort by next availability
        sorted_agent_keys = self.sort_agents_by_availability(sorted_agent_keys)
            
        if self.model.central_orchestration == False:
                # 3) sort by transition probs
                current_agent = self.case.previous_agent
                if current_agent != -1:
                    current_activity = self.case.activities_performed[-1]
                    if current_agent in self.model.agent_transition_probabilities:
                        if current_activity in self.model.agent_transition_probabilities[current_agent]:
                            current_probabilities = self.model.agent_transition_probabilities[current_agent][current_activity]
                        else:
                            current_probabilities = self.model.agent_transition_probabilities[current_agent]
                    sorted_agent_keys = sorted(sorted_agent_keys, key=lambda x: current_probabilities.get(x, 0), reverse=True)

        last_possible_agent = False

        if isinstance(sorted_agent_keys[0], list):
            for agent_key in sorted_agent_keys:
                for inner_key in agent_key:
                    if inner_key == agent_key[-1]:
                        last_possible_agent = True
                    if inner_key in scheduler._agents:
                        if self.activity_performed:
                            break
                        else:
                            current_timestamp = self.get_current_timestamp(inner_key, parallel_activity=True)
                            perform_multitask = False
                            getattr(scheduler._agents[inner_key], method)(last_possible_agent, 
                                                                          parallel_activity=True, current_timestamp=current_timestamp, perform_multitask=perform_multitask)
        else:
            for agent_key in sorted_agent_keys:
                if agent_key == sorted_agent_keys[-1]:
                    last_possible_agent = True
                if agent_key in scheduler._agents:
                    if self.activity_performed:
                        break
                    else:
                        current_timestamp = self.get_current_timestamp(agent_key)
                        perform_multitask = False
                        getattr(scheduler._agents[agent_key], method)(last_possible_agent, parallel_activity=False, current_timestamp=current_timestamp, perform_multitask=perform_multitask)
        self.activity_performed = False

    def sort_agents_by_availability(self, sorted_agent_keys):
        if isinstance(sorted_agent_keys[0], list):
            sorted_agent_keys_new = []
            for agent_list in sorted_agent_keys:
                sorted_agent_keys_new.append(sorted(agent_list, key=lambda x: self.model.agents_busy_until[x]))
        else:
            sorted_agent_keys_new = sorted(sorted_agent_keys, key=lambda x: self.model.agents_busy_until[x])

        return sorted_agent_keys_new
    
    
    def get_current_timestamp(self, agent_id, parallel_activity=False):
        if parallel_activity == False:
            current_timestamp = self.case.current_timestamp
        else:
            current_timestamp = self.case.timestamp_before_and_gateway

        return current_timestamp


    def get_activity_duration(self, agent, activity):
        activity_distribution = self.model.activity_durations_dict[agent][activity]
        if activity_distribution.type.value == "expon":
            scale = activity_distribution.mean - activity_distribution.min
            if scale < 0.0:
                print("Warning! Trying to generate EXPON sample with 'mean' < 'min', using 'mean' as scale value.")
                scale = activity_distribution.mean
            activity_duration = st.expon.rvs(loc=activity_distribution.min, scale=scale, size=1)[0]
        elif activity_distribution.type.value == "gamma":
            activity_duration = st.gamma.rvs(
                pow(activity_distribution.mean, 2) / activity_distribution.var,
                loc=0,
                scale=activity_distribution.var / activity_distribution.mean,
                size=1,
            )[0]
        elif activity_distribution.type.value == "norm":
            activity_duration = st.norm.rvs(loc=activity_distribution.mean, scale=activity_distribution.std, size=1)[0]
        elif activity_distribution.type.value == "uniform":
            activity_duration = st.uniform.rvs(loc=activity_distribution.min, scale=activity_distribution.max - activity_distribution.min, size=1)[0]
        elif activity_distribution.type.value == "lognorm":
            pow_mean = pow(activity_distribution.mean, 2)
            phi = math.sqrt(activity_distribution.var + pow_mean)
            mu = math.log(pow_mean / phi)
            sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
            activity_duration = st.lognorm.rvs(sigma, loc=0, scale=math.exp(mu), size=1)[0]
        elif activity_distribution.type.value == "fix":
            activity_duration = activity_distribution.mean


        return activity_duration
    
    def sample_starting_activity(self,):
        """
        sample the activity that starts the case based on the frequency of starting activities in the train log
        """
        start_activities = self.model.data.groupby('case_id')['activity_name'].first().tolist()
        # Count occurrences of each entry and create a dictionary
        start_count = {}
        for entry in start_activities:
            if entry in start_count:
                start_count[entry] += 1
            else:
                start_count[entry] = 1

        for key, value in start_count.items():
            start_count[key] = value / len(self.model.data['case_id'].unique())

        sampled_activity = random.choices(list(start_count.keys()), weights=start_count.values(), k=1)[0]

        return sampled_activity
    
    def check_for_other_possible_next_activity(self, next_activity):
        possible_other_next_activities = []
        for key, value in self.model.prerequisites.items():
            for i in range(len(value)):
                # if values is a single list, then only ONE of the entries must have been performed already (XOR gateway)
                if not isinstance(value[i], list):
                    if value[i] == next_activity:
                        possible_other_next_activities.append(key)
                # if value contains sublists, then all of the values in the sublist must have been performed (AND gateway)
                else:
                    # if current next_activity contained in prerequisites
                    if any(next_activity in sublist for sublist in value[i]):
                        # if all prerequisites are fulfilled
                        if all(value_ in self.case.activities_performed for value_ in value[i]):
                            possible_other_next_activities.append(key)

        return possible_other_next_activities
    

    
    def get_potential_agents(self, case):
        """
        check if there already happened activities in the current case
            if no: current activity is usual start activity
            if yes: current activity is the last activity of the current case
        """
        self.case = case
        case_ended = False

        current_timestamp = self.case.current_timestamp
        # self.case.potential_additional_agents = []

        if case.get_last_activity() == None: # if first activity in case
            # sample starting activity
            sampled_start_act = self.sample_starting_activity()
            current_act = sampled_start_act
            self.new_activity_index = self.activities.index(sampled_start_act)
            next_activity = sampled_start_act
            # print(f"start activity: {next_activity}")
        else:
            current_act = case.get_last_activity()
            self.current_activity_index = self.activities.index(current_act)

            prefix = self.case.activities_performed

            if self.model.central_orchestration:
                while tuple(prefix) not in self.transition_probabilities.keys():
                    prefix = prefix[1:]
                # Extract activities and probabilities
                # print(self.transition_probabilities[tuple(prefix)])
                activity_list = list(self.transition_probabilities[tuple(prefix)].keys())
                probabilities = list(self.transition_probabilities[tuple(prefix)].values())
                # Sample an activity based on the probabilities
                next_activity = random.choices(activity_list, weights=probabilities, k=1)[0]
                self.new_activity_index = self.activities.index(next_activity)
            else:
                while tuple(prefix) not in self.transition_probabilities.keys() or self.case.previous_agent not in self.transition_probabilities[tuple(prefix)].keys():
                    prefix = prefix[1:]
                # Extract activities and probabilities
                # print(self.transition_probabilities[tuple(prefix)])
                activity_list = list(self.transition_probabilities[tuple(prefix)][self.case.previous_agent].keys())
                probabilities = list(self.transition_probabilities[tuple(prefix)][self.case.previous_agent].values())
                # Sample an activity based on the probabilities
                next_activity = random.choices(activity_list, weights=probabilities, k=1)[0]
                self.new_activity_index = self.activities.index(next_activity)


            # check if next activity is zzz_end
            if next_activity == 'zzz_end':
                potential_agents = None
                case_ended = True
                return potential_agents, case_ended#, None, None
            
            if self.model.discover_parallel_work:
                # check if next activity is allowed by looking at prerequisites
                activity_allowed = False
                for key, value in self.model.prerequisites.items():
                    if next_activity == key:
                        for i in range(len(value)):
                            # if values is a single list, then only ONE of the entries must have been performed already (XOR gateway)
                            if not isinstance(value[i], list):
                                if value[i] in self.case.activities_performed:
                                    activity_allowed = True
                                    break
                            # if value contains sublists, then all of the values in the sublist must have been performed (AND gateway)
                            else:
                                if all(value_ in self.case.activities_performed for value_ in value[i]):
                                    activity_allowed = True
                                    break
                # if activity is not specified as prerequisite, additionally check if it is a parallel one to the last activity and thus actually can be performed
                if activity_allowed == False:
                    for i in range(len(self.model.parallel_activities)):
                        if next_activity in self.model.parallel_activities[i]:
                            if self.case.activities_performed[-1] in self.model.parallel_activities[i]:
                                activity_allowed = True
            else:
                activity_allowed = True

            # additionally check if new activity was already performed
            number_occurence_of_next_activity = self.case.activities_performed.count(next_activity)
            number_occurence_of_next_activity += 1 # add 1 as it would appear one more time in the next step
            if number_occurence_of_next_activity > self.model.max_activity_count_per_case[next_activity]:
                activity_allowed = False
                # check if there is another possible activity that can be performed
                # go through prerequisites and check for which act the current next_activity is a prerequisite for
                # if it is one, then check if this other activity can be performed
                possible_other_next_activities = self.check_for_other_possible_next_activity(next_activity)
                if len(possible_other_next_activities) > 0:
                    next_activity = random.choice(possible_other_next_activities)
                    self.new_activity_index = self.activities.index(next_activity)
                    # print(f"Changed next activity to {next_activity}")
                    activity_allowed = True
                    # check if next activity is zzz_end
                    if next_activity == 'zzz_end':
                        potential_agents = None
                        case_ended = True
                        return potential_agents, case_ended
                # to avoid that simulation does not terminate
                else:
                    activity_allowed = True

            if activity_allowed == False:
                # print(f"case_id: {self.case.case_id}: Next activity {next_activity} not allowed from current activity {current_act} with history {self.case.activities_performed}")
                # TODO: do something when activity is not allowed
                potential_agents = None
                return potential_agents, case_ended#, [], []
            else:
                pass
                # print(f"case_id: {self.case.case_id}: Next activity {next_activity} IS ALLOWED from current activity {current_act} with history {self.case.activities_performed}")
        
        # check which agents can potentially perform the next task
        potential_agents = [key for key, value in self.agent_activity_mapping.items() if any(next_activity == item for item in value)]
        # also add contractor agent to list as he is always active
        potential_agents.insert(0, 9999)


        return potential_agents, case_ended