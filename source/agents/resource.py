import pandas as pd
from datetime import datetime
import pytz
from mesa import Agent

from source.utils import sample_from_distribution

class ResourceAgent(Agent):
    """
    One agent for each resource in the event log
    """
    def __init__(self, unique_id, model, resource, timer, contractor_agent=None):
        super().__init__(unique_id, model)
        self.resource = resource
        self.model = model
        self.is_busy = False
        self.is_busy_until = None
        self.contractor_agent = contractor_agent
        self.agent_type = next((role for role, ids in self.model.roles.items() if self.resource in ids['agents']), None)
        if self.resource in self.model.calendars.keys():
            self.calendar = self.model.calendars[self.resource].intervals_to_json()
        else:
            self.calendar = next((ids['calendar'] for role, ids in self.model.roles.items() if self.resource in ids['agents']), None)
        self.timer = timer
        self.occupied_times = []

    def step(self, last_possible_agent=False, parallel_activity=False, current_timestamp=None, perform_multitask=False):
        if not parallel_activity:
            self.contractor_agent.current_activity_index = self.contractor_agent.new_activity_index
            activity = self.contractor_agent.activities[self.contractor_agent.current_activity_index]
            current_timestamp = current_timestamp

            activity_duration = self.contractor_agent.get_activity_duration(self.resource, activity)

            self.perform_task(current_timestamp, activity_duration, activity, last_possible_agent, perform_multitask=perform_multitask)

            self.contractor_agent.case.timestamp_before_and_gateway = current_timestamp
        else:
            additional_activity_index = self.model.additional_activity_index
            activity = self.contractor_agent.case.additional_next_activities[additional_activity_index]
            current_timestamp = current_timestamp
            activity_duration = self.contractor_agent.get_activity_duration(self.resource, activity)

            self.perform_task(current_timestamp, activity_duration, activity, last_possible_agent, additional_act=True, 
                              additional_agent_counter=additional_activity_index, perform_multitask=perform_multitask)

    def perform_task(self, current_timestamp, activity_duration, activity, last_possible_agent, additional_act=False, additional_agent_counter=0, perform_multitask=False):
        # print(f"current timestamp: {current_timestamp}")
        if activity in self.timer.keys():
            waiting_time_distribution = self.timer[activity]
            waiting_time = sample_from_distribution(distribution=waiting_time_distribution)
        else:
            waiting_time = 0
        current_timestamp += pd.Timedelta(seconds=waiting_time)
        # print(f"activity duration: {activity_duration}")


        # check if the activity can be performed in multi-tasking style
        if activity in self.model.activities_without_waiting_time:
            perform_multitask = True

        # check if agent is busy after updating availability status
        if self.is_busy_until != None:
            if self.is_busy_until <= current_timestamp:
                self.is_busy = False

        if self.is_occupied(current_timestamp, activity_duration) == False or perform_multitask == True:# or activity_duration == 0.0:
            # check if current timestamp lies within the availability of the agent
            if self.is_within_calendar(current_timestamp, activity_duration) or perform_multitask == True:
                # set as busy
                
                if activity_duration != 0.0:
                    self.occupied_times.append((current_timestamp, current_timestamp + pd.Timedelta(seconds=activity_duration)))

                    self.is_busy_until = current_timestamp + pd.Timedelta(seconds=activity_duration)
                    self.is_busy = True
                    self.model.agents_busy_until[self.resource] = self.is_busy_until
                else:
                    pass
                # advance current timestamp
                self.contractor_agent.case.current_timestamp = current_timestamp + pd.Timedelta(seconds=activity_duration)
                # add activity to case list to keep track of performed activities per case
                self.contractor_agent.case.add_activity_to_case(activity)
                # print(f"Activity performed: {activity}")

                # set that activity is performed
                self.contractor_agent.activity_performed = True

                self.contractor_agent.case.previous_agent = self.resource

    
                # remove activity from additional activities
                if additional_act == True:
                    index_to_delete = self.contractor_agent.case.additional_next_activities.index(activity)
                    self.contractor_agent.case.additional_next_activities.pop(index_to_delete)


                self.model.simulated_events.append({'case_id': self.contractor_agent.case.case_id, 
                                    'agent': self.resource, 
                                    'activity_name': activity,
                                    'start_timestamp': current_timestamp,
                                    'end_timestamp': self.contractor_agent.case.current_timestamp,
                                    'TimeStep': self.model.schedule.steps,
                                    })
            else:
                # print(f"#######agent {self.resource} is free but time not within calendar")
                if last_possible_agent: # then increase timer by x seconds to try to get an available agent later
                    # move timestamp until agent is available again according to calendar
                    self.contractor_agent.case.current_timestamp = self.set_time_to_next_availability_when_not_in_calendar(current_timestamp, activity_duration)
                    # print(f"set timestamp until agent is available again: {self.contractor_agent.case.current_timestamp}")
                    if additional_act == True:
                        self.model.additional_activity_index += 1
                else:
                    pass # first try if one of the other possible agents is available
        else:
            # print(f"agent {self.resource} is busy when trying to perform task {activity} until {self.is_busy_until}")
            if last_possible_agent: # then increase timer by x seconds to try to get an available agent later
                self.set_current_time_to_next_available_slot()
                if additional_act == True:
                    self.model.additional_activity_index += 1
            else:
                pass # first try if one of the other possible agents is available

    def is_occupied(self, new_start, activity_duration):
        new_end = new_start + pd.Timedelta(seconds=activity_duration)
        for start, end in self.occupied_times:
            if new_start < end and new_end > start:
                return True  # There is an overlap
        return False  # No overlap found
    
    def get_current_number_multitasking(self, new_start, activity_duration):
        new_end = new_start + pd.Timedelta(seconds=activity_duration)
        number_multitask = 1 # 1 because we have to add the current activity as well
        for start, end in self.occupied_times:
            if new_start < end and new_end > start:
                number_multitask += 1

        return number_multitask
    
    def set_current_time_to_next_available_slot(self,):
        new_time_set = False
        current_time = self.contractor_agent.case.current_timestamp
        self.occupied_times = sorted(self.occupied_times, key=lambda x: x[1])
        for start, end in self.occupied_times:
            if end > current_time:
                self.contractor_agent.case.current_timestamp = end
                new_time_set = True
                # print(f"moved time to: {end}")
                break
        if new_time_set == False:
            self.contractor_agent.case.current_timestamp += pd.Timedelta(seconds=60)
            # print(f"moved time by 60 seconds")
        
    def is_within_calendar(self, current_timestamp, activity_duration):
        """
        check if the current timestamp + activtiy duration is within the availability calendar of the agent
        param current_timestamp: datetime object
        param activity_duration: duration of next activity in seconds

        return True or False
        """
        day_of_week = current_timestamp.strftime('%A').upper()
        end_time_of_activity = current_timestamp + pd.Timedelta(seconds=activity_duration)
        # print(f"expected end time of activity: {end_time_of_activity}")

        for entry in self.calendar:
            if entry['from'] == day_of_week:
                # Try parsing with the first format '%H:%M:%S'
                try:
                    begin_time = datetime.strptime(entry['beginTime'], '%H:%M:%S')
                except ValueError:
                    # If the first format fails, try the second format '%H:%M:%S.%f'
                    begin_time = datetime.strptime(entry['beginTime'], '%H:%M:%S.%f')
                try:
                    end_time = datetime.strptime(entry['endTime'], '%H:%M:%S')
                except ValueError:
                    # If the first format fails, try the second format '%H:%M:%S.%f'
                    end_time = datetime.strptime(entry['endTime'], '%H:%M:%S.%f')

                end_time_current_activity = datetime.combine(end_time_of_activity.date(), end_time_of_activity.time())
                end_time_current_activity = end_time_current_activity.time()

                begin_time_current_activity = datetime.combine(current_timestamp.date(), current_timestamp.time())
                begin_time_current_activity = begin_time_current_activity.time()

                # if begin_time.time() <= end_time_current_activity <= end_time.time():
                if begin_time.time() <= begin_time_current_activity:
                    if end_time_current_activity <= end_time.time():
                        return True

        return False
    
    def set_time_to_next_availability_when_not_in_calendar(self, current_timestamp, activity_duration):
        """
        Set current timestamp to the next availability according to resource calendar. 
        E.g., if current_timestamp=04:30, set it to 08:00
        """
        current_timestamp += pd.Timedelta(seconds=activity_duration)
        current_day = current_timestamp.strftime('%A').upper()  # Get the current day of the week
        # print(f"day of current timestamp: {current_day}")
        current_time = current_timestamp.time()

        # Find the working hours for the current day
        working_hours = None
        for day_schedule in self.calendar:
            if day_schedule['from'] == current_day and day_schedule['to'] == current_day:
                working_hours = (day_schedule['beginTime'], day_schedule['endTime'])
                break

        # If no working hours are defined for the current day, find the next working day
        if working_hours is None:
            next_working_day = current_timestamp
            next_working_hours = None

            while next_working_day.strftime('%A').upper() not in [day_schedule['from'] for day_schedule in self.calendar]:
                next_working_day += pd.Timedelta(days=1)

            for day_schedule in self.calendar:
                if day_schedule['from'] == next_working_day.strftime('%A').upper() and day_schedule['to'] == next_working_day.strftime('%A').upper():
                    next_working_hours = (day_schedule['beginTime'], day_schedule['endTime'])
                    break

            if next_working_hours is None:
                raise ValueError(f"No working hours defined for agent {self.resource} on any day.")

            # Set the timestamp to the beginning of the working hours on the next working day
            try:
                next_possible_timestamp = datetime.combine(next_working_day, datetime.strptime(next_working_hours[0], '%H:%M:%S').time())
            except ValueError:
                next_possible_timestamp = datetime.combine(next_working_day, datetime.strptime(next_working_hours[0], '%H:%M:%S%f').time())

            next_possible_timestamp = pd.Timestamp(next_possible_timestamp, tzinfo=pytz.UTC)
        else:
            # Parse the working hours to datetime objects
            try:
                begin_time = datetime.strptime(working_hours[0], '%H:%M:%S').time()
            except ValueError:
                begin_time = datetime.strptime(working_hours[0], '%H:%M:%S.%f').time()
            try:
                end_time = datetime.strptime(working_hours[1], '%H:%M:%S').time()
            except ValueError:
                end_time = datetime.strptime(working_hours[1], '%H:%M:%S.%f').time()

            # Check if the current timestamp is beyond the working hours of the current day
            if current_time > end_time:
                next_working_day = current_timestamp + pd.Timedelta(days=1)
            else:
                next_working_day = current_timestamp

            # Find the next available working day and working hours
            next_working_hours = None

            while next_working_day.strftime('%A').upper() not in [day_schedule['from'] for day_schedule in self.calendar]:
                next_working_day += pd.Timedelta(days=1)

            for day_schedule in self.calendar:
                if day_schedule['from'] == next_working_day.strftime('%A').upper() and day_schedule['to'] == next_working_day.strftime('%A').upper():
                    next_working_hours = (day_schedule['beginTime'], day_schedule['endTime'])
                    break

            if next_working_hours is None:
                raise ValueError(f"No working hours defined for agent {self.resource} on any day.")

            # Set the timestamp to the beginning of the working hours on the next working day
            try:
                next_possible_timestamp = datetime.combine(next_working_day, datetime.strptime(next_working_hours[0], '%H:%M:%S').time())
            except ValueError:
                next_possible_timestamp = datetime.combine(next_working_day, datetime.strptime(next_working_hours[0], '%H:%M:%S%f').time())
            next_possible_timestamp = pd.Timestamp(next_possible_timestamp, tzinfo=pytz.UTC) #tz='UTC')

        return next_possible_timestamp