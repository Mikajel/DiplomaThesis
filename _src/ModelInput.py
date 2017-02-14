# This file is dedicated to classes constructing objects fed to the neural network
from numpy import dtype
from DataHandler import parse_timestamp
from DataHandler import normalize
from DataHandler import timestamp_to_datetime
from DataHandler import timestamp_diff
from DataHandler import sigmoid
from datetime import datetime
import math


# Represents objects fed to the Item Feature of the neural network
class ItemObject:

    def __init__(self, item):

        self.month = None
        self.weekday = None
        self.monthday = None
        self.daytime_minutes = None
        self.total_item_clicks = None
        self.total_item_buys = None
        self.clicktime_item_price = None
        self.max_item_price = None
        self.min_item_price = None
        self.item_on_sales = None
        self.duration_of_item_click = None
        self.category = None


        # timestamp_array = parse_timestamp(numpy_array['timestamp'])
        #
        # self.month = timestamp_array[1]
        #
        # date_string = str(timestamp_array[0] + timestamp_array[1] + timestamp_array[2])
        #
        # print('Timestring: ' + date_string)
        # date = datetime.strptime(date_string, '%Y%m%d')
        # self.day_of_week = date.weekday()
        # print(date.date())
        # print(self.day_of_week)
        #
        # # Time of the day, defined as minutes
        # self.time_of_day = timestamp_array[3]*60 + timestamp_array[4]
        #
        # self.price = numpy_array['price']
        # self.category = numpy_array['category']


# Class holding an information about items with specific ID
# Holds summary data to be later used for ItemObject object construction
# Not supposed to hold row-specific data (like actual item price)
class Item:

    def __init__(self, click_list):

        self.clicks_rows = None
        self.clicks_count = None
        self.buys_rows = None
        self.buys_count = None
        self.min_price = None
        self.max_price = None


# Represents objects fed to the Session Feature of the neural network
class SessionObject:

    def __init__(self, session):

        # Required fields
        self.id_session = session.id_session
        self.number_of_clicks = session.clicks_count
        self.unique_items_count = session.unique_item_count()
        self.avg_clicks_per_item = session.clicks_count/self.unique_items_count
        self.min_session_clicks, self.max_session_clicks = session.lowest_highest_item_clicks()

        # no idea
        # self.avg_click_rate = None

        self.session_duration = session.duration
        self.avg_click_duration = session.avg_clicktime()
        self.min_click_duration, self.max_click_duration = session.min_max_clicktime()

        # self.clicks_per_category = []
        # self.current_month_buy_count = None
        # self.current_month_buy_click_ratio = None
        # self.current_month_weekday_buy_count = None
        # self.current_month_weekday_buy_click_ratio = None
        # self.current_month_monthday_buy_count = None
        # self.current_month_monthday_buy_click_ratio = None
        # self.weekday_hour_buy_count = None
        # self.weekday_hour_buy_click_ratio = None

    # Create SessionObject array from Session list
    @staticmethod
    def create_session_object_list(session_list):

        session_object_list = []

        for element in session_list:

            session_object_list.append(SessionObject(element))

        return session_object_list
    
    
    # Create a list of input vectors
    @staticmethod
    def create_input_vectors(session_object_list, info=False):

        input_vectors = []
        shortest, longest = SessionObject.shortest_longest_session(session_object_list)

        for element in session_object_list:

            vector = element.create_input_vector(shortest, longest, info=info)
            input_vectors.append(vector)

        print ('Created %d input vectors from sessions\n' % len(input_vectors))

        return input_vectors

    # Returns vector of values to feed to neural network
    # Returns None object if session is longer than 30 clicks
    def create_input_vector(self, shortest_duration, longest_duration, info=False):

        vector = []

        # TODO: Brainstorm about this. Values are <0;1>, but most sessions will have low values
        clicks_normalized = self.number_of_clicks/30.0
        vector.append(clicks_normalized)

        unique_items_normalized = self.unique_items_count/10.0
        vector.append(unique_items_normalized)

        avg_item_clicks_normalized = self.avg_clicks_per_item/10.0
        vector.append(avg_item_clicks_normalized)

        min_item_clicks_normalized = self.min_session_clicks/10.0
        vector.append(min_item_clicks_normalized)

        max_item_clicks_normalized = self.max_session_clicks/10.0
        vector.append(max_item_clicks_normalized)

        duration_normalized = normalize(self.session_duration, shortest_duration, longest_duration)
        vector.append(duration_normalized)

        avg_click_duration_normalized = normalize(self.avg_click_duration, self.min_click_duration, self.max_click_duration)
        vector.append(avg_click_duration_normalized)

        # FIXME: achieve sigmoid going from 0 to 1 instead of 0.5 to 1
        # <0.5 comes from negative input to sigmoid
        min_click_duration_normalized = sigmoid(self.min_click_duration/100.0)
        vector.append(min_click_duration_normalized)

        max_click_duration_normalized = sigmoid(self.max_click_duration/100.0)
        vector.append(max_click_duration_normalized)

        if info:
            print ('number of clicks: %d' % self.number_of_clicks)
            print ('Vector: ')
            for element in vector:
                print ('%.2f' % element)

        return vector

    # Returns longest and shortest session duration in a list for a normalization
    @staticmethod
    def shortest_longest_session(session_object_list):

        shortest = 100
        longest = 0

        for session_object in session_object_list:

            if session_object.session_duration < shortest:
                shortest = session_object.session_duration

            if session_object.session_duration > longest:
                longest = session_object.session_duration

        return shortest, longest

    # Prints summary of SessionItem object to the standard output
    def info_print_session_object(self):

        print ('SessionObject: ')
        print ('clicks: %d, unique items: %d, avg click per item: %d, min item clicks: %d, max item clicks: %d,' \
              'session time: %d, avg click time: %d, min click time: %d, max click time: %d' \
              % (self.number_of_clicks,
                 self.unique_items_count,
                 self.avg_clicks_per_item,
                 self.min_session_clicks,
                 self.max_session_clicks,
                 self.session_duration,
                 self.avg_click_duration,
                 self.min_click_duration,
                 self.max_click_duration))


# Class used for construction of Session object -> pre-object used in construction of SessionObject object
class Session:

    def __init__(self, id_session,  click_list):

        self.id_session = id_session
        self.clicks_rows = click_list
        self.clicks_count = len(click_list)
        self.duration = self.session_duration()

    # TODO: Here we go again...
    # FIXME: This shit will take DAYS to compute... RIP RAM, RIP FIIT server, Rest in pepperoni, fettuccine and macaroni
    # TODO: Possible use of yield? Maybe baby...
    # Iterates through click dataset and sorts sessions into dictionary
    # Dictionary key is 'id_session' and value is a list of lines of session.
    @staticmethod
    def create_sessions_from_clicks(train_click_dataset):

        index = 0
        size = len(train_click_dataset)
        sessions_clicks = {}
        sessions = []

        # Create a dictionary of all sessions as keys and its clicks as an array values
        while index < size:

            id_session = train_click_dataset[index]['id_session']
            actual_session = []

            while (index < size) and (train_click_dataset[index]['id_session'] == id_session):
                if math.fmod(index, 25000) == 0:
                    print ('Currently iterating through index %d' % index)
                actual_session.append(train_click_dataset[index])
                index += 1

            if id_session not in sessions_clicks.keys():

                sessions_clicks[id_session] = actual_session
            else:
                print('Warning: Duplicate entry into supposedly non-conflict dictionary')
                print('Possible indication of sessions not being grouped in file')

        for id_session in sessions_clicks.keys():

            session = Session(id_session, sessions_clicks[id_session])
            sessions.append(session)
            if math.fmod(len(sessions), 5000) == 0:
                print ('Created %d sessions\n' % len(sessions))

        return sessions

    # Returns number of unique items in session from click list
    def unique_item_count(self):

        unique_list = []

        for row in self.clicks_rows:

            if row['id_item'] not in unique_list:
                unique_list.append(row['id_item'])

        return len(unique_list)

    # Count lowest and highest number of clicks on item in this session
    def lowest_highest_item_clicks(self):

        click_list = {}

        for row in self.clicks_rows:

            if row['id_item'] in click_list.keys():
                click_list['id_item'] += 1

            else:
                click_list['id_item'] = 1

        max_key, max_value = max(click_list.items(), key=lambda x: x[1])
        min_key, min_value = min(click_list.items(), key=lambda x: x[1])

        return min_value, max_value

    # Returns how many seconds session could have had
    # Counts difference between first and last click + 1*bonus
    # Bonus is defined as average time per click in session
    # Bonus is added because we do not know how long last clicks-view takes
    def session_duration(self):

        time_begin = self.clicks_rows[0]['timestamp']
        time_end = self.clicks_rows[-1]['timestamp']

        recorded_time = timestamp_diff(time_begin, time_end)
        bonus = recorded_time / self.clicks_count
        final_time = recorded_time + bonus

        return final_time

    # Return an average duration of a session click in seconds
    # Return full duration if session only has one click
    def avg_clicktime(self):

        time_begin = self.clicks_rows[0]['timestamp']
        time_end = self.clicks_rows[-1]['timestamp']

        recorded_time = timestamp_diff(time_begin, time_end)

        if recorded_time == 0:
            return self.duration

        else:
            return recorded_time / (self.clicks_count - 1)

    # Returns minimum and maximum click duration
    # For session with a single click, count baseline 5 seconds
    def min_max_clicktime(self):

        baseline_time = 5
        min_duration = 1000
        max_duration = 0

        if self.clicks_count == 1:
            return baseline_time, baseline_time

        else:
            for index in range(0, self.clicks_count - 1):

                actual_diff = timestamp_diff(
                    self.clicks_rows[index]['timestamp'],
                    self.clicks_rows[index+1]['timestamp']
                )

                if actual_diff < min_duration:
                    min_duration = actual_diff

                if actual_diff > max_duration:
                    max_duration = actual_diff

        return min_duration, max_duration

