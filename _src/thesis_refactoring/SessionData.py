# This file is dedicated to classes constructing objects fed to the neural network
from numpy import dtype
import DataHandler
from DataHandler import *
from datetime import datetime
import math
from ItemData import ItemData


# Class used for collecting aggregated info of a single session for later purposes
# Notes:
#     StartTime refers to the timestamp of the first click in given session
#     EndTime refers to the timestamp of the last click in given session
#     Duration has added time of one average click duration(see method session_duration),
#         therefore duration > endtime - starttime
class SessionData:

    default_clicktime = 5

    def __init__(self, click_list):

        self.id_session = click_list[0]['id_session']
        self.starttime = click_list[0]['timestamp']
        self.endtime = click_list[-1]['timestamp']
        self.number_of_clicks = len(click_list)
        self.session_duration = self.session_duration()
        self.avg_click_duration = (self.session_duration / self.number_of_clicks)

        self.create_item_dictionary(click_list)

        # count the duration (in seconds) for every click except for the last one
        # the last click does not have endtime, so we count average of session clicks durations
        # then add time to the corresponding ItemData in a dictionary "items"
        # this way, if item is clicked multiple times and clicks, they add up to a single item

        self.min_session_clicks, self.max_session_clicks = self.min_max_clicks_per_item()
        self.min_click_duration, self.max_click_duration = SessionData.min_max_click_duration(click_list)
        self.unique_items = len(self.items)
        self.avg_clicks_per_item = (self.number_of_clicks / len(self.items))

        # this has to be initialized through initialize_item_buys method
        self.bought_items = []

    def create_item_dictionary(self, click_list):

        self.items = {}

        # if there is only a single click in a session
        if self.number_of_clicks == 1:

            actual_item = click_list[0]['id_item']

            self.items[actual_item] = ItemData(
                click_list[0]['id_item'],
                click_list[0]['category'],
                1,
                SessionData.default_clicktime
            )
            return

        for index in range(0, self.number_of_clicks - 1):
            actual_item = click_list[index]['id_item']

            actual_timestamp = click_list[index]['timestamp']
            next_timestamp = click_list[index + 1]['timestamp']

            actual_time = DataHandler.timestamp_diff(next_timestamp, actual_timestamp)

            # If item does not exist in session, add new item
            if actual_item not in self.items.keys():
                self.items[actual_item] = ItemData(
                    click_list[index]['id_item'],
                    click_list[index]['category'],
                    1,
                    actual_time
                )
            # If item already exists, update its values
            else:
                self.items[actual_item].update_item(1, actual_time)

        # manually one more time for the last indexed item without clicktime
        actual_item = click_list[-1]['id_item']
        actual_time = (DataHandler.timestamp_diff(click_list[0]['timestamp'], click_list[-1]['timestamp'])/(len(click_list)-1))

        if actual_item not in self.items.keys():
            self.items[actual_item] = ItemData(
                actual_item,
                click_list[-1]['category'],
                1,
                actual_time
            )

        else:
            self.items[actual_item].update_item(1, actual_time)



    # Walks through buy dataset and writes to every session a list of items bought in the session
    # List consists of 'id_item' of every bought item
    # Quantity of buys is not measured
    @staticmethod
    def initialize_item_buys(session_dict, dataset_buys, info=False):

        for row in dataset_buys:
            # only add to sessions we identified in clicks dataset
            id_session = row['id_session']
            id_item = row['id_item']

            if id_session in session_dict.keys():
                session_dict[id_session].bought_items.append(id_item)
                if info:
                    print('Item %d bought in Session %d' % (id_item, id_session))
            else:
                pass

        print('Initialized bought items into Sessions')

    # Return an average duration of a session click in seconds
    # Return full duration if session only has one click
    def avg_clicktime(self):

        time_begin = self.starttime
        time_end = self.endtime

        recorded_time = DataHandler.timestamp_diff(time_begin, time_end)

        if recorded_time == 0:
            return self.session_duration

        else:
            return recorded_time / (self.number_of_clicks - 1)

    # Returns how many seconds session could have had
    # Session with a single click return baseline of 5 seconds
    # Counts difference between first and last click + 1*bonus
    # Bonus is defined as average time per click in session
    # Bonus is added because we do not know how long last clicks-view takes
    def session_duration(self):

        time_begin = self.starttime
        time_end = self.endtime

        recorded_time = DataHandler.timestamp_diff(time_begin, time_end)
        if recorded_time == 0:
            return 5

        bonus = recorded_time / self.number_of_clicks
        final_time = recorded_time + bonus

        return final_time

    # Counts average number of clicks per item in given session
    def avg_item_clicks(self):

        return self.number_of_clicks/len(self.items)

    # Method summing number of clicks of each item
    def count_session_clicks(self):

        sum = 0

        for key in self.items.keys():
            sum += self.items[key].clicks_count

        return sum

    # Returns maximal and minimal number of clicks on unique items in given session.
    def min_max_clicks_per_item(self):

        min_clicks = 99
        max_clicks = 0

        for key in self.items.keys():
            actual_clicks = self.items[key].number_of_clicks

            if actual_clicks < min_clicks:
                min_clicks = actual_clicks

            if actual_clicks > max_clicks:
                max_clicks = actual_clicks

        return min_clicks, max_clicks

    # Method returns shortest and longest click in session
    # For session with a single click, 5 second default is assigned
    # We do not need to count the last click time, since it is counted as average and will never reach minimum or maximum
    @staticmethod
    def min_max_click_duration(click_list):

        min_duration = 99
        max_duration = 0

        if len(click_list) == 1:
            return 5, 5

        else:
            for index in range(0, len(click_list) - 1):
                actual_time = DataHandler.timestamp_diff(
                    click_list[index]['timestamp'],
                    click_list[index + 1]['timestamp']
                )

                if actual_time < min_duration:
                    min_duration = actual_time

                if actual_time > max_duration:
                    max_duration = actual_time

        return min_duration, max_duration

    # Create a single input vector from session
    def create_session_input_vector(self, shortest_duration, longest_duration, info=False):

        vector = []

        # TODO: Brainstorm about this. Values are <0;1>, but most sessions will have low values
        clicks_normalized = self.number_of_clicks / 30.0
        vector.append(clicks_normalized)

        unique_items_normalized = self.unique_items / 10.0
        vector.append(unique_items_normalized)

        avg_item_clicks_normalized = self.avg_clicks_per_item / 10.0
        vector.append(avg_item_clicks_normalized)

        min_item_clicks_normalized = self.min_session_clicks / 10.0
        vector.append(min_item_clicks_normalized)

        max_item_clicks_normalized = self.max_session_clicks / 10.0
        vector.append(max_item_clicks_normalized)

        duration_normalized = DataHandler.normalize(self.session_duration, shortest_duration, longest_duration)
        vector.append(duration_normalized)

        avg_click_duration_normalized = DataHandler.normalize(self.avg_click_duration, self.min_click_duration,
                                                  self.max_click_duration)
        vector.append(avg_click_duration_normalized)

        # FIXME: achieve sigmoid going from 0 to 1 instead of 0.5 to 1
        # <0.5 comes from negative input to sigmoid
        min_click_duration_normalized = DataHandler.sigmoid(self.min_click_duration / 100.0)
        vector.append(min_click_duration_normalized)

        max_click_duration_normalized = DataHandler.sigmoid(self.max_click_duration / 100.0)
        vector.append(max_click_duration_normalized)

        if info:
            print('number of clicks: %d' % self.number_of_clicks)
            print('Vector: ')
            for element in vector:
                print('%.2f' % element)

        return vector

    # Create an output vector for a session
    # For buy session, return [1, 0], else return [0, 1]
    # Neural Network then uses classification into buy and non-buy classes
    # 1st output neuron is a buy class and 2nd is a non-buy class
    def create_session_output_vector(self, info=False):

        if len(self.bought_items) > 0:
            if info:
                print('Label for a buy session.')
            return [1, 0]

        else:
            if info:
                print('Label for a non-buy session.')
            return [0, 1]

    # Create a lists of input and output vectors with corresponding indices
    # Cycle through all session using create_input_vector and create_output_vector methods
    @staticmethod
    def create_session_input_output_vectors(session_dict, info=False):

        input_vectors = []
        output_vectors = []
        source_sessions = []

        shortest_duration, longest_duration = SessionData.shortest_longest_session(session_dict)

        # turn dictionary of sessions into list of input vectors, output vectors and source sessions
        for key in session_dict.keys():
            input_vector = session_dict[key].create_session_input_vector(
                shortest_duration,
                longest_duration,
                info=info
            )
            output_vector = session_dict[key].create_session_output_vector(info=info)

            # Append vectors and source
            input_vectors.append(input_vector)
            output_vectors.append(output_vector)
            source_sessions.append(session_dict[key])

        print('Created %d input and %d output vectors from %d sessions\n'
              % (len(input_vectors), len(output_vectors), len(source_sessions)))

        return input_vectors, output_vectors, source_sessions

    # Returns longest and shortest session duration in a dictionary for a normalization
    @staticmethod
    def shortest_longest_session(session_dict):

        shortest = 1000
        longest = 0

        for key in session_dict.keys():

            if session_dict[key].session_duration < shortest:
                shortest = session_dict[key].session_duration

            if session_dict[key].session_duration > longest:
                longest = session_dict[key].session_duration

        return shortest, longest

    def session_buy_amount(self):
        return len(self.bought_items)

    @staticmethod
    def buy_amount_distribution(session_dict):

        buy_dict = {}

        for key in session_dict.keys():
            amount = SessionData.session_buy_amount(session_dict[key])

            if amount not in buy_dict.keys():
                buy_dict[amount] = 1

            else:
                buy_dict[amount] += 1

        print('Session buy amount distribution:\n')

        for key in buy_dict.keys():
            print('%d: %d' % (key, buy_dict[key]))


    # Create input vector for item prediction network
    def create_item_input_vector(self, item_dict, id_item):

        input_vector = []

        # total clicks on item
        # item_dict[id_item].number_of_buys
        # total buys of item

        total_buys = item_dict[id_item].number_of_buys
        if total_buys == 0:
            normalized_total_buys = -1.0
        else:
            normalized_total_buys = total_buys / 50.0
        input_vector.append(total_buys)
        # click duration
        # all time click duration for ths item

        pass

    # Return list containing 0.0 or 1.0 output depending on whether item was actually bought or not
    def create_item_output_vector(self, id_item):

        if id_item in self.bought_items:
            return [1.0]
        else:
            return [0.0]

    # Create input and output vectors for network predicting items that will be bought
    # Inputs, outputs and session id's have matching indices in returned lists
    @staticmethod
    def create_item_input_output_vectors(item_dict, session_dict, buy_sessions):

        source_sessions = []
        input_vectors = []
        output_vectors = []

        for session in buy_sessions:

            for id_item in session_dict[session].items.keys():

                input_vector = session_dict[session].create_item_input_vector(item_dict, id_item)
                output_vector = session_dict[session].create_item_output_vector(id_item)

                input_vectors.append(input_vector)
                output_vectors.append(output_vector)
                source_sessions.append(session)

        return input_vectors, output_vectors, source_sessions




