# Data are divided as follows:
# 70% train data
# 20% validation data
# 10% test data

import numpy as np
import math
from datetime import datetime
from datetime import timedelta
import sys
from random import shuffle
from SessionData import SessionData
from ItemData import ItemData


# Description:
#     A main preprocessing cycle
#
# Behaviour:
#     Iterate through all rows
#     Sort into: session dictionary, item dictionary
#
# Input:
#     Raw file rows of test dataset
#
# Output:
#     Session dictionary, item dictionary
def structure_raw_data(train_click_dataset, train_buys_dataset):

    session_dictionary = {}
    item_dictionary = {}
    index = 0
    size = len(train_click_dataset)

    # go through whole input file
    while index < size:

        # new session info
        actual_id_session = train_click_dataset[index]['id_session']
        actual_session_clicks = []

        # continue current session rows until new session arises
        while (index < len(train_click_dataset)) and (train_click_dataset[index]['id_session'] == actual_id_session):

            if math.fmod(index, 25000) == 0:
                print('Currently iterating through index %d' % index)

            actual_session_clicks.append(train_click_dataset[index])
            index += 1



        # create a new session and add it to session dictionary
        actual_session = SessionData(actual_session_clicks)
        session_dictionary[actual_id_session] = actual_session

        for key in actual_session.items:

            # add a new item to item dictionary
            if key not in item_dictionary.keys():
                item_dictionary[key] = ItemData(
                    actual_session.items[key].id_item,
                    actual_session.items[key].category,
                    actual_session.items[key].number_of_clicks,
                    actual_session.items[key].clicktime_total
                )

            # update an existing item in item dictionary
            else:
                item_dictionary[key].update_item(
                    actual_session.items[key].number_of_clicks,
                    actual_session.items[key].clicktime_total
                )

    # initialize buy-lists in Sessions
    SessionData.initialize_item_buys(session_dictionary, train_buys_dataset)
    ItemData.initialize_item_buys(item_dictionary, session_dictionary)

    return session_dictionary, item_dictionary


def precision(predictions, labels):
    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0

    for index in range(0, len(predictions)):

        # Classify label and prediction as True or False
        if predictions[index][0] > 0.5:
            prediction = True
        else:
            prediction = False

        if labels[index][0] > 0.5:
            label = True
        else:
            label = False

        # True branch
        if prediction:
            # Positive branch
            if label:
                true_positive += 1
            # Negative branch
            else:
                false_positive += 1

                # False branch
        else:
            # Positive branch
            if label:
                false_negative += 1
            # Negative branch
            else:
                true_negative += 1

    return true_positive, true_negative, false_positive, false_negative


# Input: predictions and source sessions in lists with corresponding indices
# Example: prediction[i] is for id_session stored in source_sessions[i]
def buy_sessions_selection(predictions, source_sessions):

    buy_sessions = []

    for index in range(0, len(predictions)):

        # if prediction says session is buy-session
        if predictions[index][0] > 0.5:
            if source_sessions[index] not in buy_sessions:
                buy_sessions.append(source_sessions[index])

    return buy_sessions


# Prediction accuracy
def accuracy(predictions, labels, precision_print=False):
    if precision_print:
        true_positive, true_negative, false_positive, false_negative = precision(predictions, labels)

        print('True positive = %d' % true_positive)
        print('True negative = %d' % true_negative)
        print('False positive = %d' % false_positive)
        print('False negative = %d' % false_negative)

    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


# Returns value of logistic sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def find_buy_sessions(buys_list):
    buy_sessions = []

    for row in buys_list:
        if row['id_session'] not in buy_sessions:
            buy_sessions.append(row['id_session'])

    return buy_sessions


# Returns a list of binary values indicating buy sessions
# Order is mapped to input vectors of network
def create_labels(buys, session_object_list, info=False):
    buy_sessions = find_buy_sessions(buys)
    labels = []

    for element in session_object_list:
        if element.id_session in buy_sessions:
            labels.append(1)

        else:
            labels.append(0)

    if info:
        print('Output labels: %d' % len(labels))
        print(', '.join(str(label) for label in labels))

    return labels


# Balance dataset in terms of buy and non-buy sessions
# Labels are [1, 0] for buys session and [0, 1] for non-buy session
# Set 'info' flag to print number of inputs for balanced dataset
# Set 'deep_info' flag to print label vectors
def undersample_dataset(input_vectors, labels, info=False, deep_info=False):
    undersampled_vectors = []
    undersampled_labels = []

    for index in range(0, len(labels)):
        if labels[index] == [1, 0]:
            undersampled_vectors.append(input_vectors[index])
            undersampled_labels.append(labels[index])

    buy_vector_num = len(undersampled_vectors)
    non_buy_sessions = 0
    index = 0

    if info:
        print('\nBalanced to :')
        print('%d buy sessions' % buy_vector_num)

    while non_buy_sessions < buy_vector_num:
        if labels[index] == [0, 1]:
            undersampled_vectors.append(input_vectors[index])
            undersampled_labels.append(labels[index])
            non_buy_sessions += 1
        index += 1

    if info:
        print('%d non-buy sessions\n' % (len(undersampled_vectors) / 2))

    # At this point there should be ordered set of buy sessions and non-buy sessions
    # Now we need to shuffle them without losing the mapping between vectors and labels

    under_shuf_vectors = []
    under_shuf_labels = []
    index_shuf = list(range(len(undersampled_labels)))
    shuffle(index_shuf)
    for i in index_shuf:
        under_shuf_vectors.append(undersampled_vectors[i])
        under_shuf_labels.append(undersampled_labels[i])

    if deep_info:
        print(under_shuf_labels)

    return under_shuf_vectors, under_shuf_labels


# Oversampling of dataset
# Accepts ordered lists of input and output labels
# Returns oversampled, SHUFFLED ordered lists
# Lists are not in original order of samples but index of input vectors still matches index of output vectors
def oversample_dataset(input_vectors, labels, source_sessions, info=False, deep_info=False):
    positive_vectors = []
    negative_vectors = []
    positive_labels = []
    negative_labels = []
    positive_sessions = []
    negative_sessions = []

    extended_positive_vectors = []
    extended_positive_sessions = []

    for index in range(0, len(input_vectors)):
        if labels[index] == [1, 0]:
            positive_vectors.append(input_vectors[index])
            positive_sessions.append(source_sessions[index])
        else:
            negative_vectors.append(input_vectors[index])
            negative_sessions.append(source_sessions[index])

    while len(extended_positive_vectors) < len(negative_vectors):
        if len(negative_vectors) >= (len(extended_positive_vectors) + len(positive_vectors)):
            extended_positive_vectors.extend(positive_vectors)
            extended_positive_sessions.extend(positive_sessions)
        else:
            diff = (len(extended_positive_vectors) + len(positive_vectors)) - len(negative_vectors)
            extended_positive_vectors.extend(positive_vectors[:diff])
            extended_positive_sessions.extend(positive_sessions[:diff])

    positive_vectors = extended_positive_vectors
    positive_sessions = extended_positive_sessions

    positive_label = [1, 0]
    negative_label = [0, 1]

    while len(positive_labels) < len(positive_vectors):
        positive_labels.append(positive_label)

    while len(negative_labels) < len(negative_vectors):
        negative_labels.append(negative_label)

    all_vectors = positive_vectors + negative_vectors
    all_labels = positive_labels + negative_labels
    all_sessions = positive_sessions + negative_sessions

    if info:
        print('\nBalanced to :')
        print('%d buy sessions' % len(positive_vectors))
        print('%d non-buy sessions' % len(negative_vectors))
        print('%d total sessions\n' % len(all_vectors))

    # At this point there should be ordered set of buy sessions and non-buy sessions
    # Now we need to shuffle them without losing the mapping between vectors, labels and sessions

    shuf_vectors = []
    shuf_labels = []
    shuf_sessions = []

    # this creates a list of [0, 1, 2, 3, ...] and then shuffles it.
    # then we only need to use numbers as indexing for shuffled lists
    index_shuf = list(range(len(all_labels)))
    shuffle(index_shuf)

    for index in index_shuf:
        shuf_vectors.append(all_vectors[index])
        shuf_labels.append(all_labels[index])
        shuf_sessions.append(all_sessions[index])

    if deep_info:
        print(shuf_labels)

    return shuf_vectors, shuf_labels, shuf_sessions


def create_dataset_clicks(filename):
    print('Loading dataset of click events')

    if (sys.version_info > (3, 0)):
        data_type = np.dtype([('id_session', int), ('timestamp', 'S32'), ('id_item', int), ('category', 'S32')])
    else:
        data_type = np.dtype([('id_session', int), ('timestamp', (str, 32)), ('id_item', int), ('category', (str, 32))])
    data = np.loadtxt(filename, dtype=data_type, delimiter=',')

    print('Size of clicks dataset: %d' % len(data))

    train_range_low = 0
    train_range_high = int(math.floor(len(data) * 0.80))

    valid_range_low = int(math.floor(len(data) * 0.80))
    valid_range_high = int(math.floor(len(data) * 0.90))

    test_range_low = int(math.floor(len(data) * 0.90))
    test_range_high = len(data)

    train_dataset = data[train_range_low:train_range_high]
    valid_dataset = data[valid_range_low:valid_range_high]
    test_dataset = data[test_range_low:test_range_high]

    print('Train size: %d' % len(train_dataset))
    print('Valid size: %d' % len(valid_dataset))
    print('Test size: %d' % len(test_dataset))

    print('Finished loading dataset')

    return train_dataset, valid_dataset, test_dataset


def create_dataset_buys(filename):
    print('Loading dataset of buy events')

    if sys.version_info > (3, 0):
        data_type = np.dtype(
            [('id_session', int), ('timestamp', 'S32'), ('id_item', int), ('price', int), ('quantity', int)])
    else:
        data_type = np.dtype(
            [('id_session', int), ('timestamp', (str, 32)), ('id_item', int), ('price', int), ('quantity', int)])
    data = np.loadtxt(filename, dtype=data_type, delimiter=',')

    print('Size of buys dataset: %d' % len(data))

    train_range_low = 0
    train_range_high = int(math.floor(len(data) * 0.8))

    valid_range_low = int(math.floor(len(data) * 0.8))
    valid_range_high = int(math.floor(len(data) * 0.9))

    test_range_low = int(math.floor(len(data) * 0.9))
    test_range_high = len(data)

    train_dataset = data[train_range_low:train_range_high]
    valid_dataset = data[valid_range_low:valid_range_high]
    test_dataset = data[test_range_low:test_range_high]

    print('Train size: %d' % len(train_dataset))
    print('Validate size: %d' % len(valid_dataset))
    print('Test size: %d' % len(test_dataset))

    print('Finished loading dataset')

    return data, train_dataset, valid_dataset, test_dataset


# Break down timestamp value into detailed values
# Basic timestamp format: YYYY-MM-DDThh:mm:ss.SSSZ
# Return: timestamp array - year, month, day, hour, minute, second
def parse_timestamp(timestamp):
    # String is returned in byte format from file in python3
    if (sys.version_info > (3, 0)):
        timestamp = timestamp.decode('utf-8')

    date_time = timestamp.split('T')

    date = date_time[0]
    year = int(date.split('-')[0])
    month = int(date.split('-')[1])
    day = int(date.split('-')[2])

    time = date_time[1].split('Z')[0]
    hour = int(time.split(':')[0])
    minute = int(time.split(':')[1])
    second = int(time.split(':')[2].split('.')[0])

    timestamp_list = [year, month, day, hour, minute, second]
    return timestamp_list


# Accepts timestamp in the format YYYY-MM-DDThh:mm:ss.SSSZ
# Returns datetime object
def timestamp_to_datetime(timestamp):
    values = parse_timestamp(timestamp)

    return datetime(values[0], values[1], values[2], values[3], values[4], values[5])


# Reformat 0 or 1 label list of single element
def reformat(labels):
    label_set = []
    for element in labels:
        if element:
            label_set.append([1, 0])
        else:
            label_set.append([0, 1])

    return label_set


# Count how many times each item has been clicked
# Returns dictionary {key = id_item, value = click_count}
def total_clicks_on_items(train_click_dataset):
    item_click_count_dictionary = {}

    for row in train_click_dataset:

        if row['id_item'] in item_click_count_dictionary.keys():
            item_click_count_dictionary[row['id_item']] += 1

        else:
            item_click_count_dictionary[row['id_item']] = 1

    return item_click_count_dictionary


# Print dictionary as a control method
def dump(obj, nested_level=0, output=sys.stdout):
    spacing = '   '
    if type(obj) == dict:
        print >> output, '%s{' % ((nested_level) * spacing)
        for k, v in obj.items():
            if hasattr(v, '__iter__'):
                print >> output, '%s%s:' % ((nested_level + 1) * spacing, k)
                dump(v, nested_level + 1, output)
            else:
                print >> output, '%s%s: %s' % ((nested_level + 1) * spacing, k, v)
        print >> output, '%s}' % (nested_level * spacing)
    elif type(obj) == list:
        print >> output, '%s[' % ((nested_level) * spacing)
        for v in obj:
            if hasattr(v, '__iter__'):
                dump(v, nested_level + 1, output)
            else:
                print >> output, '%s%s' % ((nested_level + 1) * spacing, v)
        print >> output, '%s]' % ((nested_level) * spacing)
    else:
        print >> output, '%s%s' % (nested_level * spacing, obj)


# Returns lowest and highest recorded price of given item in dataset
def lowest_highest_prices_on_items(train_buy_dataset):
    max_price = 0
    min_price = 1000000
    size = len(train_buy_dataset)

    for count in range(0, size):
        if train_buy_dataset[count]['price'] > max_price:
            max_price = train_buy_dataset[count]['price']

        if train_buy_dataset[count]['price'] < min_price:
            min_price = train_buy_dataset[count]['price']

    return min_price, max_price


# Counts unique items in given session from train dataset of clicks
# Returns number of unique items and start index for next session
def unique_session_items(train_click_dataset, id_session, start_index):
    id_array = []
    index = start_index

    for row in train_click_dataset:

        while row['id_session'] == id_session:

            if row['id_session'] not in id_array:
                id_array.append(train_click_dataset[index]['id_session'])

            index += 1

    return len(id_array), index


# Create a list of all unique categories in train dataset
def list_of_categories(train_click_dataset):
    size = len(train_click_dataset)
    categories = []

    for count in range(0, size):
        if train_click_dataset[count]['category'] not in categories:
            categories.append(train_click_dataset[count]['category'])

    return categories


# Counts clicks for all categories in given session (even non-present categories)
# Return dictionary of categories and starting index for next session
def clicks_in_session_categories(train_click_dataset, start_index, categories):
    dictionary = dict.fromkeys(categories, 0)

    index = start_index
    id_session = train_click_dataset[start_index]['id_session']

    while train_click_dataset[index]['id_session'] == id_session:

        key = train_click_dataset[index]['category']
        if key in dictionary:
            dictionary[key] += 1
        else:
            pass

        index += 1

    return dictionary, id_session, start_index


# Create a dictionary of dictionaries
# Mapping: distribution{ key = 'id_session', value = { key = 'category', value = 'number_of_clicks'}}
def session_category_distribution(train_click_dataset):
    category_distributions_in_sessions = {}
    categories = list_of_categories(train_click_dataset)

    index = 0
    end = len(train_click_dataset)

    while index < end:

        dictionary, id_session, index = clicks_in_session_categories(train_click_dataset, index, categories)

        if id_session not in category_distributions_in_sessions.keys():

            category_distributions_in_sessions[id_session] = dictionary
        else:
            print('Warning: Duplicate entry into supposedly non-conflict dictionary')
            print('Possible indication of sessions not being grouped in file')

    return category_distributions_in_sessions


# Returns difference between two timestamps in seconds
def timestamp_diff(timestamp_1, timestamp_2):
    time_1 = timestamp_to_datetime(timestamp_1)
    time_2 = timestamp_to_datetime(timestamp_2)

    diff = time_1 - time_2

    return abs(diff.total_seconds())


# Returns how many minutes passed since 00:00
def time_of_day(timestamp):
    time = parse_timestamp(timestamp)

    return time[3] * 60 + time[4]


# Iterates through buys dataset and sorts sessions into dictionary
# Dictionary key is 'id_session' and value is a list of lines of session.
def create_sessions_from_buys(train_buys_dataset):
    index = 0
    size = len(train_buys_dataset)
    sessions_buys = {}

    while index < size:

        id_session = train_buys_dataset[index]['id_session']
        actual_session = []

        while (index < size) and (train_buys_dataset[index]['id_session'] == id_session):
            actual_session.append(train_buys_dataset[index])

        if id_session not in sessions_buys.keys():

            sessions_buys[id_session] = actual_session
        else:
            print('Warning: Duplicate entry into supposedly non-conflict dictionary')
            print('Possible indication of sessions not being grouped in file')

        index += 1

    return sessions_buys


# Return normalized value between <0;1>
def normalize(value, min_value, max_value):
    if max_value == min_value:
        return 0

    else:
        return (value - min_value) / (max_value - min_value)
