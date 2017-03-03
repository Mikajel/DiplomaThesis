import numpy as np
import tensorflow as tf
import os
import math
import sys
import importlib
from datetime import datetime
from datetime import timedelta

# Data preprocessing methods
import DataHandler
from DataHandler import *

# Input object classes
import SessionData
from SessionData import *

import ItemData
from ItemData import *

# Set to True to create small sample dataset, otherwise False
use_sample = False
use_subset = False
subset_size = 10000
batch_size = 512


dir = os.getcwd()

dir = '/home/michal/Documents/Jupyter_Tensorflow_Notebooks/udacity/yoochoose'

filename_sample_clicks = os.path.join(dir, 'data', 'yoochoose-clicks-sample.dat')
filename_sample_buys = os.path.join(dir, 'data', 'yoochoose-buys-sample.dat')

filename_clicks = os.path.join(dir, 'data', 'yoochoose-clicks_100k.dat')
filename_buys = os.path.join(dir, 'data', 'yoochoose-buys_100k.dat')

filename_test = os.path.join(dir, '../data/yoochoose-test.dat')

if use_sample:
    print ('Working with test dataset')

    all_buys, train_buys, valid_buys, test_buys = create_dataset_buys(filename_sample_buys)
    train_clicks, valid_clicks, test_clicks = create_dataset_clicks(filename_sample_clicks)

else:
    print ('Working with real dataset')

    all_buys, train_buys, valid_buys, test_buys = create_dataset_buys(filename_buys)
    train_clicks, valid_clicks, test_clicks = create_dataset_clicks(filename_clicks)

if use_subset:
    print ('Working with subset')

    train_buys = train_buys[:subset_size]
    train_clicks = train_clicks[:subset_size]


print('-------  Creating input  ------- \n')

train_session_dict, train_item_dict = DataHandler.structure_raw_data(train_clicks, all_buys)
print('Train data - Created [SessionData dictionary, ItemData dictionary] from dataset rows.')

for key in train_item_dict:
    if train_item_dict[key].number_of_buys > 10:
        print('%d: %d' % (key, train_item_dict[key].number_of_buys))

input('Waiting for input')

train_input_vectors, train_output_vectors, train_source_sessions = SessionData.create_session_input_output_vectors(train_session_dict, info=False)
print('Train data - Finished creating [input vectors, output vectors].')

# Balancing dataset because of incredible ratio of buy sessions and non-buy sessions
# Final ratio - 1:1
os_train_input_vectors, os_train_output_vectors, os_train_source_sessions = oversample_dataset(
    train_input_vectors,
    train_output_vectors,
    train_source_sessions,
    info=True
)

train_batches_vectors = []
train_batches_labels = []
train_batches_source_sessions = []

# Create training batches of vectors and labels
for index in range(0, int(len(os_train_input_vectors) / batch_size)):
    train_batches_vectors.append(os_train_input_vectors[batch_size * index:batch_size * (index + 1)])
    train_batches_labels.append(os_train_output_vectors[batch_size * index:batch_size * (index + 1)])
    train_batches_source_sessions.append(os_train_source_sessions[batch_size * index:batch_size * (index + 1)])

print('Created train dataset\n')

valid_session_dict, valid_item_dict = DataHandler.structure_raw_data(valid_clicks, all_buys)
print('Validation data - Created [SessionData dictionary, ItemData dictionary] from dataset rows.')

valid_input_vectors, valid_output_vectors, valid_source_sessions = SessionData.create_session_input_output_vectors(valid_session_dict, info=False)
os_valid_input_vectors, os_valid_output_vectors, os_valid_source_sessions = oversample_dataset(
    valid_input_vectors,
    valid_output_vectors,
    valid_source_sessions,
    info=True
)
print('Validation data - Finished creating [input vectors, output vectors].')

print('Created validation dataset\n')

test_session_dict, test_item_dict = DataHandler.structure_raw_data(test_clicks, all_buys)
print('Test data - Created [SessionData dictionary, ItemData dictionary] from dataset rows.')

test_input_vectors, test_output_vectors, test_source_sessions = SessionData.create_session_input_output_vectors(test_session_dict, info=False)
print('Test data - Finished creating [input vectors, output vectors].')

print('Created test dataset\n')

print('--------  Dimensions  -------- \n')

print(' Train dataset:        %d %d   ' % (len(train_input_vectors), len(train_input_vectors[0])))
print(' Train labels:         %d %d \n' % (len(train_output_vectors), len(train_output_vectors[0])))

print(' Validation dataset:   %d %d   ' % (len(os_valid_input_vectors), len(os_valid_input_vectors[0])))
print(' Validation labels:    %d %d \n' % (len(os_valid_output_vectors), len(os_valid_output_vectors[0])))

print(' Test dataset:         %d %d   ' % (len(test_input_vectors), len(test_input_vectors[0])))
print(' Test labels:          %d %d \n' % (len(test_output_vectors), len(test_output_vectors[0])))


# observe the distribution of buy amounts
# SessionData.buy_amount_distribution(train_session_dict)
# input("\n Press Enter to continue...")

graph = tf.Graph()
with graph.as_default():
    # Variables.
    num_of_labels = 2
    learning_rate = 0.005
    hidden_size = 128

    # Input data. For the training data, use a placeholder that will be fed at run time with a training minibatch
    tf_train_vectors = tf.placeholder(tf.float32, shape=(batch_size, len(os_train_input_vectors[0])))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, len(os_train_output_vectors[0])))
    tf_valid_vectors = tf.constant(os_valid_input_vectors)
    tf_test_vectors = tf.constant(test_input_vectors)

    # no need to use oversampled vectors, it would only duplicate some session results
    tf_full_train_vectors = tf.constant(train_input_vectors)

    weights_h1 = tf.Variable(tf.truncated_normal([len(os_train_input_vectors[0]), hidden_size]))
    biases_h1 = tf.Variable(tf.zeros([hidden_size]))
    h1 = tf.nn.relu(tf.matmul(tf_train_vectors, weights_h1) + biases_h1)

    weights = tf.Variable(tf.truncated_normal([hidden_size, num_of_labels]))
    biases = tf.Variable(tf.zeros([num_of_labels]))

    # Training computation
    logits = tf.matmul(h1, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(
            tf.nn.relu(
                tf.matmul(tf_valid_vectors, weights_h1) + biases_h1), weights) + biases)
    test_prediction = tf.nn.softmax(
        tf.matmul(
            tf.nn.relu(
                tf.matmul(tf_test_vectors, weights_h1) + biases_h1), weights) + biases)

    # Prediction for full training dataset without oversampling
    full_train_prediction = tf.nn.softmax(
        tf.matmul(
            tf.nn.relu(
                tf.matmul(tf_full_train_vectors, weights_h1) + biases_h1), weights) + biases)

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")

    last_validation_accuracy = 0.0
    stagnation = 0
    stagnation_tolerance = 10
    stop_training = False
    stagnation_definition = 0.00

    dataset_walks = 0


    while not stop_training:

        for batch_index in range(len(train_batches_vectors)):

            if not stop_training:
                # Generate a batch.
                batch_vectors = train_batches_vectors[batch_index]
                batch_labels = train_batches_labels[batch_index]

                # Feed the dictionary and start training
                feed_dict = {tf_train_vectors: batch_vectors, tf_train_labels: batch_labels}

                _, l, predictions = session.run(
                    [optimizer, loss, train_prediction],
                    feed_dict=feed_dict
                )

                # print("Batch loss at batch %d: %f" % (batch_index, l))
                print("Batch accuracy: %.1f%%" %
                     accuracy(predictions, batch_labels))

                validation_accuracy = accuracy(
                    valid_prediction.eval(), os_valid_output_vectors, precision_print=False)

                print("Validate accuracy: %.1f%%\n" % validation_accuracy)

                # Count the difference of validation set accuracy
                diff = validation_accuracy - last_validation_accuracy
                if diff < stagnation_definition:
                    stagnation += 1
                    print('Accuracy improvement only %.3f, stagnation increased to %d' % (diff, stagnation))
                    if stagnation > stagnation_tolerance:
                        stop_training = True

                if diff > stagnation_definition:
                    if stagnation > 0:
                        stagnation -= 1
                        print('Accuracy improvement %.3f, stagnation decreased to %d' % (diff, stagnation))

                last_validation_accuracy = validation_accuracy

        dataset_walks += 1

    print("Test accuracy: %.1f%%" %
          accuracy(test_prediction.eval(), test_output_vectors, precision_print=True))

    print('Number of dataset walks: %d' % dataset_walks)

    full_train_predictions = full_train_prediction.eval()
    predicted_buy_sessions = DataHandler.buy_sessions_selection(full_train_predictions,train_source_sessions)
    print("Amount of predicted buy sessions from original train vectors: %d" % len(predicted_buy_sessions))
    print("Full train accuracy: %.1f%%" %
          accuracy(full_train_prediction.eval(), train_output_vectors, precision_print=True))





