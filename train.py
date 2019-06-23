import csv
import os
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
import numpy as np


class dataunit:
    label = 0   # EPS in the next year, list accross several years
    feature = []    # other feature


def read_data_sequence():
    # load the data sequence for each company (each csv file)
    # path = os.getcwd() + "\\data\\"   # windows
    path = os.getcwd() + "//data//" # ubuntu
    lst = os.listdir(path)
    x_data = []
    y_data = []
    for item in lst:
        fin = open(path + item, 'r')
        # print fin
        cnt = 0
        matrix = []
        labels = []
        features = []
        # print item
        for row in csv.reader(fin):
            # print row, type(row)
            if cnt == 0:
                cnt += 1
                continue
            # print row
            matrix.append([])
            # print item, row
            for i in range(1, len(row)):
                matrix[-1].append(float(row[i]))

        # print matrix
        labels.append(max(matrix[18][0], 1e-7))
        for i in range(1, len(matrix[0])):
            features.append([])

            for j in range(0, len(matrix)):
                features[-1].append(matrix[j][i])
        features.reverse()
        # print labels
        # print features
        # unit = dataunit()
        # unit.label[:] = labels
        # unit.feature[:] = features
        x_data.append(features)
        y_data.append(labels)
        del features
        del labels
        del matrix
        fin.close()
        # return
    return x_data, y_data

def train_agent():
    tf.set_random_seed(1234)
    np.random.seed(1234)
    # Training Parameters
    learning_rate = 0.0001
    training_steps = 10000
    batch_size = 32
    display_step = 20

    # Network Parameters
    num_input = 26  # numebr of factors we consider
    timesteps = 4  # timesteps
    num_hidden = 128  # hidden layer num of features
    num_classes = 1  # output EPS in the next year

    # tf Graph input
    X = tf.placeholder("float", [None, timesteps, num_input], name='X')
    Y = tf.placeholder("float", [None, num_classes], name='Y')

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    def RNN(x, weights, biases):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, timesteps, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    prediction = RNN(X, weights, biases)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.losses.mean_squared_error(predictions=prediction, labels=Y))  # MSE
    # loss_op = tf.reduce_mean(tf.div(tf.losses.absolute_difference(labels=Y, predictions=prediction, reduction=tf.losses.Reduction.NONE), Y))    # relative error
    # loss_op = tf.reduce_mean(tf.losses.absolute_difference(labels=Y, predictions=prediction))
    # abs_loss = tf.losses.absolute_difference(labels=Y, predictions=prediction, reduction=tf.losses.Reduction.NONE)
    # abs_div = tf.div(tf.losses.absolute_difference(labels=Y, predictions=prediction, reduction=tf.losses.Reduction.NONE), Y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    # correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        x_data, y_data = read_data_sequence()   # 80% training, 20% for testing
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
        tf_writer = tf.summary.FileWriter(os.getcwd())
        tf_writer.add_graph(sess.graph)
        for step in range(0, training_steps):
            for ba in range(0, len(x_train), batch_size):
                batch_x = np.array(x_train[ba: min(ba + batch_size, len(x_train))])
                batch_y = np.array(y_train[ba: min(ba + batch_size, len(y_train))])
                # Reshape data to get 28 seq of 28 elements
                batch_x = batch_x.reshape((-1, timesteps, num_input))
                batch_y = batch_y.reshape(-1, 1)
                # Run optimization op (backprop)
                # print batch_x
                # print batch_y
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step % display_step == 0 and ba == 0:
                    # Calculate batch loss and accuracy
                    # abs_diff = sess.run(abs_loss, feed_dict={X: batch_x, Y: batch_y})
                    # abs_div = sess.run(abs_div, feed_dict={X: batch_x, Y: batch_y})
                    #
                    # pred = sess.run(prediction, feed_dict={X: batch_x, Y: batch_y})
                    # print "predict", pred
                    #
                    # print "y", batch_y
                    # print "abs loss", abs_diff
                    # print "abs div", abs_div

                    # return
                    loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
                    print("Step " + str(step) + ", MSE Loss = " + "{:.4f}".format(loss))
                    # Calculate loss of test data
                    test_data = np.array(x_test).reshape((-1, timesteps, num_input))
                    test_label = np.array(y_test).reshape((-1, 1))
                    test_error = sess.run(loss_op, feed_dict={X: test_data, Y: test_label})
                    print "Testing MSE Error:", test_error
                    loss_summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss)])
                    tf_writer.add_summary(loss_summary, step)
                    test_summary = tf.Summary(value=[tf.Summary.Value(tag="test loss", simple_value=test_error)])
                    tf_writer.add_summary(test_summary, step)
                    tf_writer.flush()
                    # return




if __name__ == '__main__':
    os.system("rm ./events.*")
    train_agent()
    # read_data_sequence()