#!/usr/bin/env python

# center code
# 1.initialize parameters and models
# 2.receive gradients from the users and update the parameters and save it for future usea
# 3.calculate the test accuracy for the MNIST test set

###########################
# import and constants

# import packages
from __future__ import division
import sys
import time
import numpy as np
import pyrebase
import tensorflow as tf
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# import functions from other folder
sys.path.insert(0, '../')
import CLIENT.Python.allFunc as usr_func
import model



# function to evaluate the accuracy on the test set
def eval(weight, bias):
    with tf.Session() as sess:
        W1 = tf.Variable(weight[0], dtype=tf.float32)  # first hidden layer
        W2 = tf.Variable(weight[1], dtype=tf.float32)  # second hidden layer
        W3 = tf.Variable(weight[2], dtype=tf.float32)  # output layer
        b1 = tf.Variable(bias[0], dtype=tf.float32)
        b2 = tf.Variable(bias[1], dtype=tf.float32)
        b3 = tf.Variable(bias[2], dtype=tf.float32)
        init = tf.global_variables_initializer()
        init_2 = tf.local_variables_initializer()
        # initialize variables
        sess.run(init)
        sess.run(init_2)

        Y1 = tf.nn.relu(tf.matmul(testimages, W1) + b1)  # first layer
        Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)  # second layer
        Ylogits = tf.matmul(Y2, W3) + b3  # output layer
        Y = tf.nn.softmax(Ylogits)
        aa = sess.run(tf.argmax(Y, 1))
        cc = sess.run(testlabels)
        accuracy = sum(aa == cc) / usr_func.NUM_TESTING_IMAGES
    return accuracy

if __name__ == '__main__':
    # read test dataset
    testimages, testlabels = usr_func.MnistInput(usr_func.mnist_test_file, whole=True)
    testimages = tf.convert_to_tensor(testimages, dtype=tf.float32)
    testlabels = tf.convert_to_tensor(testlabels)

    # initialize parameters and iter
    iter = 0
    np.savetxt('./param/iter.csv', iter, delimiter=',')

    W1 = np.zeros((784, model.dnn['units'][0]))
    b1 = np.zeros(model.dnn['units'][0])
    W2 = np.zeros((model.dnn['units'][0], model.dnn['units'][1]))
    b2 = np.zeros(model.dnn['units'][1])
    W3 = np.zeros((model.dnn['units'][1], 10))
    b3 = np.zeros(10)
    np.savetxt('./param/W1.csv', W1, delimiter=',')
    np.savetxt('./param/b1.csv', b1, delimiter=',')
    np.savetxt('./param/W2.csv', W2, delimiter=',')
    np.savetxt('./param/b2.csv', b2, delimiter=',')
    np.savetxt('./param/W3.csv', W3, delimiter=',')
    np.savetxt('./param/b3.csv', b3, delimiter=',')






