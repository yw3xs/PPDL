# Functions for different users
# Author: Yang Wang
# 1. read whole MNIST dataset and also the sub-dataset belonging to a specific user
# 2. download model and parameters from center
# 3. sample batch from the dataset
# 4. calculate gradient and add noise
# 5. keep accountant of privacy budget
# 6. implement report-noisy-max or exponential mechanism

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

# import auxiliary functions from the current folder
# import noise
# import config
# import dataPrep
# import utils
# import per_example_gradients
# import dp_optimizer
# import sanitizer

# import for testing purpose
# import CLIENT.Python.config as config
# import CLIENT.Python.noise as noise
# import CLIENT.Python.dataPrep as dataPrep
# import CLIENT.Python.utils as utils
# import CLIENT.Python.per_example_gradients as per_example_gradients
# import CLIENT.Python.dp_optimizer as dp_optimizer
# import CLIENT.Python.sanitizer as sanitizer
# import CLIENT.Python.download_and_convert_mnist as download_and_convert_mnist

# constants
NUM_TRAINING_IMAGES = 60000
NUM_TESTING_IMAGES = 10000
IMAGE_SIZE = 28
mnist_train_file = '/home/yang/Research/Privacy-preserving-DL/PPDL/DATA/MNIST/mnist_train.tfrecord'
mnist_test_file = '/home/yang/Research/Privacy-preserving-DL/PPDL/DATA/MNIST/mnist_test.tfrecord'

# parameters
Lot_size = 5000 # the number of samples used in each iteration by each party (at most can use all of the samples in the party)
batch_size = 500
epochs = 1   ## how many runs on the lot at each iteration?
epsilon = .1 # privacy budget for each epoch
delta = .000001
learning_rate = .1
grad_bound = .001
grad_threshold = .0001  # for SVT
grad_upload_ratio = .001 # ratio of parameters for uploading at each iteration
grad_upload_num = int((784*64 + 640) * grad_upload_ratio) # number of parameters for uploading at each iteration

# neural network structures
units = [128,64]  # number of units in each hidden layer
layers = 2  # number of hidden layer
input = 784  # input size
output = 10  # output size

# function to read the MNIST dataset (when whole = "True") or the user dataset (when whole = "False")
def MnistInput(mnist_data_file, whole=True, start=None, size=None):
    """Create operations to read the MNIST input file.
      Args:
        mnist_data_file: Path of a tfrecord file containing the MNIST images to process.
        whole: when set to true, return the whole MNIST dataset (training or test set)
        start: start index of the first sample in the user dataset
        size: size of the user dataset

      Returns:
        images: A list with the formatted image data. default shape [10000, 28*28]
        labels: A list with the labels for each image. default shape [10000]
      """
    with tf.Session() as sess:
        file_queue = tf.train.string_input_producer([mnist_data_file], num_epochs=1)
        reader = tf.TFRecordReader()
        _, value = reader.read(file_queue)
        example = tf.parse_single_example(
            value,
            features={"image/encoded": tf.FixedLenFeature(shape=(), dtype=tf.string),
                      "image/class/label": tf.FixedLenFeature([1], tf.int64)})

        image = tf.cast(tf.image.decode_png(example["image/encoded"], channels=1),
                        tf.float32)
        image = tf.reshape(image, [IMAGE_SIZE * IMAGE_SIZE])
        image /= 255
        label = tf.cast(example["image/class/label"], dtype=tf.int32)
        label = tf.reshape(label, [])

        init_op = tf.global_variables_initializer()
        init_op2 = tf.local_variables_initializer()
        sess.run(init_op)
        sess.run(init_op2)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        images = []
        labels = []
        if whole:
            try:
                while True:
                    i, l = sess.run([image, label])
                    i = i.tolist()
                    images.append(i)
                    labels.append(l)
            except tf.errors.OutOfRangeError, e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
        else:
            try:
                for k in xrange(start - 1):
                    sess.run([image, label])
                for k in xrange(start, start + size):
                    i, l = sess.run([image, label])
                    i = i.tolist()
                    images.append(i)
                    labels.append(l)
            except tf.errors.OutOfRangeError, e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

        return images, labels


# build the neural network given the model parameters and calculate the gradient (clipped parameter difference)
def trainNN(weight,bias,iter,images,labels, lr, bound):
    """build and train the neural network, return the gradient and updated parameter
        Args:
            weight,bias: weights and bias for each layer (list of list)
            iter: iteration number
            images,labels: images and labels for this user

        Returns:           
    """
    tf.Graph().as_default()
    tf.device('/cpu:0')
    # convert the list to tensors for model building
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels)
    with tf.Session() as sess:
        lr = tf.placeholder(tf.float32)  # learning rate
        eps = tf.placeholder(tf.float32)  # epsilon (privacy parameter)
        delta = tf.placeholder(tf.float32)  # delta (privacy parameter)
        # hard-coded the network structure (2 hidden layers)
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

        round = int(Lot_size / batch_size)
        # sample lot_size samples from the user data
        Lot_index = random.sample(range(len(images)),Lot_size)
        image_lot = [images[i] for i in Lot_index]
        label_lot = [labels[i] for i in Lot_index]

        for _ in xrange(epochs):
            for i in xrange(round):
                # sample one batch from the lot
                image_batch = image_lot[(i*batch_size):((i+1)*batch_size), :]
                label_batch = label_lot[(i*batch_size):((i+1)*batch_size)]
                # build the network
                Y1 = tf.nn.relu(tf.matmul(image_batch, W1) + b1)  # first layer
                Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)  # first layer
                Ylogits = tf.matmul(Y2, W3) + b3  # output layer
                Y = tf.nn.softmax(Ylogits)
                # objective function (cross entropy)
                cost = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=tf.one_hot(label_batch, 10))
                cost = tf.reduce_sum(cost) / batch_size
                op1 = tf.train.GradientDescentOptimizer(lr).compute_gradients(cost)
                op2 = tf.train.GradientDescentOptimizer(lr).apply_gradients(op1)
                sess.run(op1) # grad[0-1](W1-W2)[0-1](grad + var)
                sess.run([op2], feed_dict={lr: lr})

        newW1 = sess.run(W1)
        newW2 = sess.run(W2)
        newW3 = sess.run(W3)
        newb1 = sess.run(b1)
        newb2 = sess.run(b2)
        newb3 = sess.run(b3)

        # clip difference
        W1_diff = np.clip((newW1 - weight[0]).flatten(), -bound, bound)
        W2_diff = np.clip((newW2 - weight[1]).flatten(), -bound, bound)
        W3_diff = np.clip((newW3 - weight[2]).flatten(), -bound, bound)
        b1_diff = np.clip((newb1 - bias[0]).flatten(), -bound, bound)
        b2_diff = np.clip((newb2 - bias[1]).flatten(), -bound, bound)
        b3_diff = np.clip((newb3 - bias[2]).flatten(), -bound, bound)


        return W1_diff, W2_diff, W3_diff, b1_diff, b2_diff, b3_diff


def noisyMax(grad, n, scale):
    """
    report noisy maximum algorithm
            Args:
                grad: true gradient of each parameter
                n: # of output noisy gradient
                scale: scale of the laplace distribution (calibrated by sensitivity)
            Returns: 
                the noisy version of the top n gradient
    """

    noisyGrad = grad + np.random.laplace(loc = 0.0, scale = scale, size = len(grad))
    index = sorted(range(len(noisyGrad)), key=lambda i: abs(noisyGrad[i]))[-n:]
    return [noisyGrad[i] if i in index else 0 for i in range(noisyGrad)]


#def account():





##################### Start training the model ################################
# 1. set up the network
# 2. start the iteration
#    2a. Initialize variables with server parameter
#    2b. sample batch size data points
#    2c. calculate the gradient, apply the gradient
##################################################################################

# tf.Graph().as_default()
# tf.device('/cpu:0')
# sess = tf.Session()
#
# while True:
#     lr = tf.placeholder(tf.float32)  # learning rate
#     eps = tf.placeholder(tf.float32)  # epsilon
#     delta = tf.placeholder(tf.float32)  # delta
#
#     round = int(sample_size / batch_size)
#
#     # initialize the parameters using the server parameters
#     serverParam1 = db.child('parameter').child('parameter1').get().val()  # list with length 50176
#     serverParam2 = db.child('parameter').child('parameter2').get().val()
#     serverIter = db.child('parameter').child('iteration').get().val()
#     print "iter: " + str(serverIter)
#     serverParam1 = np.reshape(serverParam1, (784,64))
#     serverParam2 = np.reshape(serverParam2, (64,10))
#
#     W1 = tf.Variable(serverParam1, dtype=tf.float32)  # second layer
#     W2 = tf.Variable(serverParam2, dtype=tf.float32)  # third layer
#     init = tf.global_variables_initializer()
#     init_2 = tf.local_variables_initializer()
#     sess.run(init)
#     sess.run(init_2)
#
#     # evaluate the accuracy
#     def eval(images, labels, param1, param2):
#         Y1 = tf.nn.relu(tf.matmul(images, param1))  # first layer
#         Ylogits = tf.matmul(Y1, param2)  # output layer
#         Y = tf.nn.softmax(Ylogits)
#         aa = sess.run(tf.argmax(Y, 1))
#         cc = sess.run(labels)
#         accuracy = sum(aa == cc) / 10000
#         return accuracy
#     epochs = 1
#     for _ in xrange(epochs):
#         print "Epoch " + str(_) + " starts!"
#         print "accuracy on epoch " + str(_) + ":"
#         print eval(testimages, testlabels, W1, W2)
#         for i in xrange(round):
#             #print "epoch: "+ str(_) + " round " + str(i)
#             # sample one batch
#             image_batch = images[(i*batch_size):((i+1)*batch_size), :]
#             label_batch = labels[(i*batch_size):((i+1)*batch_size)]
#
#             # build the network
#             Y1 = tf.nn.relu(tf.matmul(image_batch, W1))  # first layer
#             Ylogits = tf.matmul(Y1, W2)  # output layer
#             Y = tf.nn.softmax(Ylogits)
#
#             # objective function (cross entropy)
#             cost = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=tf.one_hot(label_batch, 10))
#             cost = tf.reduce_sum(cost) / batch_size
#             op1 = tf.train.GradientDescentOptimizer(lr).compute_gradients(cost)
#             op2 = tf.train.GradientDescentOptimizer(lr).apply_gradients(op1)
#             # saver = tf.train.Saver()
#             # coord = tf.train.Coordinator()
#             # _ = tf.train.start_queue_runners(sess=sess, coord=coord)
#             grad = sess.run(op1) # grad[0-1](W1-W2)[0-1](grad + var)
#             sess.run([op2], feed_dict={lr: learning_rate})
#         print "Epoch " + str(_) + " ends!"
#     print "accuracy on epoch " + str(epochs) + ":"
#     print eval(testimages, testlabels, W1, W2)
#
#     #####################################################
#     newParam1 = sess.run(W1)
#     newParam2 = sess.run(W2)
#
#     Param1_diff = newParam1 - serverParam1   # range: -0.06777553 ~ 0.1147576175
#     Param1_diff = Param1_diff.flatten()
#     Param1_diff.max()
#     Param1_diff.min()
#     np.argmax(Param1_diff)
#     np.argmin(Param1_diff)
#
#     Param2_diff = newParam2 - serverParam2  # range:  -0.33470168 ~ 0.506309772
#     Param2_diff = Param2_diff.flatten()
#     Param2_diff.max()
#     Param2_diff.min()
#     np.argmax(Param2_diff)
#     np.argmin(Param2_diff)
#
#     tf.set_random_seed(0)
#
#     noise1 = np.random.laplace(loc = 0.0, scale = 2*grad_upload_num*2*grad_bound/epsilon/8*9)
#
#     noise2 = np.random.laplace(loc = 0.0, scale = 2*2*grad_upload_num*2*grad_bound/epsilon/8*9)
#
#     grad1_upload = np.zeros(len(Param1_diff))
#     grad2_upload = np.zeros(len(Param2_diff))
#
#
#
#     for i in xrange(len(Param1_diff)):
#         if abs(np.clip(Param1_diff[i], -grad_bound, grad_bound)) + noise2 >= (grad_threshold + noise1):
#             noise3 = np.random.laplace(loc = 0.0, scale = 2*grad_upload_num*2*grad_bound/epsilon*9)
#             grad1_upload[i] = np.clip(Param1_diff[i] + noise3, -grad_bound, grad_bound)
#
#     for i in xrange(len(Param2_diff)):
#         if abs(np.clip(Param2_diff[i], -grad_bound, grad_bound)) + noise2 >= grad_threshold + noise1:
#             noise3 = np.random.laplace(loc=0.0, scale=2 * grad_upload_num * 2 * grad_bound / epsilon * 9)
#             grad2_upload[i] = np.clip(Param2_diff[i] + noise3, -grad_bound, grad_bound)
#
#
#     grad_upload = np.concatenate((grad1_upload,grad2_upload))
#     index = np.argsort(grad_upload)[::-1][int(grad_upload_num/2):(len(grad_upload) - int(grad_upload_num/2))]
#     grad_upload[index] = 0
#
#     grad1_upload = grad_upload[:len(grad1_upload)]
#
#     grad2_upload = grad_upload[len(grad1_upload):]
#
#
#     userIter += 1
#     userData['users/' + uid]['userIter'] = userIter
#     userData['users/' + uid]['grad1'] = list(Param1_diff)
#     userData['users/' + uid]['grad2'] = list(Param2_diff)
#     #userData['users/' + uid]['grad1'] = list(grad1_upload)
#     #userData['users/' + uid]['grad2'] = list(grad2_upload)
#     db.update(userData)
#
