# import packages
from __future__ import division
import sys
import time
import numpy as np
import tensorflow as tf
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# define constants
NUM_TRAINING_IMAGES = 60000
NUM_TESTING_IMAGES = 10000
IMAGE_SIZE = 28
mnist_train_file = '/home/yang/Research/Privacy-preserving-DL/PPDL/DATA/MNIST/mnist_train.tfrecord'
mnist_test_file = '/home/yang/Research/Privacy-preserving-DL/PPDL/DATA/MNIST/mnist_test.tfrecord'

# define parameters
usrSize = np.array([10000, 8000, 4000, 12000, 6000, 5000])
users = 6
lotSize = 2000  # the number of samples used in each iteration by each party (at most can use all of the samples in the party)
batchSize = 500
epochs = 1   ## how many runs on the lot at each iteration?
epsilon = [.05,.04,.05,.08,.03,.04] # privacy budget for each iteration
delta = np.reciprocal(usrSize, dtype = float) * .5
totalIter = 2
learning_rate = 1
grad_bound = .001  ## how to set???
# neural network structures
layers = 2  # number of hidden layer
units = np.array([100,50])  # number of units in each hidden layer
input = IMAGE_SIZE ** 2  # input size
output = 10  # output size
ratio = .1 # the top 10% of the gradient values will be used for the update
paramD = input * units[0] + units[0] + units[0] * units[1] + units[1] + units[1] * output + output


# define functions
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
            while True:
                i, l = sess.run([image, label])
                i = i.tolist()
                images.append(i)
                labels.append(l)
        else:
           for k in xrange(start - 1):
               sess.run([image, label])
           for k in xrange(start, start + size):
               i, l = sess.run([image, label])
               i = i.tolist()
               images.append(i)
               labels.append(l)

    return images, labels


def privParm(epsilon, iter, delta):
    """
    Calculate the epsilon for each iteration based on the total privacy parameters
    ref: DP book Corollary 3.21 (strong composition theorem)
      Args:
        epsilon: total privacy budget for the user
        iter: total iteration
        delta: total tolerable privacy failure probability 
      Returns:
        stepEps: the privacy budget for each iteration
    """
    return epsilon / (2 * np.sqrt(2 * iter * np.log(1 / delta)))


def trainNN(images, labels, size, learningRate, bound, Lot_size, batch_size, epochs):
    """build and train the neural network, return the gradient and updated parameter
        Args:
            weight,bias: weights and bias for each layer (list of list)
            iter: iteration number
            images,labels: images and labels for this user

        Returns:           
            parameter (weight, bias) difference
    """
    # tf.Graph().as_default()
    # tf.device('/cpu:0')
    with tf.Session() as sess:
        # lr = tf.placeholder(tf.float32)  # learning rate
        # hard-coded the network structure (2 hidden layers)
        # W1 = tf.Variable(weight[0], dtype=tf.float32)  # first hidden layer
        # W2 = tf.Variable(weight[1], dtype=tf.float32)  # second hidden layer
        # W3 = tf.Variable(weight[2], dtype=tf.float32)  # output layer
        # b1 = tf.Variable(bias[0], dtype=tf.float32)
        # b2 = tf.Variable(bias[1], dtype=tf.float32)
        # b3 = tf.Variable(bias[2], dtype=tf.float32)

        init = tf.global_variables_initializer()
        init_2 = tf.local_variables_initializer()
        # initialize variables
        sess.run([init, init_2])

        round = int(Lot_size / batch_size)
        # sample lot_size samples from the user data
        Lot_index = random.sample(range(len(images)), Lot_size)
        image_lot = [images[i] for i in Lot_index]
        label_lot = [labels[i] for i in Lot_index]

        for _ in xrange(epochs):
            for i in xrange(round):
                # sample one batch from the lot
                image_batch = image_lot[(i * batch_size):((i + 1) * batch_size), :]
                label_batch = label_lot[(i * batch_size):((i + 1) * batch_size)]
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
                saver = tf.train.Saver()
                coord = tf.train.Coordinator()
                _ = tf.train.start_queue_runners(sess=sess, coord=coord)
                sess.run(op1)
                sess.run([op2], feed_dict={lr: learningRate})

        newW1 = sess.run(W1)
        newW2 = sess.run(W2)
        newW3 = sess.run(W3)
        newb1 = sess.run(b1)
        newb2 = sess.run(b2)
        newb3 = sess.run(b3)
    # sess.close()

    # clip difference
    W1_diff = np.clip((newW1 - weight[0]).flatten(), -bound, bound)
    W2_diff = np.clip((newW2 - weight[1]).flatten(), -bound, bound)
    W3_diff = np.clip((newW3 - weight[2]).flatten(), -bound, bound)
    b1_diff = np.clip((newb1 - bias[0]).flatten(), -bound, bound)
    b2_diff = np.clip((newb2 - bias[1]).flatten(), -bound, bound)
    b3_diff = np.clip((newb3 - bias[2]).flatten(), -bound, bound)
    print("one round")
    return W1_diff, W2_diff, W3_diff, b1_diff, b2_diff, b3_diff


def noisyMax(grad, n, epsilon, bound):
    """
    report noisy maximum algorithm
            Args:
                grad: true gradient of each parameter
                n: # of output noisy gradient
                scale: scale of the laplace distribution (calibrated by sensitivity)
            Returns: 
                the noisy version of the top n gradient
    """
    l = len(grad)
    scale = 2 * bound * 2 * n / epsilon
    noisyGrad = grad + np.random.laplace(loc=0.0, scale=scale, size=len(grad))
    index = sorted(range(l), key=lambda i: np.abs(noisyGrad[i]))[-int(n):]
    return [noisyGrad[i] if i in index else 0 for i in range(l)]


def account(epsilon, iter, delta):
    """
    return the total used privacy budget based on budget on each iteration

    """
    return np.sqrt(2 * iter * np.log(1 / delta)) * epsilon + iter * epsilon * (np.exp(epsilon) - 1)


# function to evaluate the accuracy on the test set
def eval(images, labels):
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        init_2 = tf.local_variables_initializer()
        # initialize variables
        sess.run([init, init_2])

        Y1 = tf.nn.relu(tf.matmul(images, W1) + b1)  # first layer
        Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)  # second layer
        Ylogits = tf.matmul(Y2, W3) + b3  # output layer
        Y = tf.nn.softmax(Ylogits)
        aa = sess.run(tf.argmax(Y, 1))
        cc = sess.run(labels)
        accuracy = sum(aa == cc) / NUM_TESTING_IMAGES
    return accuracy

# the main process
# the center read the test images and labels
# read test dataset
testimages, testlabels = MnistInput(mnist_test_file, whole=True)
testimages = tf.convert_to_tensor(testimages, dtype=tf.float32)
testlabels = tf.convert_to_tensor(testlabels)
# initialize parameters and iter
iter = 0

initW1 = np.zeros((input, units[0]))
initb1 = np.zeros(units[0])
initW2 = np.zeros((units[0], units[1]))
initb2 = np.zeros(units[1])
initW3 = np.zeros((units[1], output))
initb3 = np.zeros(output)

weight = [initW1,initW2,initW3]
bias = [initb1,initb2,initb3]



# build the network
tf.Graph().as_default()
tf.device('/cpu:0')
lr = tf.placeholder(tf.float32)  # learning rate
# hard-coded the network structure (2 hidden layers)
W1 = tf.Variable(weight[0], dtype=tf.float32)  # first hidden layer
W2 = tf.Variable(weight[1], dtype=tf.float32)  # second hidden layer
W3 = tf.Variable(weight[2], dtype=tf.float32)  # output layer
b1 = tf.Variable(bias[0], dtype=tf.float32)
b2 = tf.Variable(bias[1], dtype=tf.float32)
b3 = tf.Variable(bias[2], dtype=tf.float32)

imageX = tf.placeholder(tf.float32, [None, 784], name="image")
labelY = tf.placeholder(tf.int64, [None, 10], name="label")

Y1 = tf.nn.relu(tf.matmul(imageX, W1) + b1)  # first layer
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)  # first layer
Ylogits = tf.matmul(Y2, W3) + b3  # output layer
Y = tf.nn.softmax(Ylogits)
# objective function (cross entropy)
cost = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=tf.one_hot(labelY, 10))
cost = tf.reduce_sum(cost) / batchSize



# record the accuracy
accuracy = []
accuracy.append(eval(testimages, testlabels))

# read user data
images1, labels1 = MnistInput(mnist_train_file, whole=False, start=1, size = usrSize[0])
images2, labels2 = MnistInput(mnist_train_file, whole=False, start=1 + usrSize[0], size = usrSize[1])
images3, labels3 = MnistInput(mnist_train_file, whole=False, start=1 + usrSize[0] + usrSize[1], size = usrSize[2])
images4, labels4 = MnistInput(mnist_train_file, whole=False, start=1 + usrSize[0] + usrSize[1] + usrSize[2], size = usrSize[3])
images5, labels5 = MnistInput(mnist_train_file, whole=False, start=1 + usrSize[0] + usrSize[1] + usrSize[2] + usrSize[3], size = usrSize[4])
images6, labels6 = MnistInput(mnist_train_file, whole=False, start=1 + usrSize[0] + usrSize[1] + usrSize[2] + usrSize[3] + usrSize[4], size = usrSize[5])

images1 = tf.convert_to_tensor(images1, dtype=tf.float32)
labels1 = tf.convert_to_tensor(labels1)
images2 = tf.convert_to_tensor(images2, dtype=tf.float32)
labels2 = tf.convert_to_tensor(labels2)
images3 = tf.convert_to_tensor(images3, dtype=tf.float32)
labels3 = tf.convert_to_tensor(labels3)
images4 = tf.convert_to_tensor(images4, dtype=tf.float32)
labels4 = tf.convert_to_tensor(labels4)
images5 = tf.convert_to_tensor(images5, dtype=tf.float32)
labels5 = tf.convert_to_tensor(labels5)
images6 = tf.convert_to_tensor(images6, dtype=tf.float32)
labels6 = tf.convert_to_tensor(labels6)

trainImages = [images1,images2,images3,images4,images5,images6]
trainLabels = [labels1,labels2,labels3,labels4,labels5,labels6]

# go through the users for iterations
for _ in range(totalIter):
    for i in range(users):
        W1_diff, W2_diff, W3_diff, b1_diff, b2_diff, b3_diff = \
            trainNN(trainImages[i], trainLabels[i], usrSize[i], learning_rate, grad_bound, lotSize, batchSize, epochs)

        W1_diff_noisy = np.array(noisyMax(W1_diff, len(W1_diff) * ratio, epsilon[i], grad_bound)).reshape(
            weight[0].shape)
        W2_diff_noisy = np.array(noisyMax(W2_diff, len(W2_diff) * ratio, epsilon[i], grad_bound)).reshape(
            weight[1].shape)
        W3_diff_noisy = np.array(noisyMax(W3_diff, len(W3_diff) * ratio, epsilon[i], grad_bound)).reshape(
            weight[2].shape)
        b1_diff_noisy = np.array(noisyMax(b1_diff, len(b1_diff) * ratio, epsilon[i], grad_bound)).reshape(bias[0].shape)
        b2_diff_noisy = np.array(noisyMax(b2_diff, len(b2_diff) * ratio, epsilon[i], grad_bound)).reshape(bias[1].shape)
        b3_diff_noisy = np.array(noisyMax(b3_diff, len(b3_diff) * ratio, epsilon[i], grad_bound)).reshape(bias[2].shape)

        # update the parameter
        weight[0] = weight[0] + W1_diff_noisy
        weight[1] = weight[1] + W2_diff_noisy
        weight[2] = weight[2] + W3_diff_noisy
        bias[0] = bias[0] + b1_diff_noisy
        bias[1] = bias[1] + b2_diff_noisy
        bias[2] = bias[2] + b3_diff_noisy

    accuracy.append(eval(testimages, testlabels))



print(accuracy)