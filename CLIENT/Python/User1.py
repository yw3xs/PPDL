# Python Users Functions for MIDDLE
# Author: Yang Wang

from __future__ import division
from CLIENT.Python.prep import *
import sys
import time
import numpy as np
import pyrebase
import tensorflow as tf
import os
import CLIENT.Python.config as config
import CLIENT.Python.noise as noise
import CLIENT.Python.dataPrep as dataPrep
import CLIENT.Python.utils as utils
import CLIENT.Python.per_example_gradients as per_example_gradients
import CLIENT.Python.dp_optimizer as dp_optimizer
import CLIENT.Python.sanitizer as sanitizer
import CLIENT.Python.download_and_convert_mnist as download_and_convert_mnist
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# define constants for the model and the user
batch_size = 500
sample_size = 10000

# define functions
def MnistInputAll(mnist_data_file):
  """Create operations to read the MNIST input file.

  Args:
    mnist_data_file: Path of a tfrecord file containing the MNIST images to process.

  Returns:
    images: A list with the formatted image data. shape [10000, 28*28]
    labels: A list with the labels for each image.  shape [10000]
  """
  with tf.Session() as sess:
      file_queue = tf.train.string_input_producer([mnist_data_file], num_epochs= 1)
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

      return images, labels


# link to firebase database
print 'Linking to Firebase database'
firebase = pyrebase.initialize_app(config.config)
db = firebase.database()
print ' '
print 'Downloading model from server'
model = dict(db.child('model').get().val())
print model

# set up the child database
initGrad1 = np.zeros(model['param_num'][0])  # first layer
initGrad2 = np.zeros(model['param_num'][1])
uid = 'py1'
userIter = -1  # userIter is the parameter iteration on which the computation of gradient is based. e.g., if userIter = 10, that meanes the gradient is calculated using the parameter in the 10th iteration.
userData = {
    'users/' + uid: {
        'userID': uid,
        'userIter': userIter,
        'grad1': list(initGrad1),
        'grad2': list(initGrad2),
        }
}
db.update(userData)

# read the data set
user1 = '/home/yang/Research/Privacy-preserving-DL/MIDDLE_DNN/MIDDLE/CLIENT/Python/userData/user1/mnist_train.tfrecord'
images, labels = MnistInputAll(user1)  # images and labels are lists
testimages, testlabels = MnistInputAll(mnist_test_file)
images = tf.convert_to_tensor(images, dtype=tf.float32)
labels = tf.convert_to_tensor(labels)
testimages = tf.convert_to_tensor(testimages, dtype=tf.float32)
testlabels = tf.convert_to_tensor(testlabels)

#################################### re-train from here ###################################
def train():
    print "Training begins"
    serverParam1 = db.child('parameter').child('parameter1').get().val()
    serverParam2 = db.child('parameter').child('parameter2').get().val()
    serverParam1 = np.reshape(serverParam1, (784,64))
    serverParam2 = np.reshape(serverParam2, (64,10))
    serverIter = db.child('parameter').child('iteration').get().val()

    tf.Graph().as_default()
    tf.device('/cpu:0')
    sess = tf.Session()
    def eval(images, labels, param1, param2):
        Y1 = tf.nn.relu(tf.matmul(images, param1))  # first layer
        Ylogits = tf.matmul(Y1, param2)  # output layer
        Y = tf.nn.softmax(Ylogits)
        aa = sess.run(tf.argmax(Y, 1))
        cc = sess.run(labels)
        accuracy = sum(aa == cc) / 10000
        return accuracy
    lr = tf.placeholder(tf.float32)  # learning rate
    round = int(sample_size / batch_size)
    # initialize the parameters using the server parameters

    W1 = tf.Variable(serverParam1, dtype=tf.float32)  # second layer
    W2 = tf.Variable(serverParam2, dtype=tf.float32)  # third layer

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # evaluate the accuracy
    for _ in xrange(epochs):
        print "Accuracy when started " + ":"
        print eval(testimages, testlabels, W1, W2)
        for i in xrange(round):
            # sample one batch
            image_batch = images[(i*batch_size):((i+1)*batch_size), :]
            label_batch = labels[(i*batch_size):((i+1)*batch_size)]
            # build the network
            Y1 = tf.nn.relu(tf.matmul(image_batch, W1))  # first layer
            Ylogits = tf.matmul(Y1, W2)  # output layer
            Y = tf.nn.softmax(Ylogits)
            # objective function (cross entropy)
            cost = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=tf.one_hot(label_batch, 10))
            cost = tf.reduce_sum(cost) / batch_size
            op1 = tf.train.GradientDescentOptimizer(lr).compute_gradients(cost)
            print op1
            op2 = tf.train.GradientDescentOptimizer(lr).apply_gradients(op1)
            # saver = tf.train.Saver()
            # coord = tf.train.Coordinator()
            # _ = tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(op1) # grad[0-1](W1-W2)[0-1](grad + var)
            sess.run([op2], feed_dict={lr: learning_rate})

    print "Accuracy after one epoch " + ":"
    print eval(testimages, testlabels, W1, W2)

    #####################################################
    newParam1 = sess.run(W1)
    newParam2 = sess.run(W2)

    Param1_diff = newParam1 - serverParam1   # range: -0.06777553 ~ 0.1147576175
    Param1_diff = Param1_diff.flatten()
    Param2_diff = newParam2 - serverParam2  # range:  -0.33470168 ~ 0.506309772
    Param2_diff = Param2_diff.flatten()
    # privacy analysis
    # add noise to and clip the gradient
    tf.set_random_seed(0)
    noise1 = np.random.laplace(loc = 0.0, scale = 2*grad_upload_num*2*grad_bound/epsilon/8*9)
    noise2 = np.random.laplace(loc = 0.0, scale = 2*2*grad_upload_num*2*grad_bound/epsilon/8*9)
    grad1_upload = np.zeros(len(Param1_diff))
    grad2_upload = np.zeros(len(Param2_diff))
    for i in xrange(len(Param1_diff)):
        if abs(np.clip(Param1_diff[i], -grad_bound, grad_bound)) + noise2 >= (grad_threshold + noise1):
            noise3 = np.random.laplace(loc = 0.0, scale = 2*grad_upload_num*2*grad_bound/epsilon*9)
            grad1_upload[i] = np.clip(Param1_diff[i] + noise3, -grad_bound, grad_bound)
    for i in xrange(len(Param2_diff)):
        if abs(np.clip(Param2_diff[i], -grad_bound, grad_bound)) + noise2 >= grad_threshold + noise1:
            noise3 = np.random.laplace(loc=0.0, scale=2 * grad_upload_num * 2 * grad_bound / epsilon * 9)
            grad2_upload[i] = np.clip(Param2_diff[i] + noise3, -grad_bound, grad_bound)

    grad_upload = np.concatenate((grad1_upload,grad2_upload))
    index = np.argsort(grad_upload)[::-1][int(grad_upload_num/2):(len(grad_upload) - int(grad_upload_num/2))]
    grad_upload[index] = 0

    grad1_upload = grad_upload[:len(grad1_upload)]
    grad2_upload = grad_upload[len(grad1_upload):]

    userIter = serverIter + 1
    userData['users/' + uid]['userIter'] = userIter
    if with_privacy:
        userData['users/' + uid]['grad1'] = list(grad1_upload)
        userData['users/' + uid]['grad2'] = list(grad2_upload)
    else:
        userData['users/' + uid]['grad1'] = list(Param1_diff)
        userData['users/' + uid]['grad2'] = list(Param2_diff)
    db.update(userData)
    print "Training end ..."
    sess.close()


train()