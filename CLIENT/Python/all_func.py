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
import CLIENT.Python.config as config
import CLIENT.Python.noise as noise
import CLIENT.Python.dataPrep as dataPrep
import CLIENT.Python.utils as utils
import CLIENT.Python.per_example_gradients as per_example_gradients
import CLIENT.Python.dp_optimizer as dp_optimizer
import CLIENT.Python.sanitizer as sanitizer
import CLIENT.Python.download_and_convert_mnist as download_and_convert_mnist

# constants
NUM_TRAINING_IMAGES = 60000
NUM_TESTING_IMAGES = 10000
IMAGE_SIZE = 28
mnist_train_file = '/home/yang/Research/Privacy-preserving-DL/PPDL/DATA/MNIST/mnist_train.tfrecord'
mnist_test_file = '/home/yang/Research/Privacy-preserving-DL/PPDL/DATA/MNIST/mnist_test.tfrecord'

# parameters
batch_size = 500
epochs = 3   ## ?? ##
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


# build the neural network given the model parameters and calculate the gradient
def buildNN(units,layers,parameters,iter):
    """Create operations to read the MNIST input file.
        Args:
            units: number of units in each hidden layer
            layers: number of hidden layers
            parameters: weights and bias for each layer (list of list)
            iter: iteration number

        Returns:
            images: A list with the formatted image data. default shape [10000, 28*28]
            labels: A list with the labels for each image. default shape [10000]
    """
    tf.Graph().as_default()
    tf.device('/cpu:0')
    with tf.Session() as sess:
        lr = tf.placeholder(tf.float32)  # learning rate
        eps = tf.placeholder(tf.float32)  # epsilon (privacy parameter)
        delta = tf.placeholder(tf.float32)  # delta (privacy parameter)
        # hard-coded the network structure
        W1 = tf.Variable(parameters[0], dtype=tf.float32)  # second layer
        W2 = tf.Variable(parameters[1], dtype=tf.float32)  # third layer
        init = tf.global_variables_initializer()
        init_2 = tf.local_variables_initializer()
        sess.run(init)
        sess.run(init_2)









#images, labels = MnistInputAll(user1)  # images and labels are lists
#testimages, testlabels = MnistInputAll(mnist_test_file)

images = tf.convert_to_tensor(images, dtype=tf.float32)
labels = tf.convert_to_tensor(labels)

testimages = tf.convert_to_tensor(testimages, dtype=tf.float32)
testlabels = tf.convert_to_tensor(testlabels)

##################### Start training the model ################################
# 1. set up the network
# 2. start the iteration
#    2a. Initialize variables with server parameter
#    2b. sample batch size data points
#    2c. calculate the gradient, apply the gradient
##################################################################################

tf.Graph().as_default()
tf.device('/cpu:0')
sess = tf.Session()

while True:
    lr = tf.placeholder(tf.float32)  # learning rate
    eps = tf.placeholder(tf.float32)  # epsilon
    delta = tf.placeholder(tf.float32)  # delta

    round = int(sample_size / batch_size)

    # initialize the parameters using the server parameters
    serverParam1 = db.child('parameter').child('parameter1').get().val()  # list with length 50176
    serverParam2 = db.child('parameter').child('parameter2').get().val()
    serverIter = db.child('parameter').child('iteration').get().val()
    print "iter: " + str(serverIter)
    serverParam1 = np.reshape(serverParam1, (784,64))
    serverParam2 = np.reshape(serverParam2, (64,10))

    W1 = tf.Variable(serverParam1, dtype=tf.float32)  # second layer
    W2 = tf.Variable(serverParam2, dtype=tf.float32)  # third layer
    init = tf.global_variables_initializer()
    init_2 = tf.local_variables_initializer()
    sess.run(init)
    sess.run(init_2)

    # evaluate the accuracy
    def eval(images, labels, param1, param2):
        Y1 = tf.nn.relu(tf.matmul(images, param1))  # first layer
        Ylogits = tf.matmul(Y1, param2)  # output layer
        Y = tf.nn.softmax(Ylogits)
        aa = sess.run(tf.argmax(Y, 1))
        cc = sess.run(labels)
        accuracy = sum(aa == cc) / 10000
        return accuracy
    epochs = 1
    for _ in xrange(epochs):
        print "Epoch " + str(_) + " starts!"
        print "accuracy on epoch " + str(_) + ":"
        print eval(testimages, testlabels, W1, W2)
        for i in xrange(round):
            #print "epoch: "+ str(_) + " round " + str(i)
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
            op2 = tf.train.GradientDescentOptimizer(lr).apply_gradients(op1)
            # saver = tf.train.Saver()
            # coord = tf.train.Coordinator()
            # _ = tf.train.start_queue_runners(sess=sess, coord=coord)
            grad = sess.run(op1) # grad[0-1](W1-W2)[0-1](grad + var)
            sess.run([op2], feed_dict={lr: learning_rate})
        print "Epoch " + str(_) + " ends!"
    print "accuracy on epoch " + str(epochs) + ":"
    print eval(testimages, testlabels, W1, W2)

    #####################################################
    newParam1 = sess.run(W1)
    newParam2 = sess.run(W2)

    Param1_diff = newParam1 - serverParam1   # range: -0.06777553 ~ 0.1147576175
    Param1_diff = Param1_diff.flatten()
    Param1_diff.max()
    Param1_diff.min()
    np.argmax(Param1_diff)
    np.argmin(Param1_diff)

    Param2_diff = newParam2 - serverParam2  # range:  -0.33470168 ~ 0.506309772
    Param2_diff = Param2_diff.flatten()
    Param2_diff.max()
    Param2_diff.min()
    np.argmax(Param2_diff)
    np.argmin(Param2_diff)

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


    userIter += 1
    userData['users/' + uid]['userIter'] = userIter
    userData['users/' + uid]['grad1'] = list(Param1_diff)
    userData['users/' + uid]['grad2'] = list(Param2_diff)
    #userData['users/' + uid]['grad1'] = list(grad1_upload)
    #userData['users/' + uid]['grad2'] = list(grad2_upload)
    db.update(userData)


###########################################################################

#############################
#     logits, projection, training_params = utils.BuildNetwork(image_batch, network_parameters)
#     cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(label_batch, 10))
#     cost = tf.reduce_sum(cost, [0]) / batch_size
#
#
#     var_list = tf.trainable_variables()
#     var_list[0] = tf.Variable(serverParam1, dtype=tf.float32)
#     var_list[1] = tf.Variable(serverParam2, dtype=tf.float32)
#     var_list[2] = tf.Variable(serverParam3, dtype=tf.float32)
#     print '============'
#     print var_list
#     sess.close()
#
#     xs = [tf.convert_to_tensor(x) for x in var_list]
# #px_grads = per_example_gradients.PerExampleGradients(cost, xs)
#
# # initialize all variables
#

# optimizer = tf.train.GradientDescentOptimizer(0.5)
#
# gd_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
#
#
# test_op1 = tf.train.GradientDescentOptimizer(lr).compute_gradients(cost)
# test_op2 = tf.train.GradientDescentOptimizer(lr).apply_gradients(test_op1)
# global_step = tf.Variable(0, dtype=tf.int32, trainable=False,name="global_step")
#
# init = tf.global_variables_initializer()
# init_2 = tf.local_variables_initializer()
# sess.run(init)
# sess.run(init_2)
#
# saver = tf.train.Saver()
# coord = tf.train.Coordinator()
# _ = tf.train.start_queue_runners(sess=sess, coord=coord)
#
# grad = sess.run(test_op1)
# sess.run([test_op2], feed_dict={lr: .1})
# ############################
#
#
# # gd_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
#
#
# test_op1 = tf.train.GradientDescentOptimizer(lr).compute_gradients(cost)
# test_op2 = tf.train.GradientDescentOptimizer(lr).apply_gradients(test_op1)
# global_step = tf.Variable(0, dtype=tf.int32, trainable=False,name="global_step")
#
# init = tf.global_variables_initializer()
# init_2 = tf.local_variables_initializer()
# sess.run(init)
# sess.run(init_2)
#
# saver = tf.train.Saver()
# coord = tf.train.Coordinator()
# _ = tf.train.start_queue_runners(sess=sess, coord=coord)
#
# grad = sess.run(test_op1)
# sess.run([test_op2], feed_dict={lr: .1})
#
#
#
#
#
#
#
#
#
#
#     #
#     # #############3
#     # sanitized_grads = []
#     # for px_grad, v in zip(px_grads, var_list):
#     #     tensor_name = utils.GetTensorOpName(v)
#     #     sanitized_grad = self._sanitizer.sanitize(
#     #         px_grad, self._eps_delta, sigma=self._sigma,
#     #         tensor_name=tensor_name, add_noise=add_noise,
#     #         num_examples=self._batches_per_lot * tf.slice(
#     #             tf.shape(px_grad), [0], [1]))
#     #     sanitized_grads.append(sanitized_grad)
#     #
#     # # Add global_step
#     # global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
#     #                           name="global_step")
#     #
#     # if with_privacy:
#     #   gd_op = dp_optimizer.DPGradientDescentOptimizer(
#     #       lr,
#     #       [eps, delta],
#     #       gaussian_sanitizer,
#     #       sigma=sigma,
#     #       batches_per_lot=FLAGS.batches_per_lot).minimize(
#     #           cost, global_step=global_step)
#     # else:
#     #   gd_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
#     #
#     # saver = tf.train.Saver()
#     # coord = tf.train.Coordinator()
#     # _ = tf.train.start_queue_runners(sess=sess, coord=coord)
#     #
#     # for v in tf.trainable_variables():
#     #     sess.run(tf.variables_initializer([v]))
#     # sess.run(tf.global_variables_initializer())
#     #
#     # results = []
#     # start_time = time.time()
#     # prev_time = start_time
#     # filename = "results-0.json"
#     # log_path = os.path.join(save_path, filename)
#     #
#     # lot_size = FLAGS.batches_per_lot * FLAGS.batch_size
#     # lots_per_epoch = NUM_TRAINING_IMAGES / lot_size
#     #
#     # for step in xrange(num_steps):
#     #     epoch = step / lots_per_epoch
#     #     curr_lr = utils.VaryRate(FLAGS.lr, FLAGS.end_lr,
#     #                              FLAGS.lr_saturate_epochs, epoch)
#     #     curr_eps = utils.VaryRate(FLAGS.eps, FLAGS.end_eps,
#     #                               FLAGS.eps_saturate_epochs, epoch)
#     #     for _ in xrange(FLAGS.batches_per_lot):
#     #         _ = sess.run(
#     #             [gd_op], feed_dict={lr: curr_lr, eps: curr_eps, delta: FLAGS.delta})
#     #     sys.stderr.write("step: %d\n" % step)
#     #
#     #     # See if we should stop training due to exceeded privacy budget:
#     #     should_terminate = False
#     #     terminate_spent_eps_delta = None
#     #     if with_privacy and FLAGS.terminate_based_on_privacy:
#     #         terminate_spent_eps_delta = priv_accountant.get_privacy_spent(
#     #             sess, target_eps=[max_target_eps])[0]
#     #         # For the Moments accountant, we should always have
#     #         # spent_eps == max_target_eps.
#     #         if (terminate_spent_eps_delta.spent_delta > FLAGS.target_delta or
#     #                     terminate_spent_eps_delta.spent_eps > max_target_eps):
#     #             should_terminate = True
#     #
#     #     if (eval_steps > 0 and (step + 1) % eval_steps == 0) or should_terminate:
#     #         if with_privacy:
#     #             spent_eps_deltas = priv_accountant.get_privacy_spent(
#     #                 sess, target_eps=target_eps)
#     #         else:
#     #             spent_eps_deltas = [accountant.EpsDelta(0, 0)]
#     #         for spent_eps, spent_delta in spent_eps_deltas:
#     #             sys.stderr.write("spent privacy: eps %.4f delta %.5g\n" % (
#     #                 spent_eps, spent_delta))
#     #
#     #         saver.save(sess, save_path=save_path + "/ckpt")
#     #         train_accuracy, _ = Eval(mnist_train_file, network_parameters,
#     #                                  num_testing_images=NUM_TESTING_IMAGES,
#     #                                  randomize=True, load_path=save_path)
#     #         sys.stderr.write("train_accuracy: %.2f\n" % train_accuracy)
#     #         test_accuracy, mistakes = Eval(mnist_test_file, network_parameters,
#     #                                        num_testing_images=NUM_TESTING_IMAGES,
#     #                                        randomize=False, load_path=save_path,
#     #                                        save_mistakes=FLAGS.save_mistakes)
#     #         sys.stderr.write("eval_accuracy: %.2f\n" % test_accuracy)
#     #
#     #         curr_time = time.time()
#     #         elapsed_time = curr_time - prev_time
#     #         prev_time = curr_time
#     #
#     #         results.append({"step": step + 1,  # Number of lots trained so far.
#     #                         "elapsed_secs": elapsed_time,
#     #                         "spent_eps_deltas": spent_eps_deltas,
#     #                         "train_accuracy": train_accuracy,
#     #                         "test_accuracy": test_accuracy,
#     #                         "mistakes": mistakes})
#     #         loginfo = {"elapsed_secs": curr_time - start_time,
#     #                    "spent_eps_deltas": spent_eps_deltas,
#     #                    "train_accuracy": train_accuracy,
#     #                    "test_accuracy": test_accuracy,
#     #                    "num_training_steps": step + 1,  # Steps so far.
#     #                    "mistakes": mistakes,
#     #                    "result_series": results}
#     #         loginfo.update(params)
#     #         if log_path:
#     #             with tf.gfile.Open(log_path, "w") as f:
#     #                 json.dump(loginfo, f, indent=2)
#     #                 f.write("\n")
#     #                 f.close()
#     #
#     #
#
#
#
#
#
#
#
# ##################################################################################
# # Train model, and retrieve/upload the parameter and gradients
# # workflow:
# # 1. set up the model and load dataset
# # 2. start the iteration
# #    2a. read current parameters from server (check paramIter, if it doesn't change, then do nothing and repeat the loop)
# #    2b. sample batch size data points
# #    2c. calculate the gradient, and add noise to protect privacy
# #    2d. upload the gradient with noise to the server
# #    2e. set the necessary flag to indicate the progress
# ##################################################################################
# def trainModel(uid, offtime, offcount, refresh, samplesIndex):
#     '''
#     uid: the user id
#     offtime: the sleep time of the user (simulates the device turn-off)
#     offcount: after every 'offcount' iteration, the user will sleep for 'offtime'
#     refresh: the user will check out the parameters from the server once every 'refresh' seconds
#     samplesIndex: the row index of the training set samples belonging to the user. each user owns a subset of the original MNIST training set.
#     '''
#     # initialize the user database
#     print uid
#     userIter = -1 # userIter is the parameter iteration on which the computation of gradient is based. e.g., if userIter = 10, that meanes the gradient is calculated using the parameter in the 10th iteration.
#     initGrad = np.zeros(model['D'])
#
#     userData = {
#         'users/' + uid :{
#             'userID': uid,
#             'userIter': userIter,
#             'gradients': list(initGrad)
#         }
#     }
#     db.update(userData)
#
#     ########
#     # load data
#     print 'Loading training data'
#     datafile = '../../DATA/MNIST/MNIST_train.csv'
#     X = dataPrep.loadData(datafile, samplesIndex)
#     print X.shape  # X is the first 1000 rows of the MNIST training dataset with dimension (1000, 785). The first column is the label
#
#     if (model['modelName'] == 'logReg'):
#         import loss_logreg # logistic regression
#         # conver the old label to binary label
#         X = loss_logreg.binLabel(X, 0, model['baseValue'])
#     else:
#         print 'Unknown model'
#         exit()
#
#     count = 0
#     while True:
#         # check out the model every 'refresh' seconds, turn off the user every offcount for offtime.
#         if count == offcount:
#             time.sleep(offtime)
#             count = 0
#         time.sleep(refresh)
#         # check out server parameters
#         print 'Checking out parameters from server'
#         serverParam = db.child('parameter').child('parameters').get().val()
#         serverIter = db.child('parameter').child('iteration').get().val()
#         if serverIter <= userIter:
#             break;
#         else:
#             # sample batchSize data points from the dataset
#             sample = dataPrep.sampleData(X, len(samplesIndex), model['batchSize'])
#             # calculate the combined gradient for the batch
#             grad = loss_logreg.getGradient(serverParam, sample, model['l'])
#             # add noise to gradient
#             grad = grad + noise.GenerateLaplaceNoise(len(serverParam), model['eps'], model['batchSize'])
#             # convert the grad to a json object and upload it to server(Firebase database)
#             userIter = serverIter
#             userData['users/' + uid]['userIter'] = userIter
#             userData['users/' + uid]['gradients'] = list(grad)
#             db.update(userData)
#             count += 1
