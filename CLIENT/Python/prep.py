# Python Users Preparation for MIDDLE
# Author: Yang Wang
NUM_TRAINING_IMAGES = 60000
NUM_TESTING_IMAGES = 10000
IMAGE_SIZE = 28
mnist_test_file = '/home/yang/Research/Privacy-preserving-DL/MIDDLE_DNN/MIDDLE/DATA/MNIST/mnist_test.tfrecord'
epochs = 1
epsilon = 1 # privacy budget for each epoch
learning_rate = .1
grad_bound = .05
grad_threshold = .0001
grad_upload_ratio = .01 # upload gradients for 0.1% of the parameters each time. In this case,
grad_upload_num = int((784*64 + 640) * grad_upload_ratio)  # around
with_privacy = True

