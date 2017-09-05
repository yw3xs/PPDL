from __future__ import division
import numpy as np
import sys, json
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

NUM_TRAINING_IMAGES = 60000
NUM_TESTING_IMAGES = 10000
IMAGE_SIZE = 28

mnist_test_file = '/home/yang/Research/Privacy-preserving-DL/MIDDLE_DNN/MIDDLE/DATA/MNIST/mnist_test.tfrecord'

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


testimages, testlabels = MnistInputAll(mnist_test_file)
testimages = tf.convert_to_tensor(testimages, dtype=tf.float32)
testlabels = tf.convert_to_tensor(testlabels)
    
def eval(images, labels, param1, param2):
    tf.Graph().as_default()
    tf.device('/cpu:0')
    sess = tf.Session()
    param1 = np.reshape(param1, (784, 64))
    param2 = np.reshape(param2, (64, 10))
    W1 = tf.Variable(param1, dtype=tf.float32)  # second layer
    W2 = tf.Variable(param2, dtype=tf.float32)  # third layer
    init = tf.global_variables_initializer()
    init_2 = tf.local_variables_initializer()
    sess.run(init)
    sess.run(init_2)
    Y1 = tf.nn.relu(tf.matmul(images, W1))  # first layer
    Ylogits = tf.matmul(Y1, W2)  # output layer
    Y = tf.nn.softmax(Ylogits)
    aa = sess.run(tf.argmax(Y, 1))
    cc = sess.run(labels)
    accuracy = sum(aa == cc) / 10000
    return accuracy

def read_server():
    serverInput = sys.stdin.readlines()    
    return [json.loads(serverInput[0]), json.loads(serverInput[1])]
    
def main():  
    param1, param2 = read_server()

    accuracy = eval(testimages, testlabels,  param1, param2)
    print accuracy

if __name__ == '__main__':
    main()

