class Base(object):
    def __init__(self):
        print "Base created"

class ChildA(Base):
    def __init__(self):
        Base.__init__(self)

class ChildB(Base):
    def __init__(self):
        super(ChildB, self).__init__()
        print "hello"

ChildA()
ChildB()


class A(object):
    def foo(self):
        print 'A'


class B(A):
    def foo(self):
        print 'B'
        super(B, self).foo()


class C(A):
    def foo(self):
        print 'C'
        super(C, self).foo()


class D(B, C):
    def foo(self):
        print 'D'
        super(D, self).foo()


d = D()
d.foo()

import tensorflow as tf

tf.flags.DEFINE_integer("batch_size", 600, "The training batch size.")

FLAGS = tf.flags.FLAGS
print FLAGS.batch_size

import tensorflow

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.ones([784,10]))
b = tf.Variable(tf.ones([10]))

sess.run(tf.global_variables_initializer())
y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

optimizer = tf.train.GradientDescentOptimizer(0.5)
batch = mnist.train.next_batch(100)

test1 = sess.run(optimizer.compute_gradients(cross_entropy), feed_dict={x: batch[0], y_: batch[1]})


import tensorflow as tf

def read_and_decode(filename_queue):
 reader = tf.TFRecordReader()
 _, serialized_example = reader.read(filename_queue)
 features = tf.parse_single_example(
  serialized_example,
  features={
      'image_raw': tf.FixedLenFeature([], tf.string)
  })
 image = tf.decode_raw(features['image_raw'], tf.uint8)
 return image


def get_all_records(FILE):
 with tf.Session() as sess:
   filename_queue = tf.train.string_input_producer([FILE], num_epochs=1)
   image = read_and_decode(filename_queue)
   init_op = tf.initialize_all_variables()
   sess.run(init_op)
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(coord=coord)
   try:
     while True:
       example = sess.run([image])
   except tf.errors.OutOfRangeError, e:
     coord.request_stop(e)
   finally:
   coord.request_stop()
   coord.join(threads)


get_all_records('/path/to/train-0.tfrecords')

def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return tf.matmul(arg, arg) + arg

import tensorflow as tf
a = tf.constant(5.0)
b = tf.constant(6.0)
w = tf.Variable(111, name='test')
c = a * b

# Launch the graph in a session.
sess = tf.Session()

# Evaluate the tensor `c`.
print(sess.run(c))