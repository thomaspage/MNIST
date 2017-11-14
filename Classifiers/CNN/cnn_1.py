from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
import sys
import os

import matplotlib.pyplot as plt
from scipy import misc

modMNIST_train_x = "train_x.csv"
modMNIST_train_y = "train_y.csv"
modMNIST_test_x = "test_x.csv"


# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
TEST_FILE = 'test.tfrecords'

def _parse_function(example_proto):
    features = {
        "image_raw": tf.FixedLenFeature([4096], tf.float32),
        "label": tf.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, features)

    # convert to one-hot encoding
    label = tf.one_hot(parsed_features["label"], 40)

    # decode back into array
    image = parsed_features["image_raw"]
    image = tf.multiply(image, 1.0/ 255.0)
    image = tf.reshape(image, [64, 64])
    return image, label

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def main(_):

    # X_test = np.loadtxt(modMNIST_test_x, delimiter=',')
    # X_test = tf.reshape(X_test, [-1, 4096])
    # y_train = np.zeros(X_test.shape[0])
    #
    # mnist = tf.data.Dataset.from_tensors((X_test, y_train))
    # mnist = mnist.repeat()
    # mnist = mnist.batch(50)
    #
    # iterator = tf.data.Iterator.from_structure(mnist.output_types, mnist.output_shapes)
    # next_element = iterator.get_next()
    #
    # init_op = iterator.make_initializer(mnist)
    #
    #
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #
    #     for _ in range(10):
    #         element = sess.run(next_element)
    #         print(element[0].shape)
    #         print(element)
    #         print(element[0])

    filenames = tf.placeholder(tf.string, shape=[None])
    mnist = tf.data.TFRecordDataset(filenames)
    mnist = mnist.cache()
    mnist = mnist.map(_parse_function, num_parallel=4)
    mnist = mnist.batch(50)
    iterator = mnist.make_initializable_iterator()
    next_element = iterator.get_next()


    with tf.Session() as sess:
        training_files = ['train.tfrecords']
        sess.run(iterator.initializer, feed_dict={filenames: training_files})
        for _ in range(5):
            print(sess.run(next_element[0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
