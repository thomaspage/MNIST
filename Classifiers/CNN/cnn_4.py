import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import argparse
import sys
import tempfile
import matplotlib.pyplot as plt
import time
import shutil

train_x = 'train_x.csv'
train_y = 'train_y.csv'
test_x = 'test_x.csv'

def deepnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 64, 64, 1])
    tf.summary.image('img', x_image)

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([7, 7, 1, 64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    L2 = tf.nn.l2_loss(W_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([7, 7, 64, 128])
        b_conv2 = bias_variable([128])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    L2 = tf.add(L2, tf.nn.l2_loss(W_conv2))

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # with tf.name_scope('conv3'):
    #     W_conv3 = weight_variable([3, 3, 64, 128])
    #     b_conv3 = bias_variable([128])
    #     h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    # L2 = tf.add(L2, tf.nn.l2_loss(W_conv3))
    #
    # with tf.name_scope('conv4'):
    #     W_conv4 = weight_variable([3, 3, 128, 128])
    #     b_conv4 = bias_variable([128])
    #     h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
    # L2 = tf.add(L2, tf.nn.l2_loss(W_conv4))
    #
    # with tf.name_scope('conv5'):
    #     W_conv5 = weight_variable([3, 3, 128, 128])
    #     b_conv5 = bias_variable([128])
    #     h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
    # L2 = tf.add(L2, tf.nn.l2_loss(W_conv5))
    #
    # with tf.name_scope('pool3'):
    #     h_pool3 = max_pool_2x2(h_conv5)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([16 * 16 * 128, 2048])
        b_fc1 = bias_variable([2048])

        h_pool3_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    L2 = tf.add(L2, tf.nn.l2_loss(W_fc1))

    with tf.name_scope('dropout1'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([2048, 40])
        b_fc2 = bias_variable([40])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        # h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    L2 = tf.add(L2, tf.nn.l2_loss(W_fc2))

    # with tf.name_scope('dropout2'):
    #     keep_prob2 = tf.placeholder(tf.float32)
    #     h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob2)
    #
    # with tf.name_scope('fc3'):
    #     W_fc3 = weight_variable([2048, 40])
    #     b_fc3 = bias_variable([40])
    # L2 = tf.add(L2, tf.nn.l2_loss(W_fc3))

    return y_conv, keep_prob, L2

def conv2d(x, W, step):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, step, step, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def display_image(x):
    x = np.reshape(x, [-1, 64, 64])
    plt.figure()
    plt.imshow(x, 'grayscale')
    plt.show()
    return


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
    image = tf.multiply(image, 1.0/tf.reduce_max(image))

    return image, label

def main(_):
    filenames = tf.placeholder(tf.string, shape=[None])
    mnist = tf.data.TFRecordDataset(filenames)
    mnist = mnist.map(_parse_function, num_parallel_calls=8)
    mnist = mnist.repeat()
    mnist = mnist.shuffle(10000)
    mnist = mnist.batch(100)
    iterator = mnist.make_initializable_iterator()
    next_element = iterator.get_next()

    validation = tf.placeholder(tf.string, shape=[None])
    validation_set = tf.data.TFRecordDataset(validation)
    validation_set = validation_set.map(_parse_function, num_parallel_calls=8)
    validation_set = validation_set.batch(1000)
    valid_iterator = validation_set.make_initializable_iterator()
    next_test = valid_iterator.get_next()

    # Create the model
    x = tf.placeholder(tf.float32, [None, 4096])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 40])

    # Build the graph for the deep net
    y_conv, keep_prob, L2 = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)

    # loss = tf.add(tf.reduce_mean(cross_entropy), tf.multiply(L2, 0.01))
    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross entropy', loss)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-5, epsilon=0.001).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)

    # graph_location = tempfile.mkdtemp()
    # print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter('/Users/Matt/Assignments/551P3v.2/output')
    train_writer.add_graph(tf.get_default_graph())

    merge = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer, feed_dict={filenames: ['train.tfrecords']})

        start = time.time()

        for i in range(80000):
            batch = sess.run(next_element)

            if i % 100 == 0:
                train_accuracy, result = sess.run([accuracy, merge],feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                train_writer.add_summary(result, i)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})

        validation_filenames = ['validation.tfrecords']
        sess.run(valid_iterator.initializer, feed_dict={validation: validation_filenames})
        score = 0.0

        for i in range(10):
            test = sess.run(next_test)
            temp_score = accuracy.eval(feed_dict={
                x: test[0], y_: test[1], keep_prob: 1.0})
            print(temp_score)
            score += temp_score
            if i == 9:
                score = score/10
                print(score)
                print('test accuracy %g' % score)


        print("--- %s seconds ---" % (time.time() - start))

if __name__ == '__main__':
    tf.app.run(main=main)