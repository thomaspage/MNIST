import argparse
import sys
import tempfile
import struct
import numpy as np
from skimage import measure
from skimage import filters
from skimage import exposure
import os

import tensorflow as tf

FLAGS = None

def load_mnist(kind='train'):
    images_path = 'emnist-byclass-%s-images-idx3-ubyte' % kind
    labels_path = 'emnist-byclass-%s-labels-idx1-ubyte' % kind

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    digits = (labels == 0) | (labels == 1) | (labels == 2) | (labels == 3) | (labels == 4) | (labels == 5) | (labels == 6) | (labels == 7) | (labels == 8) | (labels == 9)
    letters = (labels == 10) | (labels == 22) | (labels == 36) | (labels == 48)

    images = images[digits | letters]
    labels = labels[digits | letters]
    labels = [12 if x == 22 else x for x in labels]
    labels = [12 if x == 48 else x for x in labels]
    labels = [10 if x == 36 else x for x in labels]

    return images, labels

def canny_edges(x):

    dim = int(np.sqrt(x.shape[0]))

    temp = exposure.adjust_gamma(x, (np.mean(x) ** 2) / 90 / 40)
    temp = temp.reshape(-1, dim, dim)
    temp = filters.gaussian(temp, 0.5)
    temp = (254 * np.divide(temp, np.max(temp))).astype(int)

    x = exposure.adjust_gamma(x, 15)
    x = x.reshape(-1, dim, dim)
    x = filters.gaussian(x, 0.5)
    x = (254 * np.divide(x, np.max(x))).astype(int)

    contours = measure.find_contours(temp[0], np.mean(temp) + (np.std(temp)), fully_connected='high')

    value = 225

    while not (len(contours) >= 3):
        contours = measure.find_contours(temp[0], value, fully_connected='high')
        value -= 1

        if value == 50:
            break

    contours.sort(key=len)
    contours = contours[-3:]

    point = []
    newimages = []

    if len(contours) == 3:
        for i in range(0, 3):
            point.append((np.max(contours[i], axis=0) + np.min(contours[i], axis=0)) / 2)

            x_coordinate = int(point[i][0])
            if x_coordinate < 14:
                x_coordinate = 14
            elif x_coordinate > 50:
                x_coordinate = 50
            else:
                x_coordinate = int(point[i][0])

            y_coordinate = int(point[i][1])
            if y_coordinate < 14:
                y_coordinate = 14
            elif y_coordinate > 50:
                y_coordinate = 50
            else:
                y_coordinate = int(point[i][1])

            newimages.append(x[0][x_coordinate - 14:x_coordinate + 14, y_coordinate - 14:y_coordinate + 14])

    else:
        newimages = np.zeros((3, 28, 28))

        # plot_contours(x, contours, newimages)
    newimages = np.asarray(newimages, dtype=np.float32)
    return newimages

def character_splitter(test_x):
    newimages = []
    for i in range(len(test_x)):
        newimages.append(canny_edges(test_x[i]))
        newimages[i] = np.array(newimages[i])
        newimages[i] = newimages[i].reshape(-1, 28 * 28)

    return newimages

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
  tf.summary.image('img', x_image, max_outputs=20)
  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 64, 128])
    b_conv2 = bias_variable([128])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 128, 2048])
    b_fc1 = bias_variable([2048])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([2048, 12])
    b_fc2 = bias_variable([12])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob, x_image

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


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

def parse_emnist(image, label):
    label = tf.one_hot(label, 12)

    image = tf.reshape(image, [28, 28, 1])
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.reshape(image, [784])

    return image, label

def parse_modmnist(example_proto):
    features = {
        "image_raw": tf.FixedLenFeature([4096], tf.float32),
        "label": tf.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    # decode back into array
    image = parsed_features["image_raw"]
    image = tf.multiply(image, 1.0/255.0)

    return image, parsed_features['label']

def split_images(example, label):
    example = canny_edges(example)
    example = np.reshape(example, [-1, 784])
    return example, label

def main(_):
  # Import data
  images, labels = load_mnist()
  mnist = tf.data.Dataset.from_tensor_slices((images, labels))
  mnist = mnist.map(parse_emnist, num_parallel_calls=4)
  mnist = mnist.shuffle(10000)
  mnist = mnist.repeat()
  mnist = mnist.batch(50)
  iterator = mnist.make_initializable_iterator()
  next_element = iterator.get_next()

  test_image, test_labels = load_mnist('test')
  test_mnist = tf.data.Dataset.from_tensor_slices((test_image, test_labels))
  test_mnist = test_mnist.map(parse_emnist, num_parallel_calls=4)
  test_mnist = test_mnist.batch(10000)
  test_iterator = test_mnist.make_initializable_iterator()
  test_element = test_iterator.get_next()

  validation = tf.placeholder(tf.string, shape=[None])
  validation_set = tf.data.TFRecordDataset(validation)
  validation_set = validation_set.map(parse_modmnist, num_parallel_calls=4)
  validation_set = validation_set.map(lambda example, label: tf.py_func(split_images, [example, label], [tf.float32, tf.int64]), num_parallel_calls=8)
  validation_set = validation_set.shuffle(50)
  validation_set = validation_set.batch(1)
  valid_iterator = validation_set.make_initializable_iterator()
  next_test = valid_iterator.get_next()

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 12])

  # Build the graph for the deep net
  y_conv, keep_prob, x_im = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  tf.summary.tensor_summary('cross', cross_entropy)
  cross_entropy = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-6, epsilon=0.001).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)
  tf.summary.scalar('accuracy', accuracy)

  with tf.name_scope('mod_accuracy'):
    char_1 = tf.argmax(y_conv, 1)

  # graph_location = tempfile.mkdtemp()
  # print('Saving graph to: %s' % graph_location)
  log_dir = '/Users/Matt/Assignments/551P3v.2/output'
  run_num = [f for f in os.listdir(log_dir)]
  run = '/run' + str(len(run_num) + 1)
  train_writer = tf.summary.FileWriter(log_dir + run)
  train_writer.add_graph(tf.get_default_graph())
  merge = tf.summary.merge_all()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)

    for i in range(80000):
      batch = sess.run(next_element)

      if i % 100 == 0:
        train_accuracy, summ = sess.run([accuracy, merge],feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
        train_writer.add_summary(summ, i)
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    sess.run(test_iterator.initializer)
    test = sess.run(test_element)
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: test[0], y_: test[1], keep_prob: 1.0}))

    sess.run(valid_iterator.initializer, feed_dict={validation: ['validation.tfrecords']})

    for i in range(10):
        modmnist_test = sess.run(next_test)
        result, summ = sess.run([char_1, merge], feed_dict={
            x: modmnist_test[0][0], y_: np.zeros([3, 12]), keep_prob: 1.0
        })
        print(result)
        print(modmnist_test[1])
        train_writer.add_summary(summ, 1)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)