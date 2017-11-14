import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import struct
from skimage import measure
from skimage import filters
from skimage import exposure
import os

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

def deepnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('img', x_image, max_outputs=20)

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 1, 64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 1) + b_conv1)
    L2 = tf.nn.l2_loss(W_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 64, 128])
        b_conv2 = bias_variable([128])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 1) + b_conv2)
    L2 = tf.add(L2, tf.nn.l2_loss(W_conv2))

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 128, 256])
        b_conv3 = bias_variable([256])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
    L2 = tf.add(L2, tf.nn.l2_loss(W_conv3))

    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([3, 3, 256, 256])
        b_conv4 = bias_variable([256])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)
    L2 = tf.add(L2, tf.nn.l2_loss(W_conv4))

    # with tf.name_scope('conv5'):
    #     W_conv5 = weight_variable([3, 3, 256, 512])
    #     b_conv5 = bias_variable([512])
    #     h_conv5 = tf.nn.relu(conv2d(h_pool3, W_conv5, 1) + b_conv5)
    # L2 = tf.add(L2, tf.nn.l2_loss(W_conv5))
    #
    # with tf.name_scope('conv6'):
    #     W_conv6 = weight_variable([3, 3, 512, 512])
    #     b_conv6 = bias_variable([512])
    #     h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6, 1) + b_conv6)
    # L2 = tf.add(L2, tf.nn.l2_loss(W_conv6))

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 256, 2048])
        b_fc1 = bias_variable([2048])

        h_conv6_flat = tf.reshape(h_conv4, [-1, 7 * 7 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv6_flat, W_fc1) + b_fc1)
    L2 = tf.add(L2, tf.nn.l2_loss(W_fc1))

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([2048, 2048])
        b_fc2 = bias_variable([2048])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    L2 = tf.add(L2, tf.nn.l2_loss(W_fc2))

    with tf.name_scope('dropout'):
        keep_prob2 = tf.placeholder(tf.float32)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob2)

    with tf.name_scope('fc3'):
        W_fc3 = weight_variable([2048, 12])
        b_fc3 = bias_variable([12])

        y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    return y_conv, keep_prob, keep_prob2, L2

def conv2d(x, W, step):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, step, step, 1], padding='SAME')

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.random_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(1.0)
    return tf.Variable(initial)

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
    image = tf.reshape(image, [64, 64, 1])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.reshape(image, [4096])
    image = tf.multiply(image, 1.0/tf.reduce_max(image))

    return image, label

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
    validation_set = validation_set.map(
        lambda example, label: tf.py_func(split_images, [example, label], [tf.float32, tf.int64]), num_parallel_calls=8)
    validation_set = validation_set.shuffle(50)
    validation_set = validation_set.batch(1)
    valid_iterator = validation_set.make_initializable_iterator()
    next_test = valid_iterator.get_next()

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 12])

    # Build the graph for the deep net
    y_conv, keep_prob, keep_prob2, L2 = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)

    # loss = tf.add(tf.reduce_mean(cross_entropy), tf.multiply(L2, 0.01))
    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross entropy', loss)

    with tf.name_scope('adam_optimizer'):
        alpha = tf.placeholder(tf.float32)
        train_step = tf.train.AdamOptimizer(alpha, epsilon=0.1).minimize(loss)

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
    run = '/run' + str(len(run_num)+1)
    train_writer = tf.summary.FileWriter(log_dir + run)
    train_writer.add_graph(tf.get_default_graph())

    merge = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)

        for i in range(80000):
            batch = sess.run(next_element)

            if i < 40000:
                if i % 100 == 0:
                    train_accuracy, summ = sess.run([accuracy, merge], feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0, alpha: 1e-5, keep_prob2: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                    train_writer.add_summary(summ, i)
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, alpha: 1e-6, keep_prob2: 0.5})
            else:
                if i % 100 == 0:
                    train_accuracy, summ = sess.run([accuracy, merge], feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0, alpha: 1e-7, keep_prob2: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                    train_writer.add_summary(summ, i)
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, alpha: 1e-7, keep_prob2: 0.5})

        sess.run(test_iterator.initializer)
        test = sess.run(test_element)
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: test[0], y_: test[1], keep_prob: 1.0, keep_prob2: 1.0}))

        sess.run(valid_iterator.initializer, feed_dict={validation: ['validation.tfrecords']})

        for i in range(10):
            modmnist_test = sess.run(next_test)
            result, summ = sess.run([char_1, merge], feed_dict={
                x: modmnist_test[0][0], y_: np.zeros([3, 12]), keep_prob: 1.0, alpha: 1e-6, keep_prob2:1.0
            })
            print(result)
            print(modmnist_test[1])
            train_writer.add_summary(summ, 1)

if __name__ == '__main__':
    tf.app.run(main=main)