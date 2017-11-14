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

import dataset
tf.logging.set_verbosity(tf.logging.INFO)

modMNIST_train_x = "train_x.csv"
modMNIST_train_y = "train_y.csv"
modMNIST_test_x = "test_x.csv"

def cnn_model(features, labels, mode):
    # Input layer
    input_layer = tf.reshape(features['x'], [-1, 64, 64, 1])

    # Convolutional Layer 1
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu
    )

    # Pooling Layer 1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,2],
        strides=2
    )

    # Convolutional Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3,3],
        padding='same',
        activation=tf.nn.relu
    )

    # Pooling Layer 2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2,2],
        strides=2
    )

    # Fully Connected Layer
    pool2_flat = tf.reshape(pool2, [-1, 16 * 16 *64])
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # Logits
    logits = tf.layers.dense(inputs=dropout, units=40)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return  tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    # Training
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes'])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(_):

    mnist = dataset.read_data_sets(modMNIST_train_x, modMNIST_train_y, modMNIST_test_x)
    train_data = mnist.train.images
    train_labels = mnist.train.labels
    validation_data = mnist.validation.images
    validation_labels = mnist.validation.labels

    mnist_classifier = tf.estimator.Estimator(
        model_fn = cnn_model,
        model_dir='/tmp/mnist_convnet_model'
    )

    # Logging
    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50
    )

    # Start Training
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook]
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': validation_data},
        y=validation_labels,
        num_epochs=1,
        shuffle=False
    )

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == '__main__':
    tf.app.run()