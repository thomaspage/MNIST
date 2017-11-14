"""
This was mostly copied from a GitHub Project:
https://gist.github.com/ambodi/408301bc5bc07bc5afa8748513ab9477
"""

import numpy as np
import collections
from tensorflow.python.framework import dtypes
import tensorflow as tf

class Dataset(object):
    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=False):
        if reshape:
            # not really sure that this is doing
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])

        self._images = images
        self._num_examples = images.shape[0]
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1

            # Shuffle data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end]

def read_data_sets(train_xfile, train_yfile, test_xfile, fake_data=False, one_hot=True, dtype=dtypes.float32, reshape=True, validation_size=10000):

    train_y = np.loadtxt(train_yfile, delimiter=',')
    train_y = train_y.reshape(-1, 1)

    train_x = np.loadtxt(train_xfile, delimiter=',')
    train_x = train_x.reshape(-1, 4096)

    test_x = np.loadtxt(test_xfile, delimiter=',')

    num_training = train_x.shape[0] - validation_size
    num_validation = validation_size
    num_test = test_x.shape[0]

    mask = range(num_training)
    train_images = train_x[mask]
    train_labels = train_y[mask]

    mask = range(num_training, num_training + num_validation)
    validation_images = train_x[mask]
    validation_labels = train_y[mask]

    train = Dataset(train_images, train_labels, dtype=dtype, reshape=False)
    validation = Dataset(validation_images, validation_labels, dtype=dtype, reshape=False)

    test = Dataset(test_x, np.zeros(test_x.shape[0]), dtype=dtype, reshape=False)
    ds = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

    return ds(train=train, validation=validation, test=test)



def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot