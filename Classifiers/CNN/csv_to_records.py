from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import dataset

import tensorflow as tf
modMNIST_train_x = "train_x.csv"
modMNIST_train_y = "train_y.csv"
modMNIST_test_x = "test_x.csv"

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

<<<<<<< HEAD
def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
=======
>>>>>>> 97a8f73ac3b0f51380eb4e69584660dd710a664d

def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = data_set.images
  labels = data_set.labels
  num_examples = data_set.num_examples

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
<<<<<<< HEAD

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)

  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):

    image_raw = images[index]
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(int(labels[index])),
        'image_raw': _float_feature(image_raw)}))
=======
  rows = images.shape[1]
  cols = images.shape[2]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
>>>>>>> 97a8f73ac3b0f51380eb4e69584660dd710a664d
    writer.write(example.SerializeToString())
  writer.close()


def main(unused_argv):
  # Get the data.
  data_sets = dataset.read_data_sets(modMNIST_train_x,
                                     modMNIST_train_y,
                                     modMNIST_test_x,
                                   dtype=tf.uint8,
                                   reshape=False,
                                   validation_size=FLAGS.validation_size)

  # Convert to Examples and write the result to TFRecords.
  convert_to(data_sets.train, 'train')
  convert_to(data_sets.validation, 'validation')
  convert_to(data_sets.test, 'test')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
<<<<<<< HEAD
      default=10000,
=======
      default=5000,
>>>>>>> 97a8f73ac3b0f51380eb4e69584660dd710a664d
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)