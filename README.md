# Modified MNIST

The goal of this project is to classify images based on the mathematical equations present within them. The algorithms used to perform this image analysis task were logistic regression (LR), a feed forward neural network (NN), and a convolutional neural network (CNN).

The classifiers can be found in the Classifiers folder.
Supplementary training data such as the EMNIST dataset can be found in the Data folder.

Running any of the CNNs requires running the csv_to_records script first to generate .tfrecords binary files. This only needs to be done once. None of the trained weights are stored, so the nets will train before generating outputs. The training and testing data need to be placed in the same directory as the python code. Running it requires no flags but requires Tensorflow and/or Theano.