# Modified MNIST Classifier - Logistic Regression
# Thomas Page

import numpy as np
import scipy.misc  # to visualize only

train_x = '../../Data/Modified MNIST/train_x.csv'
train_y = '../../Data/Modified MNIST/train_y.csv'

x = np.loadtxt(train_x, delimiter=",")  # load from text
y = np.loadtxt(train_y, delimiter=",")
x = x.reshape(-1, 64, 64)  # reshape
y = y.reshape(-1, 1)
scipy.misc.imshow(x[0])  # to visualize only
