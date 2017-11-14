
from __future__ import print_function
import csv
import keras
import struct
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GaussianNoise
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator as gen
import os
os.environ["KERAS_BACKEND"] = 'tensorflow'
K.set_image_dim_ordering('tf')


batch_size = 64
num_classes = 82
epochs = 30
zca = True
data_aug = False

def load_files():
    train_x = './train_x.csv'
    train_y = './train_y.csv'
    test_x = './test_x.csv'

    print('Loading Training')
    train_x = np.loadtxt(train_x, delimiter=",")  # load from text
    train_y = np.loadtxt(train_y, delimiter=",")
    print('Loading Kaggle')
    test_x = np.loadtxt(test_x, delimiter=",")

    return train_x, train_y, test_x


def write_csv(output):
    with open('predictions_cnn.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput)
        writer.writerow(['Id'] + ['Label'])
        for i in range(len(output)):
            writer.writerow([str(i + 1)] + [output[i]])


# input image dimensions
img_rows, img_cols = 64, 64

# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

print('Loading Files...\n')
train_x, train_y, kaggle = load_files()

kaggle = kaggle.reshape(-1, 64, 64, 1)

x_train = train_x.reshape(-1, 64, 64, 1)
x_test = x_train[40000:]
x_train = x_train[:40000]

y_train = train_y
y_test = y_train[40000:]
y_train = y_train[:40000]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    kaggle = kaggle.reshape(kaggle.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    kaggle = kaggle.reshape(kaggle.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    print('channels last')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
kaggle = kaggle.astype('float32')
# x_train /= 255
# x_test /= 255
# kaggle /= 255set
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(kaggle.shape[0], 'kaggle samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

train_preproc = gen(
    rescale=1./255,
    vertical_flip=data_aug,
    horizontal_flip=data_aug,
    zca_whitening=zca
)

valid_preproc = gen(
    rescale=1./255,
    zca_whitening=zca
)

kaggle_preproc = gen(
    rescale=1./255,
    zca_whitening=zca
)

if zca:
    train_preproc.fit(x_train)
    valid_preproc.fit(x_test)
    kaggle_preproc.fit(kaggle)

    i = 0
    for x_kaggle, y_kaggle in kaggle_preproc.flow(x_test, np.zeros([len(x_test)]), batch_size=1):
        if i < len(x_test):
            kaggle[i] = x_kaggle
        else:
            break
        i += 1


train_generator = train_preproc.flow(x_train, y_train, batch_size=batch_size)
validation_generator = valid_preproc.flow(x_test, y_test, batch_size=batch_size)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(GaussianNoise(0.1))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(Conv2D(512, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.25))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit_generator(train_generator,
      steps_per_epoch=800,
      epochs=epochs,
      verbose=1,
      validation_data=validation_generator,
      validation_steps=625,
)

x_train /= 255
x_test /= 255
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = model.predict(kaggle, verbose=1)
# print(predictions[:100])
output = predictions.argmax(axis=1)
# print(output[:100])

print('\nWriting CSV File...\n')
write_csv(output)
