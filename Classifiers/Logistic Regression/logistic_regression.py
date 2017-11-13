# Modified MNIST Classifier - Logistic Regression
# Thomas Page

# import cv2
import csv
import struct
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage import filters
from skimage import exposure
from sklearn.linear_model import LogisticRegression


def main():
    print('Loading Files...\n')
    train_x, train_y, test_x, emnist_images, emnist_labels = load_files()

    print('Splitting Characters...\n')
    newimages = character_splitter(train_x)

    print('Learning Data...\n')
    output = predictor_LR(emnist_images, emnist_labels, newimages)

    training_accuracy(output, train_y)

    print('Writing CSV File...\n')
    write_csv(output)


def load_files():
    train_x = '../../Data/Modified MNIST/train_x_small.csv'
    train_y = '../../Data/Modified MNIST/train_y_small.csv'
    test_x = '../../Data/Modified MNIST/test_x.csv'

    train_x = np.loadtxt(train_x, delimiter=",")  # load from text
    train_y = np.loadtxt(train_y, delimiter=",")
    # test_x = np.loadtxt(test_x, delimiter=",")

    emnist_images, emnist_labels = load_mnist()
    emnist_images = emnist_images.reshape(-1, 28, 28)
    emnist_images = np.transpose(emnist_images, (0, 2, 1))
    emnist_images = emnist_images.reshape(-1, 28 * 28)

    return train_x, train_y, test_x, emnist_images, emnist_labels


def load_mnist():
    images_path = '../../Data/EMNIST/emnist-byclass-train-images-idx3-ubyte'
    labels_path = '../../Data/EMNIST/emnist-byclass-train-labels-idx1-ubyte'

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    digits = (labels == 0) | (labels == 1) | (labels == 2) | (labels == 3) | (labels == 4) | (labels == 5) | (labels == 6) | (labels == 7) | (labels == 8) | (labels == 9)
    letters = (labels == 10) | (labels == 22)

    images = images[digits | letters]
    labels = labels[digits | letters]

    return images, labels


def display_image(x):
    dim = int(np.sqrt(x.shape[0]))
    x = x.reshape(-1, dim, dim)
    plt.figure()
    plt.imshow(x[0], 'gray')
    plt.show()
    return


def canny_edges(x):

    dim = int(np.sqrt(x.shape[0]))

    temp = exposure.adjust_gamma(x, (np.mean(x) ** 2) / 90 / 40)
    temp = temp.reshape(-1, dim, dim)
    temp = filters.gaussian(temp, 0.5)
    temp = (254 * np.divide(temp, np.max(temp))).astype(int)

    x = exposure.adjust_gamma(x, 10)
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

    return newimages


def predictor_LR(emnist_images, emnist_labels, newimages):
    output = []
    logisticRegr = LogisticRegression(verbose=True, solver='sag', multi_class='multinomial', tol=0.05)
    logisticRegr.fit(emnist_images, emnist_labels)
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

    for i in range(len(newimages)):
        prediction = logisticRegr.predict(newimages[i])

        digits = prediction[prediction < 10]
        letters = prediction[prediction >= 10]

        digit_count = len(digits)
        letter_count = len(letters)

        if digit_count == 2 and letter_count == 1:
            # print('made it')
            if 10 in letters:
                # print('test1')
                output.append(np.sum(digits))
            elif 22 in letters:
                # print('test2')
                output.append(np.prod(digits))
            # print(output[i])
        else:
            output.append(random.choice(classes))
            # print(output[i])

    return output


def write_csv(output):
    with open('predictions.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput)
        writer.writerow(['Id'] + ['Label'])
        for i in range(len(output)):
            writer.writerow([str(i + 1)] + [output[i]])


def training_accuracy(output, train_y):
    truth = [el for el in train_y]
    correct = [i for i, j in zip(output, truth) if i == j]
    percent_correct = float(len(correct)) / float(len(output))
    print('\nPercent Correct: ' + str(percent_correct))


def character_splitter(test_x):
    newimages = []
    for i in range(len(test_x)):
        newimages.append(canny_edges(test_x[i]))
        newimages[i] = np.array(newimages[i])
        newimages[i] = newimages[i].reshape(-1, 28 * 28)

    return newimages


def plot_contours(x, contours, newimages):
    plt.figure(figsize=(3.75, 6))
    ax = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)
    ax.imshow(x[0], 'gray', interpolation='nearest')

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax0 = plt.subplot2grid((3, 3), (2, 0))
    ax0.imshow(newimages[0], 'gray')

    ax1 = plt.subplot2grid((3, 3), (2, 1))
    ax1.imshow(newimages[1], 'gray')

    ax2 = plt.subplot2grid((3, 3), (2, 2))
    ax2.imshow(newimages[2], 'gray')

    ax.set_xticks([])
    ax.set_yticks([])
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.show()

    print('test')

    return


if __name__ == '__main__':
    main()
