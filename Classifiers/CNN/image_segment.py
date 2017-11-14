from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

test_images = np.loadtxt('./test_x,csv', delimiter=',')
train_image = np.loadtxt('./train_x.csv', delimiter=',')


clf = KMeans(n_clusters=4, random_state=1337, n_jobs=-1).fit(train_image)
results = clf.predict(test_images)

axis = 1
u, indices = np.unique(results, return_inverse=True)
background_index = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(results.shape), None, np.max(indices) + 1), axis=axis)]


