#######
#1st run of K-means, Digits, Post PCA
#######
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_digits
from sklearn import metrics

# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn import preprocessing
import numpy as np
from sklearn.svm import SVC

#Load the dataset
iris = datasets.load_digits()

#######
#1st run of PCA, Digits
#######
from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca.fit(iris.data)
X = pca.transform(iris.data)
import pylab as pl
pl.scatter(X[:, 0], X[:, 1], c=iris.target)
pl.show()

#dataset = load_digits()
#print("Number of Samples: %d" %len(dataset.target))
#print("Output Categories: %s" %dataset.target_names)
#features = dataset.data
features = X
print("Feature Vectors: %s" %features)
labels = dataset.target
print("Labels: %s" %labels)

trainIdx = np.random.rand(len(labels)) < 0.8
features_train = features[trainIdx]
labels_train = labels[trainIdx]
features_test = features[~trainIdx]
labels_test = labels[~trainIdx]
print("Number of training samples: ",features_train.shape[0])
print("Number of test samples: ",features_test.shape[0])
print("Feature vector dimensionality: ",features_train.shape[1])

# import modules
from sklearn.neighbors import KNeighborsClassifier
# initiate the classifier
knn = KNeighborsClassifier(n_neighbors=3)
# fit the classifier model with training data
knn.fit(features_train, labels_train)
# predict the output labels of test data
labels_pred = knn.predict(features_test)
# print classification metrics 
print(metrics.classification_report(labels_test, labels_pred))
# print confusion matrix
print(metrics.confusion_matrix(labels_test, labels_pred))
