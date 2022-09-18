"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import rescale
import numpy as np
###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.


# Image size and resize
print(f"Input Image size: \t{digits.images[0].shape}\n")
image_rescaled = np.asarray([rescale(img, 0.5, anti_aliasing=False) for img in digits.images])
print(f"Image resized to: \t{image_rescaled[0].shape}\n")

# flatten the images
n_samples = len(image_rescaled)
data = image_rescaled.reshape((n_samples, -1))

# Split data into 50% train and 50% test subsets
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False
)
X_dev, X_test, y_dev, y_test = train_test_split(
    X_dev_test, y_dev_test, test_size=0.5, shuffle=False
)


# Create a classifier: a support vector classifier
parameters = {'Gamma': [0.00001,0.0001,0.001,0.01], 
              'C': [ 1, 2, 5,10,50,100]}
best_acc=[-1,-1,-1]
best_model = None

print(f"Classifier \t\t\t\t\t\t\t|\t Train Acc \t|\t Dev Acc \t|\t Test Acc \n")
for GAMMA in parameters['Gamma']:
    for C in parameters['C']:

        hyper_params = {'gamma':GAMMA, 'C':C}
        clf = svm.SVC()
        clf.set_params(**hyper_params)


        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        pred_dev = clf.predict(X_dev)
        acc_dev= metrics.accuracy_score(y_dev, pred_dev)
        pred_test = clf.predict(X_test)
        acc_test= metrics.accuracy_score(y_test, pred_test)
        pred_train = clf.predict(X_train)
        acc_train= metrics.accuracy_score(y_train, pred_train)

        print(f"{clf}\t\t\t\t\t\t\t {acc_train*100:.3f}\t\t\t {acc_dev*100:.3f}\t\t\t {acc_test*100:.3f}")
        if acc_dev > best_acc[1]:
            best_acc = [acc_train,acc_dev,acc_test]
            best_model = hyper_params

print(f"\nBest Classification accuracy for classifier {best_model}: \tTrain Acc:{best_acc[0]*100:.2f};\tDev Acc:{best_acc[1]*100:.2f};\tTest Acc:{best_acc[2]*100:.2f};\n")
