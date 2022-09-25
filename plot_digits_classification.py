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
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


def data_viz(data_to_viz):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, data_to_viz.images, data_to_viz.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

def data_preprocess(data):
    # flatten the images
    n_samples = len(data.images)
    x = data.images.reshape((n_samples, -1))
    return x,digits.target

def train_dev_test_split(data, label, train_frac, dev_frac):

    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True
    )

    return x_train, y_train, x_dev, y_dev, x_test, y_test



def h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric):
    best_metric = -1.0
    best_model = None
    best_h_params = None
    # 2. For every combination-of-hyper-parameter values
    for cur_h_params in h_param_comb:

        # PART: setting up hyperparameter
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        # PART: Train model
        # 2.a train the model
        # Learn the digits on the train subset
        clf.fit(x_train, y_train)

        # print(cur_h_params)
        # PART: get dev set predictions
        predicted_dev = clf.predict(x_dev)

        # 2.b compute the accuracy on the validation set
        cur_metric = metric(y_pred=predicted_dev, y_true=y_dev)

        # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest.
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_model = clf
            best_h_params = cur_h_params
            print("Found new best metric with :" + str(cur_h_params))
            print("New best val metric:" + str(cur_metric))
    return best_model, best_metric, best_h_params



## Starts actual execution

digits = datasets.load_digits()

data_viz(digits)

data, label = data_preprocess(digits)
# housekeeping
del digits




# Image size and resize
""" print(f"Input Image size: \t{digits.images[0].shape}\n")
image_rescaled = np.asarray([rescale(img, 8, anti_aliasing=False) for img in digits.images])
print(f"Image resized to: \t{image_rescaled[0].shape}\n")
 """


X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
    data, label, 0.8, 0.1
)

# Create a classifier: a support vector classifier
#parameters = {'Gamma': [0.00001,0.0001,0.001,0.01], 'C': [ 1, 2, 5,10,50,100]}
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

h_param_comb = [{"gamma": g, "C": c} for g in gamma_list for c in c_list]

clf = svm.SVC()
metric=metrics.accuracy_score
best_model, best_metric, best_h_params = h_param_tuning(h_param_comb, clf, X_train, y_train, X_dev, y_dev, metric)

print(f"\nBest Classification {metric} for classifier {best_model} is {best_metric:.2f}\n")