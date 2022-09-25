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
from sklearn import datasets, metrics, svm
from util import data_preprocess,data_viz,h_param_tuning,train_dev_test_split,visualize_pred_data
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


# PART: Get test set predictions
# Predict the value of the digit on the test subset
predicted = best_model.predict(X_test)

visualize_pred_data(X_test,predicted)

# 4. report the test set accurancy with that best model.
# PART: Compute evaluation metrics
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

print("Best hyperparameters were:")
print(best_h_params)