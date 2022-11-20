
from unittest import result
from sklearn import datasets, metrics, svm
from sklearn import tree
import statistics
from utils import confusionMatrixAndAccuracyReport, data_preprocess, data_viz, h_param_tuning, train_dev_test_split, visualize_pred_data
import pandas as pd
import numpy as np
from joblib import dump
import argparse
# Starts actual execution
parser = argparse.ArgumentParser(
    prog='plot_graphs',
    description='MLOpsFinalExam-M21AIE225',
    epilog='Text at the bottom of help')
# option that takes a value
parser.add_argument('-c', '--clf_name',
                    choices=['svm', 'tree'], action='store')
parser.add_argument('-v', '--random_state', action='store')  # on/off flag

args = parser.parse_args()
print(args.clf_name, args.random_state)

digits = datasets.load_digits()

#data_viz(digits)

data, label = data_preprocess(digits)
# housekeeping
del digits

gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
h_param_comb_svm = [{"gamma": g, "C": c} for g in gamma_list for c in c_list]

min_samples_split_list = [2, 3, 5, 10]
min_samples_leaf_list = [1, 3, 5, 10]
h_param_comb_dtree = [{"min_samples_leaf": g, "min_samples_split": c}
                      for g in min_samples_leaf_list for c in min_samples_split_list]

model_of_choices = [svm.SVC(), tree.DecisionTreeClassifier()]
hp_of_choices = [h_param_comb_svm, h_param_comb_dtree]
metric = metrics.accuracy_score
result = [[], []]

i = 0 if args.clf_name == 'svm' else 1
clf = model_of_choices[i]
# for i, clf in enumerate(model_of_choices):
# for k in range(5):

X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
    data, label, 0.8, 0.1, random_state=int(args.random_state)
)

best_model, best_metric, best_h_params = h_param_tuning(
    hp_of_choices[i], clf, X_train, y_train, X_dev, y_dev, X_test, y_test, metric)

predicted = best_model.predict(X_test)
result[i].append(best_metric)
#visualize_pred_data(X_test, predicted)

# 4. report the test set accurancy with that best model.
# PART: Compute evaluation metrics
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
confusionMatrixAndAccuracyReport(y_test, predicted)
dump(best_model, f'models/{best_model}')


result[i].append(statistics.mean(result[i]))
result[i].append(statistics.stdev(result[i]))
