
import statistics
from math import floor
from unittest import result

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets, metrics, svm, tree

from util import (confusionMatrixAndAccuracyReport, data_preprocess,
                  h_param_tuning, train_dev_test_split)

sns.set(rc={'axes.facecolor': 'lightblue', 'figure.facecolor': 'lightblue'})
# Starts actual execution

digits = datasets.load_digits()

# data_viz(digits)

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
metrices_to_measure = [metrics.accuracy_score, metrics.f1_score,
                       metrics.recall_score, metrics.precision_score]
result = [[[], []] for _ in range(len(metrices_to_measure))]
cm_list = [[], []]
for i, clf in enumerate(model_of_choices):
    for k in range(5):

        print(f'\nChecking Classifier {clf} for split no : {k+1}')

        X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
            data, label, 0.8, 0.1
        )

        best_model, best_metric, best_h_params = h_param_tuning(
            hp_of_choices[i], clf, X_train, y_train, X_dev, y_dev, X_test, y_test, metrics.accuracy_score)

        predicted = best_model.predict(X_test)
        for mi, metric in enumerate(metrices_to_measure):
            result[mi][i].append(
                metric(y_test, predicted, average='weighted') if mi > 0 else metric(
                    y_test, predicted)
            )

        cm_list[i].append(metrics.confusion_matrix(y_test, predicted))
        #visualize_pred_data(X_test, predicted)

        # 4. report the test set accurancy with that best model.
        # PART: Compute evaluation metrics
        print(
            f"Classification report for classifier {clf}:\n"
            f"{metrics.classification_report(y_test, predicted)}\n"
        )
        confusionMatrixAndAccuracyReport(y_test, predicted)

result_df: list[pd.DataFrame] = []
for mi, metric in enumerate(metrices_to_measure):
    result[mi][0].append(statistics.mean(result[mi][0]))
    result[mi][0].append(statistics.stdev(result[mi][0]))
    result[mi][1].append(statistics.mean(result[mi][1]))
    result[mi][1].append(statistics.stdev(result[mi][1]))
    result_df.append(
        pd.DataFrame(np.transpose(result[mi]), index=[1, 2, 3, 4, 5, 'Mean', 'STD'], columns=['SVM', 'DecisionTree']))

print('\nComparison of two Classifier')
print('____________________________\n')
for mi, metric in enumerate(metrices_to_measure):
    print(f'\nMetric : {metric.__name__} \n')
    print(result_df[mi])

with open('README.md', 'w') as f:
    print('\n# ML-Ops 2022 :: Assignment 4', file=f)
    print('\n by **Debonil Ghosh [M21AIE225]**', file=f)

    print('\n### Comparison of two Classifier ###\n', file=f)
    for mi, metric in enumerate(metrices_to_measure):
        print('____________________________\n', file=f)
        print(f' - **metric : {metric.__name__}** \n', file=f)
        print(result_df[mi].to_markdown(), file=f)

fig, axes = plt.subplots(2, 2)
fig.set_size_inches(12, 6)
fig.suptitle('Comparison of two Classifier on different metric')

for mi, metric in enumerate(metrices_to_measure):
    data1 = result_df[mi]['SVM'].head(5)
    data2 = result_df[mi]['DecisionTree'].head(5)
    width = 0.3
    axes_i = axes[floor(mi/2)][mi % 2]
    axes_i.title.set_text(metric.__name__)
    axes_i.bar(np.arange(len(data1)), data1,
               width=width, label='SVM')
    axes_i.bar(np.arange(len(data2)) + width, data2,
               width=width, label='DecisionTree')
    #axes_i.set_xlabel('Trials with different train test split')
    axes_i.set_xticks([r + width for r in range(len(data1))],
                      np.arange(1, len(data1)+1))
    axes_i.legend()
plt.savefig("comparison_on_different_metric.png")
# plt.show()

# ploting confusion matrix


for i, clf in enumerate(model_of_choices):
    for k in range(5):
        cm = cm_list[i][k]
        overallAccuracy = np.trace(cm)/sum(cm.flatten())
        fig, ax = plt.subplots()
        #fig.set_size_inches(10, 10)
        fig.suptitle(f'Confusion matrices of {clf} of run # {k+1}')
        plt.title('Accuracy Score: {0:3.3f}'.format(overallAccuracy), size=14)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        sns.heatmap(data=cm, annot=True, square=True,
                    cmap='Blues', fmt='g', ax=ax)
        plt.savefig(f"confusion_matrix_{clf}_{k+1}.png")
        # plt.show()
