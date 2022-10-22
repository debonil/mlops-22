
from unittest import result
from sklearn import datasets, metrics, svm
from sklearn import tree
import statistics
from util import data_preprocess,data_viz,h_param_tuning,train_dev_test_split,visualize_pred_data
## Starts actual execution

digits = datasets.load_digits()

data_viz(digits)

data, label = data_preprocess(digits)
# housekeeping
del digits

gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
h_param_comb_svm = [{"gamma": g, "C": c} for g in gamma_list for c in c_list]

min_samples_split_list = [2,3,5,10]
min_samples_leaf_list = [1,3,5,10]
h_param_comb_dtree = [{"min_samples_leaf": g, "min_samples_split": c} for g in min_samples_leaf_list for c in min_samples_split_list]

model_of_choices = [svm.SVC(),tree.DecisionTreeClassifier()]
hp_of_choices = [h_param_comb_svm,h_param_comb_dtree]
metric=metrics.accuracy_score
result=[[],[]]
for i,clf in enumerate(model_of_choices):
    for k in range(5):

        X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
            data, label, 0.8, 0.1
        )

        best_model, best_metric, best_h_params = h_param_tuning(hp_of_choices[i], clf, X_train, y_train, X_dev, y_dev, X_test, y_test, metric)

        predicted = best_model.predict(X_test)
        result[i].append(best_metric)
        visualize_pred_data(X_test,predicted)

        # 4. report the test set accurancy with that best model.
        # PART: Compute evaluation metrics
        print(
            f"Classification report for classifier {clf}:\n"
            f"{metrics.classification_report(y_test, predicted)}\n"
        )

print(f'SVM = {result[0]}')
print(f'Decision Tree = {result[1]}')
print(f'SVM = \t\tmean:\t{statistics.mean(result[0]):.2f},\t sd:\t{statistics.stdev(result[0]):.2f},')
print(f'Decision Tree = \tmean:\t{statistics.mean(result[1]):.2f},\t sd:\t{statistics.stdev(result[1]):.2f},')
