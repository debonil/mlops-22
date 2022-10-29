# Standard scientific Python imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from statistics import mean, median
import seaborn as sns

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
    return x,data.target

def train_dev_test_split(data, label, train_frac, dev_frac):

    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True
    )

    return x_train, y_train, x_dev, y_dev, x_test, y_test



def h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev, x_test, y_test, metric):
    best_metric_train = -1.0
    best_h_params_train = None
    best_metric_dev = -1.0
    best_h_params_dev = None
    best_metric_test = -1.0
    best_h_params_test = None
    metric_list_train = []
    metric_list_dev = []
    metric_list_test = []
    # 2. For every combination-of-hyper-parameter values
    print(f"Classifier \t\t\t\t\t\t\t|\t Train Acc \t|\t Dev Acc \t|\t Test Acc \n")
    for cur_h_params in h_param_comb:

        # PART: setting up hyperparameter
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        # PART: Train model
        # 2.a train the model
        # Learn the digits on the train subset
        clf.fit(x_train, y_train)


        # Predict the value of the digit on the test subset
        pred_dev = clf.predict(x_dev)
        acc_dev= metric(y_dev, pred_dev)
        pred_test = clf.predict(x_test)
        acc_test= metric(y_test, pred_test)
        pred_train = clf.predict(x_train)
        acc_train= metric(y_train, pred_train)

        metric_list_train.append(acc_train)
        metric_list_dev.append(acc_dev)
        metric_list_test.append(acc_test)
        print(f"{clf}\t\t\t\t\t\t\t {acc_train*100:.3f}\t\t\t {acc_dev*100:.3f}\t\t\t {acc_test*100:.3f}")
        # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest.
        if acc_dev > best_metric_dev:
            best_metric_dev = acc_dev
            best_h_params_dev = cur_h_params
        if acc_train > best_metric_train:
            best_metric_train = acc_train
            best_h_params_train = cur_h_params
        if acc_test > best_metric_test:
            best_metric_test = acc_test
            best_h_params_test = cur_h_params
    print("\n\t* Min, max, mean, median of the accuracies obtained in previous step : *\t\n")
    print(f"Train Set Accuracy\t\t Min: {min(metric_list_train)*100:.3f}\t\t Max: {max(metric_list_train)*100:.3f}\t\t Mean: {mean(metric_list_train)*100:.3f}\t\t Median: {median(metric_list_train)*100:.3f}")
    print(f"Dev Set Accuracy\t\t Min: {min(metric_list_dev)*100:.3f}\t\t Max: {max(metric_list_dev)*100:.3f}\t\t Mean: {mean(metric_list_dev)*100:.3f}\t\t Median: {median(metric_list_dev)*100:.3f}")
    print(f"Test Set Accuracy\t\t Min: {min(metric_list_test)*100:.3f}\t\t Max: {max(metric_list_test)*100:.3f}\t\t Mean: {mean(metric_list_test)*100:.3f}\t\t Median: {median(metric_list_test)*100:.3f}")
    
    print(f"\nBest Classification Train Accuracy for {best_h_params_train} is {best_metric_train:.2f}")
    print(f"Best Classification Dev Accuracy for {best_h_params_dev} is {best_metric_dev:.2f}")
    print(f"Best Classification Test Accuracy for {best_h_params_test} is {best_metric_test:.2f}")
    clf.set_params(**best_h_params_test)
    return clf, best_metric_test, best_h_params_test


# PART: Sanity check of predictions
def visualize_pred_data(X_test, predicted):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

def generate_h_param_comb(gamma_list,c_list):
    return [{"gamma": g, "C": c} for g in gamma_list for c in c_list]


sns.set(rc={'axes.facecolor': 'lightblue', 'figure.facecolor': 'lightblue'})


def confusionMatrixAndAccuracyReport(Y_test, Y_pred):
    cm = metrics.confusion_matrix(Y_test, Y_pred)
    overallAccuracy = np.trace(cm)/sum(cm.flatten())

    classwiseAccuracy = cm.diagonal()/cm.sum(axis=1)

    plt.figure(figsize=(10, 10))
    plt.title('Accuracy Score: {0:3.3f}'.format(overallAccuracy), size=14)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    sns.heatmap(data=cm, annot=True, square=True,  cmap='Blues', fmt='g')
    
    plt.show()
    print('Overall Accuracy Score: {0:3.3f}'.format(overallAccuracy))
    print('Classwise Accuracy Score: {0}'.format(classwiseAccuracy))

