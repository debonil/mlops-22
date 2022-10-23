import sys
from sklearn import datasets, metrics, svm
import numpy
sys.path.append('.')

from util import generate_h_param_comb,data_preprocess,data_viz,train_dev_test_split,visualize_pred_data

# test case to check if all the combinations of the hyper parameters are indeed getting created
def removed_test_generate_h_param_comb():
    gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
    h_param_comb = generate_h_param_comb(gamma_list,c_list)
    assert len(h_param_comb)==len(gamma_list)*len(c_list)

# test case to check if all the combinations of the hyper parameters are indeed getting created
def test_quize_quest3():
    data, label = data_preprocess(datasets.load_digits())
    metric=metrics.accuracy_score
    X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
        data, label, 0.8, 0.1
    )
    best_metric = {'gamma': 0.001, 'C': 2}
    best_model = svm.SVC()
    best_model.set_params(**best_metric)
    best_model.fit(X_train, y_train)
    predicted = best_model.predict(X_test)
    unique_pred_classes = numpy.unique(predicted)
    assert len(unique_pred_classes)>1

def test_quize_quest4():
    data, label = data_preprocess(datasets.load_digits())
    X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
        data, label, 0.8, 0.1
    )
    best_metric = {'gamma': 0.001, 'C': 2}
    best_model = svm.SVC()
    best_model.set_params(**best_metric)
    best_model.fit(X_train, y_train)
    predicted = best_model.predict(X_test)
    unique_pred_classes = numpy.unique(predicted)
    #print(unique_pred_classes)
    #print(numpy.unique(label))
    assert len(unique_pred_classes)>=len(numpy.unique(label))


#what more test cases should be there 
#irrespective of the changes to the refactored code.

# train/dev/test split functionality : input 200 samples, fraction is 70:15:15, then op should have 140:30:30 samples in each set


# preprocessing gives ouput that is consumable by model

# accuracy check. if acc(model) < threshold, then must not be pushed.

# hardware requirement test cases are difficult to write.
# what is possible: (model size in execution) < max_memory_you_support

# latency: tik; model(input); tok == time passed < threshold
# this is dependent on the execution environment (as close the actual prod/runtime environment)


# model variance? -- 
# bias vs variance in ML ? 
# std([model(train_1), model(train_2), ..., model(train_k)]) < threshold


# Data set we can verify, if it as desired
# dimensionality of the data --

# Verify output size, say if you want output in certain way
# assert len(prediction_y) == len(test_y)

# model persistance?
# train the model -- check perf -- write the model to disk
# is the model loaded from the disk same as what we had written?
# assert acc(loaded_model) == expected_acc 
# assert predictions (loaded_model) == expected_prediction 


