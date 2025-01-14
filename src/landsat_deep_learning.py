import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import os
import argparse

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

# Argument processing
parser = argparse.ArgumentParser(description='Machine learning algorithm for landsat data. Options are available for re-training the model, testing the model with external data or running performance analysis. Note that only one option should be supplied at a time e.g. you should not run the script with both --test and --re-train.')

parser.add_argument('--test', metavar='TEST_FILE', nargs=1, dest='test_file', help='path to additional features to test against. Path must be relative to the current directory. If supplied, results of predictions against this test data will be the last thing printed by this script (optional)')

parser.add_argument('--re-train', action='store_const', const=True, default=False, help='will re-train the model and document cross validation process')

parser.add_argument('--analysis', action='store_const', const=True, default=False, help='will run the model against test data and report performance')

parser.add_argument('--test-dev', action='store_const', const=True, default=False, help='Will do a train-test and print the results to the command line using the dataset from Canvas. For development purpose only')

parsed_args = parser.parse_args()

DATASET = '../datasets/landsat.csv'

MODEL_DIR = '../model'
MODEL_NAME = 'landsat_deep_model.h5'

RESULTS_DIR = '../results'
RESULTS_NAME = 'landsat_deep_results.csv'

PERFORMANCE_DIR = '../performance'
PERFORMANCE_NAME = 'landsat_deep.csv'

CV_RESULTS_DIR = '../cv_results'
CV_RESULTS_NAME = 'landsat_deep_cv.csv'

target_col = 36

label_dict = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        7: 5
}

def as_keras_metric(method):
    import functools
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        keras.backend.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

def f1_score(precision, recall):
    return  (2 * precision * recall) / (precision + recall)

def preprocess(data):
    target_col = 36
    features = data
    y = None
    if target_col in data.columns:
        last_feature = data.columns.get_loc(target_col)
        features = data.iloc[:, :last_feature].copy()
        y = data.loc[:, target_col].copy()
        y = y.apply(lambda x: label_dict[x])

    full_pipeline = Pipeline([
            ('imputer', Imputer(strategy="median")),
            ('std_scaler', StandardScaler())
        ])

    X = pd.DataFrame(data=full_pipeline.fit_transform(features))
    
    return X, y
    
def train(save=True, split_data=None):
    print("Training model")
    if split_data is None:
        data = pd.read_csv(DATASET, header=None)
        X, y = preprocess(data)
        sss = StratifiedShuffleSplit(n_splits=3, test_size=0.1, random_state=42)
        for train_index, test_index in sss.split(X, y):
                    train_ind,test_ind = train_index, test_index
    
        X_train, X_test = X.loc[train_ind], X.loc[test_ind]
        y_train, y_test = y.loc[train_ind], y.loc[test_ind]

    else:
        X_train, X_test, y_train, y_test = split_data
    
    model = keras.Sequential([
            keras.layers.Dense(21, activation=tf.nn.relu),
            keras.layers.Dense(6, activation=tf.nn.softmax)
    ])
    
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
        
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', precision, recall])
    
    model.fit(X_train.values, y_train.values, epochs=5)
    
    test_loss, test_acc, test_pr, test_rec = model.evaluate(X_test, y_test)
    if save:
        print("Saving model in {}/{}".format(MODEL_DIR, MODEL_NAME))
        model.save(os.path.join(MODEL_DIR, MODEL_NAME))
    
    return model, X_test, y_test
    
def test_extern(test_file):
    print("Loading model from {}/{}".format(MODEL_DIR, MODEL_NAME))
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    model = keras.models.load_model(os.path.join(MODEL_DIR, MODEL_NAME), custom_objects={'precision': precision, 'recall': recall})
    print("Loading test data from {}".format(test_file))
    test_data = pd.read_csv(test_file, header=None)
    
    X, _ = preprocess(test_data)

    print("Making predictions")
    prediction = pd.DataFrame(data=model.predict(X.values))
    
    with open(os.path.join(RESULTS_DIR,RESULTS_NAME), 'w') as outfile:
        print("Printing prediction results in {}".format(outfile.name))
        prediction.to_csv(path_or_buf=outfile, index=False)
        
    print("Finished testing")
        
def test_dev():
    model, X_test, y_test = train()
    loss, test_acc, test_pr, test_rec = model.evaluate(X_test, y_test)
    print('*******************************************************************')

    print('*******************************************************************')
    print("The prediction accuracy (tuned) for all testing instances is : {:.2f}%.".format(test_acc * 100))
    print("The f1 score (tuned) for all testing instances is : {:.2f}%.".format(f1_score(test_pr, test_rec)))
        
def performance():
    num_trials = 3
    acc = []
    f1 = []
    
    data = pd.read_csv(DATASET, header=None)
    X, y = preprocess(data)
    skf = StratifiedKFold(n_splits=num_trials, random_state=1)
    skf.get_n_splits(X)
    
    i = 1
    print("Running {} train and evaluate iterations".format(num_trials))
    for train_index, test_index in skf.split(X, y):
        print("Iteration {} out of {}".format(i, num_trials))
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        split_data = [X_train, X_test, y_train, y_test]
        model, X_test, y_test = train(False, split_data)
        loss, test_acc, test_pr, test_rec = model.evaluate(X_test, y_test)
        
        acc.append(test_acc * 100)
        f1.append(f1_score(test_pr, test_rec))
        
        i += 1
        
    # Take mean of all the trials
    acc_av = pd.DataFrame(acc).mean()
    f1_av =  pd.DataFrame(f1).mean()

    print('*******************************************************************')
    print("The average prediction accuracy (tuned) for all testing instances is : {:.2f}%.".format(acc_av.iloc[0]))
    print("The average f1 score (tuned) for all testing instances is : {:.2f}.".format(f1_av.iloc[0]))
    
    ## Print all this stuff to a file
    headers = ['Model', 'Model Tuning', 'Accuracy (%)', 'F1 Score']
    model_name = 'deep_learning'
    tuned_results = pd.DataFrame([[model_name, 'Tuned', acc_av.iloc[0], f1_av.iloc[0]]], columns=headers)
    
    results = pd.DataFrame(columns=headers)
    results = results.append(tuned_results)
    results.to_csv(path_or_buf=os.path.join(PERFORMANCE_DIR, PERFORMANCE_NAME), index=False)

if __name__ == "__main__":
    parsed_args = parser.parse_args()
    if parsed_args.re_train:
        print("Re-training model")
        train()
    elif parsed_args.analysis:
        print("Doing performance analysis")
        performance()
    elif parsed_args.test_file != None:
        print("Testing with external data from {}".format(parsed_args.test_file))
        test_extern(parsed_args.test_file[0])
    elif parsed_args.test_dev:
        print("Running development test")
        test_dev()
    else:
        parser.print_help()

test_extern('../datasets/landsat.csv')


