import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.externals import joblib

# Argument processing
parser = argparse.ArgumentParser(description='Machine learning algorithm for landsat data. Options are available for re-training the model, testing the model with external data or running performance analysis. Note that only one option should be supplied at a time e.g. you should not run the script with both --test and --re-train.')

parser.add_argument('--test', metavar='TEST_FILE', nargs=1, dest='test_file', help='path to additional features to test against. Path must be relative to the current directory. If supplied, results of predictions against this test data will be the last thing printed by this script (optional)')

parser.add_argument('--re-train', action='store_const', const=True, default=False, help='will re-train the model and document cross validation process')

parser.add_argument('--analysis', action='store_const', const=True, default=False, help='will run the model against test data and report performance')

parser.add_argument('--test-dev', action='store_const', const=True, default=False, help='Will do a train-test and print the results to the command line using the dataset from Canvas. For development purpose only')

parsed_args = parser.parse_args()

DATASET = '../datasets/landsat.csv'

MODEL_DIR = '../model'
MODEL_NAME = 'landsat_logistic_model'

RESULTS_DIR = '../results'
RESULTS_NAME = 'landsat_logistic_results.csv'

PERFORMANCE_DIR = '../performance'
PERFORMANCE_NAME = 'landsat_logistic_regression.csv'

CV_RESULTS_DIR = '../cv_results'
CV_RESULTS_NAME = 'landsat_logistic_cv.csv'

target_col = 36

for directory in [MODEL_DIR, RESULTS_DIR, PERFORMANCE_DIR, CV_RESULTS_DIR]:
    if not os.path.isdir(directory):
        print("Making directory: {}".format(directory))
        os.mkdir(directory)

def preprocess(data):
    target_col = 36
    features = data
    y = None
    if target_col in data.columns:
        last_feature = data.columns.get_loc(target_col)
        features = data.iloc[:, :last_feature].copy()
        y = data.loc[:, target_col].copy()

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
        
    cv_pipeline = Pipeline([
            ('feature_select', SelectKBest()),
            ('classify', LogisticRegression())
        ])

    DIMENSIONS_TEST = [4, 12, 36]

    param_grid = [
                {
                    'feature_select__k': DIMENSIONS_TEST,
                    'classify__penalty': ['l1', 'l2'],
                    'classify__C': [ 2**x for x in range(-6,6) ]
                }
            ]

    grid_search = GridSearchCV(cv_pipeline, param_grid, cv=3, scoring='accuracy',verbose=1)
    grid_search.fit(X_train, y_train)

    cv_results_ = grid_search.cv_results_
    cvs = pd.DataFrame(cv_results_)
    best_params_ = grid_search.best_params_
    print(best_params_)

    with open(os.path.join(CV_RESULTS_DIR, CV_RESULTS_NAME), 'w') as outfile:
        cvs.to_csv(path_or_buf=outfile, index=False)

    tuned_model = grid_search.best_estimator_

    untuned_model = LogisticRegression()
    untuned_model.fit(X_train, y_train)

    if save:
        print("Saving model in {}/{}".format(MODEL_DIR, MODEL_NAME))
        joblib.dump(tuned_model, os.path.join(MODEL_DIR, MODEL_NAME))

    return tuned_model, untuned_model, X_test, y_test

def test_extern(test_file):
    print("Loading model from {}/{}".format(MODEL_DIR, MODEL_NAME))
    model = joblib.load(os.path.join(MODEL_DIR, MODEL_NAME))
    print("Loading test data from {}".format(test_file))
    test_data = pd.read_csv(test_file, header=None)

    X, _ = preprocess(test_data)

    print("Making predictions")
    prediction = pd.DataFrame(data=model.predict(X), columns=['Target'])

    with open(os.path.join(RESULTS_DIR,RESULTS_NAME), 'w') as outfile:
        print("Printing prediction results in {}".format(outfile.name))
        prediction.to_csv(path_or_buf=outfile, index=False)

def test_dev():
    tuned_model, untuned_model, X_test, y_test = train()

    print("Making test predictions")
    T_untuned = untuned_model.predict(X_test)
    acc_untuned = (100*accuracy_score(y_test, T_untuned))
    f1_untuned = (f1_score(y_test, T_untuned, average='weighted', labels=pd.unique(pd.unique(y_test))))

    #tuned score
    T_tuned = tuned_model.predict(X_test)
    acc_tuned = (100*accuracy_score(y_test, T_tuned))
    f1_tuned = (f1_score(y_test, T_tuned,average='weighted', labels=pd.unique(pd.unique(y_test))))

    print(acc_untuned)
    print('*******************************************************************')
    print("The prediction accuracy (untuned) for all testing instances is : {:.2f}%.".format(acc_untuned))
    print("The f1 score (untuned) for all testing instances is : {:.2f}.".format(f1_untuned))


    print('*******************************************************************')
    print("The prediction accuracy (tuned) for all testing instances is : {:.2f}%.".format(acc_tuned))
    print("The f1 score (tuned) for all testing instances is : {:.2f}.".format(f1_tuned))

def performance():
    acc_untuned = []
    acc_tuned = []
    f1_untuned = []
    f1_tuned = []
    num_trials = 3

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
        tuned_model, untuned_model, X_test, y_test = train(False, split_data)

        #untuned score
        T_untuned = untuned_model.predict(X_test)
        acc_untuned.append(100*accuracy_score(y_test, T_untuned))
        f1_untuned.append(f1_score(y_test, T_untuned, average='weighted', labels=pd.unique(pd.unique(y_test))))

        #tuned score
        T_tuned = tuned_model.predict(X_test)
        acc_tuned.append(100*accuracy_score(y_test, T_tuned))
        f1_tuned.append(f1_score(y_test, T_tuned,average='weighted', labels=pd.unique(pd.unique(y_test))))
        
        i += 1

    # Take mean of all the trials
    acc_untuned_av = pd.DataFrame(acc_untuned).mean()
    f1_untuned_av =  pd.DataFrame(f1_untuned).mean()

    acc_tuned_av = pd.DataFrame(acc_tuned).mean()
    f1_tuned_av =  pd.DataFrame(f1_tuned).mean()

    print(acc_untuned)
    print('({:d} trials)'.format(num_trials))
    print('*******************************************************************')
    print("The average prediction accuracy (untuned) for all testing instances is : {:.2f}%.".format(acc_untuned_av.iloc[0]))
    print("The average f1 score (untuned) for all testing instances is : {:.2f}.".format(f1_untuned_av.iloc[0]))

    print('*******************************************************************')
    print("The average prediction accuracy (tuned) for all testing instances is : {:.2f}%.".format(acc_tuned_av.iloc[0]))
    print("The average f1 score (tuned) for all testing instances is : {:.2f}.".format(f1_tuned_av.iloc[0]))

    ## Print all this stuff to a file
    headers = ['Model', 'Model Tuning', 'Accuracy (%)', 'f1 (%)']
    model_name = 'logistic_regression'
    untuned_results = pd.DataFrame([[model_name, 'Untuned', acc_untuned_av.iloc[0], f1_untuned_av.iloc[0]]], columns=headers)
    tuned_results = pd.DataFrame([[model_name, 'Tuned', acc_tuned_av.iloc[0], f1_tuned_av.iloc[0]]], columns=headers)

    results = pd.DataFrame(columns=headers)

    results = results.append(untuned_results)
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

