import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

# Argument processing
parser = argparse.ArgumentParser(description='Machine learning algorithm for traffic data. Options are available for re-training the model, testing the model with external data or running performance analysis. Note that only one option should be supplied at a time e.g. you should not run the script with both --test and --re-train.')

parser.add_argument('--test', metavar='TEST_FILE', nargs=1, dest='test_file', help='path to additional features to test against. Path must be relative to the current directory. If supplied, results of predictions against this test data will be the last thing printed by this script (optional)')

parser.add_argument('--re-train', action='store_const', const=True, default=False, help='will re-train the model and document cross validation process')

parser.add_argument('--analysis', action='store_const', const=True, default=False, help='will run the model against test data and report performance')

parser.add_argument('--test-dev', action='store_const', const=True, default=False, help='Will do a train-test and print the results to the command line using the dataset from Canvas. For development purpose only')

parsed_args = parser.parse_args()

DATASET = '../datasets/traffic.csv'

MODEL_DIR = '../model'
MODEL_NAME = 'traffic_model'

RESULTS_DIR = '../results'
RESULTS_NAME = 'traffic_results.csv'

PERFORMANCE_DIR = '../performance'
PERFORMANCE_NAME = 'traffic_performance.csv'

CV_RESULTS_DIR = '../cv_results'
CV_RESULTS_NAME = 'traffic_regression_cv.csv'

target_label='Segment23_(t+1)'

for directory in [MODEL_DIR, RESULTS_DIR, PERFORMANCE_DIR, CV_RESULTS_DIR]:
    if not os.path.isdir(directory):
        print("Making directory: {}".format(directory))
        os.mkdir(directory)

def preprocess(data):
    # With test extern can't guarantee target column will be present
    if target_label in data.columns:
        features = data.drop([target_label], axis=1, inplace=False)
        y = data[target_label]

    full_pipeline = Pipeline([
            ('imputer', Imputer(strategy="median")),
            ('std_scaler', StandardScaler())
        ])

    X = pd.DataFrame(data=full_pipeline.fit_transform(features))
    return X, y

def train(save=True, split_data=None):
    print("Training model")
    data = pd.read_csv(DATASET)
    X, y = preprocess(data)

    if split_data is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    else:
        X_train, X_test, y_train, y_test = split_data

    cv_pipe = Pipeline([
        ('feature_select', SelectKBest(f_regression)),
        ('regression', BayesianRidge())
    ])

    param_range = [ 10**x for x in range(-1, 0)]
    feature_range = [100, 200, 300, 400, 'all']

    param_grid = [
             {'feature_select__k': feature_range,
             'regression__alpha_1': param_range,
             'regression__alpha_2': param_range,
             'regression__lambda_1': param_range,
             'regression__lambda_2': param_range
             },
    ]

    print("Doing cross-validation on BayesianRidge")
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)
    grid_search = GridSearchCV(cv_pipe, param_grid, cv=inner_cv, scoring='neg_mean_squared_error',verbose=1)
    grid_search.fit(X_train, y_train)

    cv_results_ = grid_search.cv_results_
    cv_df = pd.DataFrame(cv_results_)

    with open(os.path.join(CV_RESULTS_DIR, CV_RESULTS_NAME), 'w') as outfile:
        cv_df.to_csv(path_or_buf=outfile, index=False)

    best_params_ = grid_search.best_params_
    print("Best params: {}".format(best_params_))

    tuned_model = grid_search.best_estimator_

    untuned_model = BayesianRidge()
    untuned_model.fit(X_train, y_train)

    if save:
        print("Saving model in {}/{}".format(MODEL_DIR, MODEL_NAME))
        joblib.dump(tuned_model, os.path.join(MODEL_DIR, MODEL_NAME))

    return tuned_model, untuned_model, X_test, y_test

def test_extern(test_file):
    print("Loading model from {}/{}".format(MODEL_DIR, MODEL_NAME))
    model = joblib.load(os.path.join(MODEL_DIR, MODEL_NAME))
    print("Loading test data from {}".format(test_file))
    test_data = pd.read_csv(test_file)

    #ignore return value y as it is not present for testing
    X, _ = preprocess(test_data)

    print("Making predictions")
    prediction = pd.DataFrame(data=model.predict(X), columns=[target_label])

    with open(os.path.join(RESULTS_DIR,RESULTS_NAME), 'w') as outfile:
        print("Printing prediction results in {}".format(outfile.name))
        prediction.to_csv(path_or_buf=outfile, index=False)

def test_dev():
    tuned_model, untuned_model, X_test, y_test = train()

    print("Making test predictions")
    y_pred = tuned_model.predict(X_test)

    regression_object = tuned_model.named_steps['regression']
    print('_____________________TUNED MODEL_____________________ ')
    # The coefficients
    print('Coefficients and Intercept are: ', regression_object.coef_,"   ", regression_object.intercept_,' respectively')
    # The mean squared error
    print('_________________###################____________________')
    print("Mean squared error for testing data: %.2f"
          % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score for testing data: %.2f' % r2_score(y_test, y_pred))
    print('******************************************************* ')

    y_pred = untuned_model.predict(X_test)

    print('_____________________UNTUNED MODEL_____________________ ')
    # The coefficients
    print('Coefficients and Intercept are: ', untuned_model.coef_,"   ", untuned_model.intercept_,' respectively')
    # The mean squared error
    print('_________________###################____________________')
    print("Mean squared error for testing data: %.2f"
          % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score for testing data: %.2f' % r2_score(y_test, y_pred))
    print('******************************************************* ')


def performance():
    mse_untuned = []
    mse_tuned = []
    r2_untuned = []
    r2_tuned = []
    num_trials = 3

    data = pd.read_csv(DATASET)
    X, y = preprocess(data)
    kf = KFold(n_splits=num_trials)
    kf.get_n_splits(X)

    i = 1
    print("Running {} train and evaluate iterations".format(num_trials))
    for train_index, test_index in kf.split(X):
        print("Iteration {} out of {}".format(i, num_trials))
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        split_data = [X_train, X_test, y_train, y_test]
        tuned_model, untuned_model, X_test, y_test = train(False, split_data)

        untuned_prediction = untuned_model.predict(X_test)
        tuned_prediction = tuned_model.predict(X_test)

        mse_untuned.append(mean_squared_error(y_test, untuned_prediction))
        mse_tuned.append(mean_squared_error(y_test, tuned_prediction))

        r2_untuned.append(r2_score(y_test, untuned_prediction))
        r2_tuned.append(r2_score(y_test, tuned_prediction))

        i += 1

    mse_untuned_av = pd.DataFrame(mse_untuned).mean()
    r2_untuned_av =  pd.DataFrame(r2_untuned).mean()

    mse_tuned_av = pd.DataFrame(mse_tuned).mean()
    r2_tuned_av =  pd.DataFrame(r2_tuned).mean()

    print('_________________###################____________________')
    print('({:d} trials)'.format(num_trials))
    print("The average Mean squared error (Untuned) for testing data: %.2f"
          % mse_untuned_av.iloc[0])
    # Explained variance score: 1 is perfect prediction
    print("The average variance (Untuned) for testing data: %.2f"
          % r2_untuned_av.iloc[0])

    print("The average Mean squared error (Tuned) for testing data: %.2f"
          % mse_tuned_av.iloc[0])
    # Explained variance score: 1 is perfect prediction
    print("The average variance (Tuned) for testing data: %.2f"
          % r2_tuned_av.iloc[0])
    print('******************************************************* ')

    with open(os.path.join(PERFORMANCE_DIR, PERFORMANCE_NAME), 'w') as outfile:
        headers = ['Model', 'Model Tuning', 'Mean Squared Error', 'R2 Score']
        model_name = 'bayesian ridge regression'
        untuned_results = pd.DataFrame([[model_name, 'Untuned', mse_untuned_av.iloc[0], r2_untuned_av.iloc[0]]], columns=headers)
        tuned_results = pd.DataFrame([[model_name, 'Tuned', mse_tuned_av.iloc[0], r2_tuned_av.iloc[0]]], columns=headers)

        results = pd.DataFrame(columns=headers)

        results = results.append(untuned_results)
        results = results.append(tuned_results)

        print("Writing results to {}".format(outfile.name))
        results.to_csv(path_or_buf=outfile, index=False)

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
