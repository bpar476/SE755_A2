import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

# Argument processing
parser = argparse.ArgumentParser(description='Machine learning algorithm for 2018 world cup data.')
parser.add_argument('--test', help='path to additional features to test against. Path must be relative to the current directory. If supplied, results of predictions against this test data will be the last thing printed by this script (optional)')
parser.add_argument('--re-train', help='will re-train the model and document cross validation process')
parser.add_argument('--analysis', help='will run the model against test data and report performance')

parsed_args = parser.parse_args()

MODEL_LOCATION = './model/traffic_model'
RESULTS_LOCATION = './results/traffic_results.csv'
PERFORMANCE_LOCATION = './performance/traffic_performance.csv'

target_label='Segment23_(t+1)'

def train(save=True):
    fileName = '../datasets/traffic.csv'
    data = pd.read_csv(fileName)
    features = data.drop([target_label], axis=1, inplace=False)
    target = data[target_label]

    full_pipeline = Pipeline([
            ('imputer', Imputer(strategy="median")),
            ('std_scaler', StandardScaler())
        ])

    processed_features = pd.DataFrame(data=full_pipeline.fit_transform(features))

    X_train, X_test, y_train, y_test = train_test_split(processed_features, target, test_size=0.1, random_state=1)

    param_range = [ 10**x for x in range(-2, 0)]
    param_grid = [
            {'alpha_1': param_range,
             'alpha_2': param_range,
             'lambda_1': param_range,
             'lambda_2': param_range,
             },
    ]

    model = BayesianRidge()
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)
    grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='neg_mean_squared_error',verbose=2)
    grid_search.fit(X_train, y_train)

    cv_results_ = grid_search.cv_results_
    best_params_ = grid_search.best_params_
    tuned_model = grid_search.best_estimator_

    untuned_model = BayesianRidge()
    untuned_model.fit(X_train, y_train)

    if save:
        joblib.dump(tuned_model, MODEL_LOCATION)

    return tuned_model, untuned_model, X_test, y_test

def test_extern(test_file):
    model = joblib.load(MODEL_LOCATION)

    test_data = read_csv(test_file)

    features = test_data.drop([target_label], axis=1, inplace=False)

    full_pipeline = Pipeline([
            ('imputer', Imputer(strategy="median")),
            ('std_scaler', StandardScaler())
        ])

    processed_features = pd.DataFrame(data=full_pipeline.fit_transform(features))

    prediction = pd.DataFrame(data=model.predict(processed_features), columns=[target_label])

    with open(RESULTS_LOCATION, 'w') as outfile:
        prediction.to_csv(path_or_buf=outfile, index=false)

def test():
    tuned_model, untuned_model, X_test, y_test = train()

    y_pred = tuned_model.predict(X_test)

    print('_____________________TUNED MODEL_____________________ ')
    # The coefficients
    print('Coefficients and Intercept are: ', tuned_model.coef_,"   ", tuned_model.intercept_,' respectively')
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

    for x in range(10):
        tuned_model, untuned_model, X_test, y_test = train(False)

        untuned_prediction = untuned_model.predict(X_test)
        tuned_prediction = tuned_model.predict(X_test)

        mse_untuned.append(mean_squared_error(untuned_prediction, y_test))
        mse_tuned.append(mean_squared_error(tuned_prediction, y_test))

        r2_untuned.append(r2_score(untuned_prediction, y_test))
        r2_tuned.append(r2_score(tuned_prediction, y_test))

    with open(PERFORMANCE_LOCATION, 'w') as outfile:
        for x in range(10):
            outfile.write("Iteration {}\n".format(x +1))
            outfile.write("--Mean Squared Error--\n")
            outfile.write("\tUntuned model: {}\n".format(mse_untuned[x]))
            outfile.write("\tTuned model: {}\n".format(mse_tuned[x]))
            outfile.write("--R2 score--\n")
            outfile.write("\tUntuned model: {}\n".format(r2_untuned[x]))
            outfile.write("\tTuned model: {}\n".format(r2_tuned[x]))
            outfile.write("\n")

