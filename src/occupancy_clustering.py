# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, cluster
from sklearn.externals import joblib
from sklearn.decomposition import PCA

import datetime
import argparse
import os

# Argument processing
parser = argparse.ArgumentParser(description='Machine learning algorithm for occupancy dataset. Options are available for re-training the model, testing the model with external data or running performance analysis. Note that only one option should be supplied at a time e.g. you should not run the script with both --test and --re-train.')

parser.add_argument('--test', metavar='TEST_FILE', nargs=1, dest='test_file', help='path to additional features to test against. Path must be relative to the current directory. If supplied, results of predictions against this test data will be the last thing printed by this script (optional)')

parser.add_argument('--re-train', action='store_const', const=True, default=False, help='will re-train the model and document cross validation process')

parser.add_argument('--analysis', action='store_const', const=True, default=False, help='will run the model against test data and report performance')

parser.add_argument('--test-dev', action='store_const', const=True, default=False, help='Will do a train-test and print the results to the command line using the dataset from Canvas. For development purpose only')

parsed_args = parser.parse_args()

DATASET = '../datasets/occupancy.csv'

MODEL_DIR = '../model'
MODEL_NAME = 'occupancy_model'

RESULTS_DIR = '../results'
RESULTS_NAME = 'occupancy_results.csv'

PERFORMANCE_DIR = '../performance'
PERFORMANCE_NAME = 'occupancy_both.csv'

KMEAN_PREFIX = 'kmean_'
GMM_PREFIX = 'gmm_'

for directory in [MODEL_DIR, RESULTS_DIR, PERFORMANCE_DIR]:
    if not os.path.isdir(directory):
        print("Making directory: {}".format(directory))
        os.mkdir(directory)

# Purity algorithm from StackOverflow user David: https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
def purity(y_true, y_pred):
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def preprocess(data):
    target_col = 'Occupancy'

    day_of_month = list()
    day_of_week = list()
    hour = list()
    for date_data in data.iloc[:,0].copy():
        date = datetime.datetime.strptime(date_data, "%d/%m/%Y %H:%M")
        day_of_month.append(date.day)
        day_of_week.append(date.weekday())
        hour.append(date.hour)

    data.loc[:, 'day_of_week'] = day_of_week
    data.loc[:, 'day_of_month'] = day_of_month
    data.loc[:, 'hour'] = hour

    y = None

    if not target_col in data.columns:
        cols_to_drop = ['HumidityRatio', 'date']
    else:
        cols_to_drop = [target_col, 'HumidityRatio', 'date']
        y = data[target_col]

    features = data.drop(cols_to_drop, axis=1, inplace=False)

    full_pipeline = Pipeline([
            ('pca', PCA(n_components=2)),
            ('imputer', Imputer(strategy="median")),
            ('std_scaler', StandardScaler())
        ])

    X = pd.DataFrame(data=full_pipeline.fit_transform(features))

    return X, y


def train(save=True, split_data=None):
    if split_data is None:
        data = pd.read_csv(DATASET)
        X, y = preprocess(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    else:
        X_train, X_test, y_train, y_test = split_data

    kmeans = KMeans(n_clusters=2, random_state=0)
    train_cluster_labels = kmeans.fit_predict(X_train)

    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(X_train)

    if save:
        kmean_model_file = KMEAN_PREFIX + MODEL_NAME
        gmm_model_file = GMM_PREFIX + MODEL_NAME
        print("Saving kmean model in {}/{}".format(MODEL_DIR, kmean_model_file))
        print("Saving gmm model in {}/{}".format(MODEL_DIR, gmm_model_file))
        joblib.dump(kmeans, os.path.join(MODEL_DIR, kmean_model_file))
        joblib.dump(gmm, os.path.join(MODEL_DIR, gmm_model_file))

    return kmeans, gmm, X_test, y_test

def test_extern(test_file):
    print("Loading kmean model from {}/{}".format(MODEL_DIR, KMEAN_PREFIX + MODEL_NAME))
    kmeans = joblib.load(os.path.join(MODEL_DIR, KMEAN_PREFIX + MODEL_NAME))
    print("Loading gmm model from {}/{}".format(MODEL_DIR, GMM_PREFIX + MODEL_NAME))
    gmm = joblib.load(os.path.join(MODEL_DIR, GMM_PREFIX + MODEL_NAME))

    print("Loading test data from {}".format(test_file))
    test_data = pd.read_csv(test_file)

    X, _ = preprocess(test_data)

    kmean_prediction = kmeans.predict(X)
    gmm_prediction = gmm.predict(X)

    data = np.array([kmean_prediction, gmm_prediction])
    prediction = pd.DataFrame(data=data.transpose(), columns=['K-Mean Prediction', 'GMM Prediction'])

    with open(os.path.join(RESULTS_DIR,RESULTS_NAME), 'w') as outfile:
        print("Outputting results to {}".format(outfile.name))
        prediction.to_csv(path_or_buf=outfile, index=False)

def test_dev():
    kmeans, gmm, X_test, y_test = train()

    kmeans_pred = kmeans.predict(X_test)

    print('###### K-MEANS MODEL #######')
    print('adjusted rand index for testing data: %.2f' % adjusted_rand_score(y_test, kmeans_pred))
    print('******************************************************* ')
    print('purity for testing data: %.2f' % purity(y_test, kmeans_pred))
    print('******************************************************* ')

    gmm_pred = gmm.predict(X_test)

    # Explained variance score: 1 is perfect prediction
    print('###### GAUSSIAN MIXTURE MODEL #######')
    print('adjusted rand index for testing data: %.2f' % adjusted_rand_score(y_test, gmm_pred))
    print('******************************************************* ')

    print('purity for testing data: %.2f' % purity(y_test, gmm_pred))
    print('******************************************************* ')

def performance():
    randind_kmean = []
    purity_kmean = []
    randind_gmm = []
    purity_gmm = []
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
        kmean, gmm, X_test, y_test = train(False, split_data)

        kmean_pred = kmean.predict(X_test)
        gmm_pred = gmm.predict(X_test)

        randind_kmean.append(adjusted_rand_score(y_test, kmean_pred))
        randind_gmm.append(adjusted_rand_score(y_test, gmm_pred))

        purity_kmean.append(purity(y_test, kmean_pred))
        purity_gmm.append(purity(y_test, gmm_pred))

        i += 1

    randind_gmm_av = pd.DataFrame(randind_gmm).mean()
    purity_gmm_av =  pd.DataFrame(purity_gmm).mean()

    randind_kmean_av = pd.DataFrame(randind_kmean).mean()
    purity_kmean_av =  pd.DataFrame(purity_kmean).mean()

    print('({:d} trials)'.format(num_trials))
    print('_________________K-Means Model____________________')
    print("The average adjusted rand score for testing data: %.2f"
          % randind_kmean_av.iloc[0])
    print("The average purity for testing data: %.2f"
          % purity_kmean_av.iloc[0])

    print('_________________GMM Model____________________')
    print("The average adjusted rand score for testing data: %.2f"
          % randind_gmm_av.iloc[0])
    print("The average purity for testing data: %.2f"
          % purity_gmm_av.iloc[0])
    print('******************************************************* ')

    with open(os.path.join(PERFORMANCE_DIR, PERFORMANCE_NAME), 'w') as outfile:
        headers = ['Model', 'Adjusted Rand Score', 'Purity']
        kmean_results = pd.DataFrame([['K-Means', randind_kmean_av.iloc[0], purity_kmean_av.iloc[0]]], columns=headers)
        gmm_results = pd.DataFrame([['Guassian Mixture', randind_gmm_av.iloc[0], purity_gmm_av.iloc[0]]], columns=headers)

        results = pd.DataFrame(columns=headers)

        results = results.append(kmean_results)
        results = results.append(gmm_results)

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
