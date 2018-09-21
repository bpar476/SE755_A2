# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, cluster

import datetime

# Purity algorithm from StackOverflow user David: https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
def purity(y_true, y_pred):
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

fileName = '../datasets/occupancy.csv'
data = pd.read_csv(fileName)
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

features = data.drop([target_col, 'HumidityRatio', 'date'], axis=1, inplace=False)
target = data[target_col]

full_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])

processed_features = pd.DataFrame(data=full_pipeline.fit_transform(features))

X_train, X_test, y_train, y_test = train_test_split(processed_features, target, test_size=0.1, random_state=1)

## TODO: Implement Cross Validation?

kmeans = KMeans(n_clusters=2, random_state=0)
train_cluster_labels = kmeans.fit_predict(X_train)

pred_labels = kmeans.predict(X_test)

# Explained variance score: 1 is perfect prediction
print('adjusted rand index for testing data: %.2f' % adjusted_rand_score(y_test, pred_labels))
print('******************************************************* ')
print('adjusted rand index for training data: %.2f' % adjusted_rand_score(y_train, train_cluster_labels))
print('******************************************************* ')

print('purity for testing data: %.2f' % purity(y_test, pred_labels))
print('******************************************************* ')
print('purity for training data: %.2f' % purity(y_train, train_cluster_labels))
print('******************************************************* ')