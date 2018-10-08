import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

fileName = '../datasets/landsat.csv'
data = pd.read_csv(fileName, header=None)
target_col = 36

features = data.loc[:, :target_col - 1].copy()
target = data.loc[:, target_col].copy()


label_dict = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        7: 5
}

target = target.apply(lambda x: label_dict[x])

full_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])

processed_features = pd.DataFrame(data=full_pipeline.fit_transform(features))
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(features, target):
            train_ind,test_ind = train_index, test_index
            
X_train, X_test = processed_features.loc[train_ind], processed_features.loc[test_ind]
y_train, y_test = target.loc[train_ind], target.loc[test_ind]


labels = [1, 2, 3, 4, 5, 7]
model = keras.Sequential([
        keras.layers.Dense(21, activation=tf.nn.relu),
        keras.layers.Dense(6, activation=tf.nn.softmax)
])
    
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train.values, y_train.values, epochs=5)

test_loss, test_acc = model.evaluate(X_test, y_test)


