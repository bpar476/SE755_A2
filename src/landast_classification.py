import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

fileName = '../datasets/landsat.csv'
data = pd.read_csv(fileName, header=None)
columns_to_keep = [16, 17, 18, 19]
target_col = 36

features = data.loc[:, columns_to_keep].copy()
target = data.loc[:, target_col].copy()

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
         
model = LogisticRegression()

param_grid = [
            {'penalty': ['l1', 'l2'], 'C': [ 2**x for x in range(-6,6) ]},
        ]

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy',verbose=1)
grid_search.fit(X_train, y_train)

cv_results_ = grid_search.cv_results_
best_params_ = grid_search.best_params_
tuned_model = grid_search.best_estimator_

untuned_model = LogisticRegression()
untuned_model.fit(X_train, y_train)

T_untuned = untuned_model.predict(X_test)
T_tuned = tuned_model.predict(X_test)

acc_untuned = 100*accuracy_score(y_test, T_untuned)
acc_tuned = 100*accuracy_score(y_test, T_tuned)

print('*******************************************************************')
print("The average prediction accuracy (untuned) for all testing instances is : {:.2f}%.".format(acc_untuned))

print('*******************************************************************')
print("The average prediction accuracy (tuned) for all testing instances is : {:.2f}%.".format(acc_tuned))






    
        
            
        



        
        
        
    