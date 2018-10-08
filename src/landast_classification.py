import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

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
         





    
        
            
        



        
        
        
    