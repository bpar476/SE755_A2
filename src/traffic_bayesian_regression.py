import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

fileName = '../datasets/traffic.csv'
data = pd.read_csv(fileName)
target_label='Segment23_(t+1)'
features = data.drop([target_label], axis=1, inplace=False)
target = data[target_label]

full_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])

processed_features = pd.DataFrame(data=full_pipeline.fit_transform(features))

X_train, X_test, y_train, y_test = train_test_split(processed_features, target, test_size=0.1, random_state=1)

regr = BayesianRidge()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)
y_train_pred = regr.predict(X_train)

print(' ')
# The coefficients
print('Coefficients and Intercept are: ', regr.coef_,"   ",regr.intercept_,' respectively')
# The mean squared error
print('_________________###################____________________')
print("Mean squared error for testing data: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score for testing data: %.2f' % r2_score(y_test, y_pred))
print('******************************************************* ')
print("Mean squared error for training data: %.2f"
      % mean_squared_error(y_train, y_train_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score for training data: %.2f' % r2_score(y_train, y_train_pred))

