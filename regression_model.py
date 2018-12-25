import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd


dataset = pd.read_table('data/movies_data_12-18.csv', sep=';')

features = ['averageRating', 'year', 'votes', 'budget']
target = 'boxOffice'


def run_regression(dataset, features, target, enable_scaler=False):

    X = dataset[features]
    y = dataset[target]

    if enable_scaler:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    algorithms = [
        SVR(kernel='rbf', C=1, gamma=.002, degree=3),
        AdaBoostRegressor(DecisionTreeRegressor(max_depth=15), n_estimators=200),
        GradientBoostingRegressor(loss='huber', n_estimators=200),
        RandomForestRegressor(n_estimators=200)
    ]

    for a in algorithms:
        a.fit(X_train, y_train)
        y_pred_train = a.predict(X_train)
        y_pred = a.predict(X_test)
        print(mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), mean_squared_error(y_train, y_pred_train),
              r2_score(y_train, y_pred_train))

run_regression(dataset, features, target)#, enable_scaler=True)

dataset = dataset.drop(dataset[dataset.boxOffice > 1500000000].index)
data2= dataset.drop(dataset[dataset.budget > 300000000].index)

run_regression(dataset, features, target)#, enable_scaler=True)