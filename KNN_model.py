import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import random


dataset = pd.read_table('data/movies_data_12-18.csv', sep=';')

def append_feature(X, feature, dataset):
    feature_set = list(set(dataset.iloc[:, feature].values))
    feature_org = dataset.iloc[:, feature].values
    feature_con = []
    for entry in feature_org:
        feature_con.append([feature_set.index(entry)])
    X = np.append(X, feature_con, 1)
    return X

def run_knn(dataset, features, neighbors, appendable_features = [], enable_scaler = False, PCA = 0):

    X = dataset.iloc[:, features].values
    y = dataset.iloc[:, 7].values

    for f in appendable_features:
        X = append_feature(X, f, dataset)

    if enable_scaler:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    if PCA != 0:
        pca = decomposition.PCA(n_components=PCA)
        X = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    classifier = KNeighborsRegressor(n_neighbors=neighbors)
    classifier.fit(X_train, y_train)

    y_pred_train = classifier.predict(X_train)
    y_pred = classifier.predict(X_test)

    info = [mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), mean_squared_error(y_train, y_pred_train), r2_score(y_train, y_pred_train)]
    return info

print(run_knn(dataset, [6, 11, 12], 10, [3], True))
