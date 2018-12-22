import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

data = pd.read_table('data/movies_data_12-18.csv', sep=';')

def append_feature(X, feature, dataset):
    feature_set = list(set(dataset.iloc[:, feature].values))
    feature_org = dataset.iloc[:, feature].values
    feature_con = []
    for entry in feature_org:
        feature_con.append([feature_set.index(entry)])
    X = np.append(X, feature_con, 1)
    return X

def neural_network(dataset, features,no_CatFeatures  ,appendable_features = [], enable_scaler = False, PCA = 0):
    X = dataset.iloc[:, features].values
    y = dataset.iloc[:, 7].values

   #for f in appendable_features:
   #     X = append_feature(X, f, dataset)

    #print(X)

    if(no_CatFeatures>0):
        for i in range(no_CatFeatures):
            enc = preprocessing.OneHotEncoder()
            enc.fit(X[:,[0]])
            #print(enc.categories_)
            #print("this here")
            temp = enc.transform(X[:,[0]]).toarray()
            X = np.delete(X,[0],1)
            #print(X)
            X=np.append(X,temp,axis=1)
            #print(X.shape)

    if enable_scaler:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    if PCA != 0:
        pca = decomposition.PCA(n_components=PCA)
        X = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    classifier =MLPRegressor(hidden_layer_sizes=(100,100,100,100,), learning_rate_init=0.1 , alpha= 100 ,max_iter = 20000 , verbose= True, early_stopping= True, tol= 0.0000001)
    #parameters = {'alpha': 10.0** -np.arange(1,7)}
    #gscv = GridSearchCV(classifier,parameters)
    #gscv.fit(X_train,y_train)
    classifier.fit(X_train,y_train)

    #print(gscv.get_params())
    #y_pred_train = gscv.predict(X_train)
    #y_pred = gscv.predict(X_test)
    y_pred_train = classifier.predict(X_train)
    y_pred = classifier.predict(X_test)

    info = [mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), mean_squared_error(y_train, y_pred_train),
            r2_score(y_train, y_pred_train)]
    return info

results= neural_network(data, [3, 6, 11, 12], 1, True)
results2= neural_network(data, [4, 6, 11, 12], 1, True)
results3= neural_network(data, [3, 4, 6, 11, 12], 2, True)

print(results)
print(results2)
print(results3)