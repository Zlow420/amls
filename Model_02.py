import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn import model_selection as ms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import random as rnd

import sklearn.metrics as metrics
import sklearn.utils

from sklearn.neural_network import MLPRegressor



# Logistic Regression
def logReg(X_train, Y_train, X_test, Y_test):
    logReg = LogisticRegression(max_iter=10000)
    logReg.fit(X_train, Y_train)
    predictedTypes = logReg.predict(X_test)
    print("[LOGISTIC REGRESSION]")
    print('test score: ',logReg.score(X_test, Y_test))
    return logReg

# Support Vector Machines
def svclassifier(X_train, Y_train, X_test, Y_test):
    svm = SVC(decision_function_shape='ovo')
    svm.fit(X_train, Y_train)
    print("[SUPPORT VECTOR MACHINE]")
    print('test score: ',svm.score(X_test, Y_test))
    return logReg

# MLP regression
def MLPreg(X_train, Y_train, X_test, Y_test):
    parameters = [{  # 'hidden_layer_sizes': [3, 5, 10, 100],
        'alpha': [0.01, 0.03, 0.1],
        'activation': ['relu', 'identity']}]
    # 'activation': ['relu','logistic','tanh']}]

    regressor = MLPRegressor(solver="adam", max_iter=10000, activation="relu", alpha=0.01,
                             hidden_layer_sizes=(24, 30, 24))
    # regressor = GridSearchCV(regressor, parameters, verbose=3)
    regressor.fit(X_train, Y_train)
    # best_parameters = regressor.best_params_

    # print ('best parameters:', best_parameters)

    # best parameters: {'activation': 'logistic', 'alpha': 1, 'hidden_layer_sizes': 100}

    regressorResult = regressor.predict(X_test)
    for i in range(0, len(regressorResult)):
        regressorResult[i] = np.round(regressorResult[i])
    #plt.close("all")
    #plt.matshow(metrics.confusion_matrix(Y_test, regressorResult))
    #plt.show()

    trainResult = regressor.predict(X_train)
    regressorScore = np.mean(np.abs(trainResult - Y_train))

    print("Average Distance train: ", regressorScore)

    regressorScore = np.mean(np.abs(regressorResult - Y_test))

    print("Average Distance: ", regressorScore)
    return regressor


